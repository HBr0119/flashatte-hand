#!/usr/bin/env python3
"""Test v4 with single block launch to check multi-block interaction."""
import os
os.environ.setdefault("MACA_PATH", "/opt/maca")
os.environ["PATH"] = f"/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch
import math
import numpy as np
from torch.utils.cpp_extension import load

BINDING_CPP = """
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <stdint.h>

extern "C" void run_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache_paged,
    const __nv_bfloat16* v_cache_paged,
    __nv_bfloat16* output,
    const int32_t* cache_seqlens,
    const int32_t* block_table,
    int64_t batch_size,
    int64_t seqlen_k,
    int64_t seqlen_q,
    int64_t num_heads,
    int64_t num_heads_k,
    int64_t headdim,
    int64_t page_block_size,
    int64_t num_blocks,
    int64_t causal);

void run_kernel_torch(
    torch::Tensor q,
    torch::Tensor k_cache_paged,
    torch::Tensor v_cache_paged,
    torch::Tensor output,
    torch::Tensor cache_seqlens,
    torch::Tensor block_table,
    int64_t batch_size,
    int64_t seqlen_k,
    int64_t seqlen_q,
    int64_t num_heads,
    int64_t num_heads_k,
    int64_t headdim,
    int64_t page_block_size,
    int64_t num_blocks,
    int64_t causal) {
    run_kernel(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache_paged.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache_paged.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        cache_seqlens.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        batch_size,
        seqlen_k,
        seqlen_q,
        num_heads,
        num_heads_k,
        headdim,
        page_block_size,
        num_blocks,
        causal);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_kernel", &run_kernel_torch, "FlashAttention KVCACHE run_kernel wrapper");
}
"""

# ---- Build a custom single-block launcher as separate module ----
CUSTOM_SRC = """
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Forward declare the template kernel from v4
template <int NUM_KV_HEADS>
__global__ void paged_decode_gqa_mma_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch);

// Launch exactly 1 block, targeting a specific kv_head of a specific batch
void launch_single_block(
    torch::Tensor q,
    torch::Tensor k_cache_paged,
    torch::Tensor v_cache_paged,
    torch::Tensor output,
    torch::Tensor cache_seqlens,
    torch::Tensor block_table,
    int blocks_per_batch,
    int block_idx  // which block to launch (0 = first batch, kv_head=0)
) {
    paged_decode_gqa_mma_kernel<4><<<1, 64>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache_paged.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache_paged.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        cache_seqlens.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        blocks_per_batch);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_single_block", &launch_single_block);
}
"""

import tempfile
build_dir = "/tmp/smoke_v4_single_build"
os.makedirs(build_dir, exist_ok=True)

# Build custom launcher that includes v4.cu
print("Compiling single-block launcher...")
mod = load(
    name="flashattn_v4_single",
    sources=[
        "/root/flashattention-c500-agent-oj-five/submission-oj/flashattn_kvcache_decode_mma_v4.cu",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    build_directory=build_dir,
    verbose=False,
)

# Manually build a separate module for the single-block launcher
# that links with v4's code
# Actually, we can't link separately. Let me embed the launcher.

# Alternative: just use the full run_kernel but with batch_size=1
print("Testing with batch_size=1 in dispatch...")

batch_size = 1
seqlen_k = 141
seqlen_q = 1
NUM_KV_HEADS = 4
NUM_HEADS = 32
HEAD_DIM = 128
PAGE_SIZE = 16

blocks_per_batch = math.ceil(seqlen_k / PAGE_SIZE)
num_blocks = batch_size * blocks_per_batch

torch.manual_seed(0)
q = torch.randn(batch_size, seqlen_q, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
k_cache_paged = torch.randn(num_blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
v_cache_paged = torch.randn(num_blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

block_table = torch.randperm(num_blocks, dtype=torch.int32, device="cuda").reshape(batch_size, -1)
cache_seqlens = torch.tensor([seqlen_k], dtype=torch.int32, device="cuda")

output = torch.zeros(batch_size, seqlen_q, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

mod.run_kernel(q, k_cache_paged, v_cache_paged, output, cache_seqlens, block_table,
               batch_size, seqlen_k, seqlen_q, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
               PAGE_SIZE, num_blocks, 0)

torch.cuda.synchronize()

out_np = output.cpu().float().numpy()
has_nan = np.isnan(out_np).any()
has_inf = np.isinf(out_np).any()
print(f"v4 batch_size=1 kv={seqlen_k}: NaN={has_nan}, Inf={has_inf}")
print(f"Output stats: min={out_np.min():.6f}, max={out_np.max():.6f}, mean={out_np.mean():.6f}")

if has_nan:
    print("FAIL: NaN even with batch_size=1!")
else:
    print("PASS: No NaN with batch_size=1!")

#!/usr/bin/env python3
"""Smoke test (5 cases) for v10 — Full McFlashInfer restoration."""
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

build_dir = "/tmp/smoke_v10_build"
os.makedirs(build_dir, exist_ok=True)
with open(os.path.join(build_dir, "binding.cpp"), "w") as f:
    f.write(BINDING_CPP)

print("Compiling v10...")
mod = load(
    name="flashattn_smoke_v10",
    sources=[
        os.path.join(build_dir, "binding.cpp"),
        "/root/flashattention-c500-agent-oj-five/submission-oj/flashattn_kvcache_decode_v10.cu",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    build_directory=build_dir,
    verbose=True,
)
print("Compilation done.\n")

# ── Test Cases ──────────────────────────────────────────────────────────
# 5 cases: short, medium, long KV; GQA=8 and GQA=4
test_cases = [
    # (batch, seqlen_kv, num_kv_heads, name)
    (8,  141,    4, "bs8_kv141_g4"),
    (4,  2048,   4, "bs4_kv2048_g4"),
    (2,  4096,   8, "bs2_kv4096_g8"),
    (1,  8192,   4, "bs1_kv8192_g4"),  # triggers num_splits
    (16, 512,    8, "bs16_kv512_g8"),
]

HEAD_DIM = 128
PAGE_SIZE = 16
results = []

for batch_size, seqlen_k, nkh, name in test_cases:
    nh = 32
    blocks_per_batch = math.ceil(seqlen_k / PAGE_SIZE)
    num_blocks = batch_size * blocks_per_batch

    torch.manual_seed(42)
    q = torch.randn(batch_size, 1, nh, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_cache_paged = torch.randn(num_blocks, PAGE_SIZE, nkh, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v_cache_paged = torch.randn(num_blocks, PAGE_SIZE, nkh, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    block_table = torch.randperm(num_blocks, dtype=torch.int32, device="cuda").reshape(batch_size, -1)
    vals = [seqlen_k, max(1, seqlen_k - 1)]
    for b_idx in range(2, batch_size):
        span = max(1, seqlen_k - 1)
        vals.append(1 + ((b_idx * 9973 + seqlen_k // 3) % span))
    cache_seqlens = torch.tensor(vals, dtype=torch.int32, device="cuda")
    output = torch.zeros(batch_size, 1, nh, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    mod.run_kernel(q, k_cache_paged, v_cache_paged, output, cache_seqlens, block_table,
                   batch_size, seqlen_k, 1, nh, nkh, HEAD_DIM,
                   PAGE_SIZE, num_blocks, 0)
    torch.cuda.synchronize()

    out_np = output.cpu().float().numpy()
    has_nan = np.isnan(out_np).any()
    has_inf = np.isinf(out_np).any()
    status = "PASS" if not has_nan and not has_inf else "FAIL"
    print(f"  [{status}] {name}: NaN={has_nan}, Inf={has_inf}, min={out_np.min():.4f}, max={out_np.max():.4f}, mean={out_np.mean():.6f}")
    results.append((status, name))

print()
n_pass = sum(1 for s, _ in results if s == "PASS")
print(f"Smoke test: {n_pass}/{len(results)} passed")
if n_pass == len(results):
    print("ALL PASSED")
else:
    print("SOME FAILED!")

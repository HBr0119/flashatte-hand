#!/usr/bin/env python3
"""Reproduce eval case for v3: batch_size=16, seqlen_k=141"""
import os
os.environ.setdefault("MACA_PATH", "/opt/maca")
os.environ["PATH"] = f"/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch, math, numpy as np
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
    torch::Tensor q, torch::Tensor k_cache_paged, torch::Tensor v_cache_paged,
    torch::Tensor output, torch::Tensor cache_seqlens, torch::Tensor block_table,
    int64_t batch_size, int64_t seqlen_k, int64_t seqlen_q,
    int64_t num_heads, int64_t num_heads_k, int64_t headdim,
    int64_t page_block_size, int64_t num_blocks, int64_t causal) {
    run_kernel(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache_paged.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache_paged.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        cache_seqlens.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
        batch_size, seqlen_k, seqlen_q, num_heads, num_heads_k, headdim,
        page_block_size, num_blocks, causal);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_kernel", &run_kernel_torch, "FlashAttention KVCACHE run_kernel wrapper");
}
"""

build_dir = "/tmp/smoke_v3_bs16_build"
os.makedirs(build_dir, exist_ok=True)
with open(os.path.join(build_dir, "binding.cpp"), "w") as f: f.write(BINDING_CPP)

print("Compiling v3...")
mod = load(name="flashattn_smoke_v3_bs16",
    sources=[os.path.join(build_dir, "binding.cpp"),
             "/root/flashattention-c500-agent-oj-five/submission-oj/flashattn_kvcache_decode_mma_v3.cu"],
    extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], build_directory=build_dir, verbose=False)
print("Done.")

batch_size = 16; seqlen_k = 141; seqlen_q = 1
NUM_KV_HEADS = 4; NUM_HEADS = 32; HEAD_DIM = 128; PAGE_SIZE = 16
blocks_per_batch = math.ceil(seqlen_k / PAGE_SIZE)
num_blocks = batch_size * blocks_per_batch

torch.manual_seed(0)
q = torch.randn(batch_size, seqlen_q, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
k = torch.randn(num_blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
v = torch.randn(num_blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
bt = torch.randperm(num_blocks, dtype=torch.int32, device="cuda").reshape(batch_size, -1)
vals = [seqlen_k, 1]
for i in range(2, batch_size):
    span = max(1, seqlen_k - 1)
    vals.append(1 + ((i * 9973 + seqlen_k // 3) % span))
sl = torch.tensor(vals, dtype=torch.int32, device="cuda")
o = torch.zeros(batch_size, seqlen_q, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

mod.run_kernel(q, k, v, o, sl, bt, batch_size, seqlen_k, seqlen_q, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, num_blocks, 0)
torch.cuda.synchronize()

onp = o.cpu().float().numpy()
hn = np.isnan(onp).any(); hi = np.isinf(onp).any()
for b in range(batch_size):
    bn = np.isnan(onp[b]).any(); bi = np.isinf(onp[b]).any()
    if bn or bi: print(f"  Batch {b}: NaN={bn}, Inf={bi}")
print(f"v3 bs=16 kv=141: NaN={hn}, Inf={hi}, min={onp.min():.6f}, max={onp.max():.6f}")

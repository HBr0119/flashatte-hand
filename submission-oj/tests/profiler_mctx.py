#!/usr/bin/env python3
"""Run smoke test under mcTracer to get hardware counters"""
import os, subprocess, sys

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
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache_paged, const __nv_bfloat16* v_cache_paged,
    __nv_bfloat16* output, const int32_t* cache_seqlens, const int32_t* block_table,
    int64_t batch_size, int64_t seqlen_k, int64_t seqlen_q,
    int64_t num_heads, int64_t num_heads_k, int64_t headdim,
    int64_t page_block_size, int64_t num_blocks, int64_t causal);
void run_kernel_torch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor o, torch::Tensor sl, torch::Tensor bt,
    int64_t bs, int64_t sk, int64_t sq, int64_t nh, int64_t nhk, int64_t hd,
    int64_t pbs, int64_t nb, int64_t c) {
    run_kernel((const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k.data_ptr(),
        (const __nv_bfloat16*)v.data_ptr(), (__nv_bfloat16*)o.data_ptr(),
        sl.data_ptr<int32_t>(), bt.data_ptr<int32_t>(),
        bs, sk, sq, nh, nhk, hd, pbs, nb, c);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_kernel", &run_kernel_torch, "v5 profiler");
}
"""

d = "/tmp/profiler_build"
os.makedirs(d, exist_ok=True)
with open(os.path.join(d, "binding.cpp"), "w") as f:
    f.write(BINDING_CPP)

print("Compiling v5 for profiler...")
mod = load(
    name="v5_profiler",
    sources=[os.path.join(d, "binding.cpp"),
             "/root/flashattention-c500-agent-oj-five/submission-oj/flashattn_kvcache_decode_mma_v5.cu"],
    extra_cflags=["-O3"], extra_cuda_cflags=["-O3"],
    build_directory=d, verbose=False)
print("Done.")

# Test parameters
bs, sk, NH, HD, PS = 8, 141, 32, 128, 16
NUM_KV = 4
bpb = math.ceil(sk / PS)
nb = bs * bpb

torch.manual_seed(0)
q = torch.randn(bs, 1, NH, HD, device="cuda", dtype=torch.bfloat16)
k = torch.randn(nb, PS, NUM_KV, HD, device="cuda", dtype=torch.bfloat16)
v = torch.randn(nb, PS, NUM_KV, HD, device="cuda", dtype=torch.bfloat16)
bt = torch.randperm(nb, dtype=torch.int32, device="cuda").reshape(bs, -1)
vals = [sk, 1]
[vals.append(1 + ((i * 9973 + sk // 3) % max(1, sk - 1))) for i in range(2, bs)]
sl = torch.tensor(vals, dtype=torch.int32, device="cuda")
o = torch.zeros(bs, 1, NH, HD, device="cuda", dtype=torch.bfloat16)

# Warmup
for _ in range(3):
    mod.run_kernel(q, k, v, o, sl, bt, bs, sk, 1, NH, NUM_KV, HD, PS, nb, 0)
torch.cuda.synchronize()

# Profiled run
torch.cuda.profiler.start()
for _ in range(100):
    mod.run_kernel(q, k, v, o, sl, bt, bs, sk, 1, NH, NUM_KV, HD, PS, nb, 0)
torch.cuda.synchronize()
torch.cuda.profiler.stop()

onp = o.cpu().float().numpy()
print(f"v5 bs={bs} kv={sk}: NaN={np.isnan(onp).any()}, max={onp.max():.4f}")
print("Profiling complete.")

#!/usr/bin/env python3
"""GPU diagnostic: compare MMA P×V vs scalar P×V for a single block."""
import os
os.environ.setdefault("MACA_PATH", "/opt/maca")
os.environ["PATH"] = f"/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import numpy as np
import torch
from torch.utils.cpp_extension import load

mod = load(name="pv_mma_diag", sources=["/tmp/diag_pv_mma.cu"], verbose=True)

print("Running diagnostic...")

kGroup = 8
PAGE_SIZE = 16
HEAD_DIM = 128

rng = np.random.RandomState(42)
prob_host = rng.rand(kGroup, PAGE_SIZE).astype(np.float32)
prob_host = prob_host / prob_host.sum(axis=1, keepdims=True)
v_host = rng.randn(PAGE_SIZE, HEAD_DIM).astype(np.float32) * 0.5

prob_t = torch.from_numpy(prob_host).cuda()
v_bf16 = torch.from_numpy(v_host).bfloat16().cuda()

out_mma = torch.zeros(kGroup * HEAD_DIM, dtype=torch.float32).cuda()
out_scalar = torch.zeros(kGroup * HEAD_DIM, dtype=torch.float32).cuda()

mod.launch_diag(out_mma, out_scalar, prob_t, v_bf16)
torch.cuda.synchronize()

mma_np = out_mma.cpu().numpy().reshape(kGroup, HEAD_DIM)
scalar_np = out_scalar.cpu().numpy().reshape(kGroup, HEAD_DIM)

has_nan_mma = np.isnan(mma_np).any()
has_nan_scalar = np.isnan(scalar_np).any()
diff = np.abs(mma_np - scalar_np)
max_err = diff.max()

print(f"MMA NaN: {has_nan_mma}, Scalar NaN: {has_nan_scalar}")
print(f"Max abs error (MMA vs Scalar): {max_err}")
print(f"MMA sample[0,:8]:    {mma_np[0,:8]}")
print(f"Scalar sample[0,:8]: {scalar_np[0,:8]}")

cpu_ref = np.zeros((kGroup, HEAD_DIM), dtype=np.float32)
for h in range(kGroup):
    for t in range(PAGE_SIZE):
        for d in range(HEAD_DIM):
            cpu_ref[h, d] += prob_host[h, t] * v_host[t, d]

cpu_diff_mma = np.abs(mma_np - cpu_ref).max()
cpu_diff_scalar = np.abs(scalar_np - cpu_ref).max()
print(f"CPU ref vs MMA max err: {cpu_diff_mma}")
print(f"CPU ref vs Scalar max err: {cpu_diff_scalar}")

if cpu_diff_mma > 1e-3:
    print("\nMMA WRONG! First 4x8 of MMA vs CPU:")
    print("MMA[0:4,0:8]:")
    print(mma_np[:4,:8])
    print("CPU[0:4,0:8]:")
    print(cpu_ref[:4,:8])

print("Done.")

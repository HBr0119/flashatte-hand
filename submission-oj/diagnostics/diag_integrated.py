#!/usr/bin/env python3
"""Integrated diagnostic: QK MMA → softmax → MMA P×V vs Scalar P×V in same kernel."""
import os
os.environ.setdefault("MACA_PATH", "/opt/maca")
os.environ["PATH"] = f"/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import numpy as np
import torch
from torch.utils.cpp_extension import load

mod = load(name="integrated_mma_diag", sources=["/tmp/diag_integrated.cu"], verbose=True)

print("Running integrated diagnostic...")

kGroup = 8
PAGE_SIZE = 16
HEAD_DIM = 128

rng = np.random.RandomState(42)
q_host = rng.randn(kGroup, HEAD_DIM).astype(np.float32) * 0.1
k_host = rng.randn(PAGE_SIZE, HEAD_DIM).astype(np.float32) * 0.1
v_host = rng.randn(PAGE_SIZE, HEAD_DIM).astype(np.float32) * 0.1

q_bf16 = torch.from_numpy(q_host).bfloat16().cuda()
k_bf16 = torch.from_numpy(k_host).bfloat16().cuda()
v_bf16 = torch.from_numpy(v_host).bfloat16().cuda()

out_mma = torch.zeros(kGroup * HEAD_DIM, dtype=torch.float32).cuda()
out_scalar = torch.zeros(kGroup * HEAD_DIM, dtype=torch.float32).cuda()

mod.launch_integrated_diag(out_mma, out_scalar, q_bf16, k_bf16, v_bf16)
torch.cuda.synchronize()

mma_np = out_mma.cpu().numpy().reshape(kGroup, HEAD_DIM)
scalar_np = out_scalar.cpu().numpy().reshape(kGroup, HEAD_DIM)

has_nan_mma = np.isnan(mma_np).any()
has_nan_scalar = np.isnan(scalar_np).any()
diff = np.abs(mma_np - scalar_np)
max_err = diff.max()

print(f"MMA NaN: {has_nan_mma}, Scalar NaN: {has_nan_scalar}")
print(f"Max abs error (MMA vs Scalar): {max_err}")
print(f"MMA[0,:8]:    {mma_np[0,:8]}")
print(f"Scalar[0,:8]: {scalar_np[0,:8]}")

# Also compute CPU reference (full pipeline)
scale = 0.08838834764831845
# QK dot product
scores = q_host @ k_host.T * scale  # [8 heads, 16 tokens]
# softmax
scores_max = scores.max(axis=1, keepdims=True)
probs = np.exp(scores - scores_max)
probs /= probs.sum(axis=1, keepdims=True)
# P×V
cpu_ref = probs @ v_host  # [8 heads, 128 dims]

cpu_diff_mma = np.abs(mma_np - cpu_ref).max()
cpu_diff_scalar = np.abs(scalar_np - cpu_ref).max()
print(f"CPU ref vs MMA max err: {cpu_diff_mma}")
print(f"CPU ref vs Scalar max err: {cpu_diff_scalar}")

# Check if values look reasonable
print(f"\nMMA max abs value: {np.abs(mma_np).max()}")
print(f"Scalar max abs value: {np.abs(scalar_np).max()}")

if cpu_diff_mma > 0.01:
    print("\nMMA significant deviation detected!")
    # Find the worst element
    flat_idx = np.argmax(np.abs(mma_np - cpu_ref))
    h, d = divmod(flat_idx, HEAD_DIM)
    print(f"Worst at head={h}, dim={d}: MMA={mma_np[h,d]}, CPU={cpu_ref[h,d]}")

print("Done.")

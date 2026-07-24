#!/usr/bin/env python3
"""Minimal multi-page test for v4 kernel: kv=33 (2 full pages + 1 token)."""
import os, sys
sys.path.insert(0, "/root/flashattention-c500-agent-oj-five")
os.environ.setdefault("MACA_PATH", "/opt/maca")
os.environ["PATH"] = f"/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Import the evaluation module
from scripts.evaluate_flashattn_submission import main as eval_main

# Run just a specific shape
import subprocess, json

# Use the OJ wrapper to test specific shape
cmd = [
    sys.executable, 
    "scripts/evaluate_flashattn_submission_oj.py",
    "--src", "submission-oj/flashattn_kvcache_decode_mma_v4.cu",
    "--out", "/tmp/mma_v4_33",
]
# Can we pass specific shapes? Let me check...
print("Running v4 with kv=33 test through smoke+")


# Actually let me just write a quick inline test
import torch
from torch.utils.cpp_extension import load
import numpy as np

# Use the already-compiled extension path from the evaluation system
# Actually, just run evaluate with --smoke first to see if it compiles, then modify

print("Running kv=33 as a specific test - using OJ eval but with limited shapes")
# The smoke test already includes kv=17 which is single-page
# Let me just run the smoke and see

# Actually, let me just run the full test and focus on the results
# The full test already showed kv=141 (9 pages), kv=362, kv=2048 fail
# Let me add a minimal test at kv=33

# Create a quick test
import tempfile

# Read the example shapes from the test runner
# Let me just use existing infrastructure
result = subprocess.run(
    [sys.executable, "scripts/evaluate_flashattn_submission_oj.py",
     "--src", "submission-oj/flashattn_kvcache_decode_mma_v4.cu",
     "--out", "/tmp/mma_v4_full2",
     "--task-profile", "contest"],
    capture_output=True, text=True, timeout=120
)

for line in result.stdout.strip().split('\n'):
    try:
        d = json.loads(line)
        if 'ok' in d:
            print(f"  batch={d.get('batch_size')}, kv={d.get('seqlen_kv')}, hk={d.get('num_heads_k')}: {'PASS' if d['ok'] else 'FAIL'} err={d.get('max_abs_err')}")
    except json.JSONDecodeError:
        pass

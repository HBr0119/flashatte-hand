#!/usr/bin/env python3
"""Roofline experiment: measure kernel time with and without P×V"""
import os, time
os.environ.setdefault("MACA_PATH", "/opt/maca")
os.environ["PATH"] = f"/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch, math
from torch.utils.cpp_extension import load

BINDING = """
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <stdint.h>
extern "C" void run_kernel(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* o,const int32_t* sl,const int32_t* bt,int64_t bs,int64_t sk,int64_t sq,int64_t nh,int64_t nhk,int64_t hd,int64_t pbs,int64_t nb,int64_t c);
void run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor o,torch::Tensor sl,torch::Tensor bt,int64_t bs,int64_t sk,int64_t sq,int64_t nh,int64_t nhk,int64_t hd,int64_t pbs,int64_t nb,int64_t c){
  run_kernel((const __nv_bfloat16*)q.data_ptr(),(const __nv_bfloat16*)k.data_ptr(),(const __nv_bfloat16*)v.data_ptr(),(__nv_bfloat16*)o.data_ptr(),sl.data_ptr<int32_t>(),bt.data_ptr<int32_t>(),bs,sk,sq,nh,nhk,hd,pbs,nb,c);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run,"");}
"""

def benchmark(mod, bs, sk, label, warmup=3, repeat=20):
    NUM_KV=4; NH=32; HD=128; PS=16
    bpb=math.ceil(sk/PS); nb=bs*bpb
    torch.manual_seed(0)
    q=torch.randn(bs,1,NH,HD,device="cuda",dtype=torch.bfloat16)
    k=torch.randn(nb,PS,NUM_KV,HD,device="cuda",dtype=torch.bfloat16)
    v=torch.randn(nb,PS,NUM_KV,HD,device="cuda",dtype=torch.bfloat16)
    bt=torch.randperm(nb,dtype=torch.int32,device="cuda").reshape(bs,-1)
    vals=[sk,1]; [vals.append(1+((i*9973+sk//3)%max(1,sk-1))) for i in range(2,bs)]
    sl=torch.tensor(vals,dtype=torch.int32,device="cuda")
    o=torch.zeros(bs,1,NH,HD,device="cuda",dtype=torch.bfloat16)
    for _ in range(warmup): mod.run(q,k,v,o,sl,bt,bs,sk,1,NH,NUM_KV,HD,PS,nb,0)
    torch.cuda.synchronize()
    t0=time.time()
    for _ in range(repeat):
        mod.run(q,k,v,o,sl,bt,bs,sk,1,NH,NUM_KV,HD,PS,nb,0)
    torch.cuda.synchronize()
    elapsed=(time.time()-t0)/repeat*1000
    print(f"{label:20s} bs={bs:2d} kv={sk:4d}: {elapsed:.4f}ms")

d="/tmp/roofline_build"; os.makedirs(d,exist_ok=True)
with open(os.path.join(d,"b.cpp"),"w") as f: f.write(BINDING)

# Benchmark v2 and v5
for ver,src in [("v2","flashattn_kvcache_decode_mma_v2.cu"),("v5","flashattn_kvcache_decode_mma_v5.cu")]:
    print(f"\n=== {ver} ===")
    mod=load(name=f"rl_{ver}",sources=[os.path.join(d,"b.cpp"),f"/root/flashattention-c500-agent-oj-five/submission-oj/{src}"],extra_cflags=["-O3"],extra_cuda_cflags=["-O3"],build_directory=d,verbose=False)
    for bs,sk in [(8,141),(16,141),(64,2048),(16,17),(4,2)]:
        benchmark(mod,bs,sk,f"{ver}")

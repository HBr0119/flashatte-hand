import os, subprocess
os.environ["MACA_PATH"] = "/opt/maca"
os.environ["PATH"] = f"/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:{os.environ.get('PATH','')}"
os.environ["LD_LIBRARY_PATH"] = f"/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:{os.environ.get('LD_LIBRARY_PATH','')}"

import torch, math

BINDING = """
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <stdint.h>
extern "C" void run_kernel(const __nv_bfloat16*,const __nv_bfloat16*,const __nv_bfloat16*,__nv_bfloat16*,const int32_t*,const int32_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
void run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor o,torch::Tensor sl,torch::Tensor bt,int64_t bs,int64_t sk,int64_t sq,int64_t nh,int64_t nhk,int64_t hd,int64_t pbs,int64_t nb,int64_t c){
  run_kernel((const __nv_bfloat16*)q.data_ptr(),(const __nv_bfloat16*)k.data_ptr(),(const __nv_bfloat16*)v.data_ptr(),(__nv_bfloat16*)o.data_ptr(),sl.data_ptr<int32_t>(),bt.data_ptr<int32_t>(),bs,sk,sq,nh,nhk,hd,pbs,nb,c);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run,"");}
"""

from torch.utils.cpp_extension import load
d="/tmp/trace_build"; os.makedirs(d,exist_ok=True)
with open(os.path.join(d,"b.cpp"),"w") as f: f.write(BINDING)

mod=load(name="trace_v5",sources=[os.path.join(d,"b.cpp"),"/root/flashattention-c500-agent-oj-five/submission-oj/flashattn_kvcache_decode_mma_v5.cu"],extra_cflags=["-O3"],extra_cuda_cflags=["-O3"],build_directory=d,verbose=False)

bs,sk,NH,HD,PS=8,141,32,128,16
NUM_KV=4; bpb=math.ceil(sk/PS); nb=bs*bpb
torch.manual_seed(0)
q=torch.randn(bs,1,NH,HD,device="cuda",dtype=torch.bfloat16)
k=torch.randn(nb,PS,NUM_KV,HD,device="cuda",dtype=torch.bfloat16)
v=torch.randn(nb,PS,NUM_KV,HD,device="cuda",dtype=torch.bfloat16)
bt=torch.randperm(nb,dtype=torch.int32,device="cuda").reshape(bs,-1)
sl=torch.tensor([sk,1,50,20,30,10,5,40],dtype=torch.int32,device="cuda")
o=torch.zeros(bs,1,NH,HD,device="cuda",dtype=torch.bfloat16)

# Warmup
for _ in range(3): mod.run(q,k,v,o,sl,bt,bs,sk,1,NH,NUM_KV,HD,PS,nb,0)
torch.cuda.synchronize()

# Profile with torch profiler
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) as prof:
    for _ in range(10):
        mod.run(q,k,v,o,sl,bt,bs,sk,1,NH,NUM_KV,HD,PS,nb,0)
    torch.cuda.synchronize()

for evt in prof.key_averages():
    if evt.key != "ProfilerStep":
        print(f"{evt.key:50s} count={evt.count:3d}  device_time={evt.device_time:.1f}us")

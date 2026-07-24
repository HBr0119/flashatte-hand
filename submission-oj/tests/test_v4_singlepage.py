#!/usr/bin/env python3
"""Single-page test for v4 MMA P×V kernel."""
import os
os.environ.setdefault("MACA_PATH", "/opt/maca")
os.environ["PATH"] = f"/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch
from torch.utils.cpp_extension import load

SRC = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>
#include <torch/extension.h>

typedef _Float16 maca_half4 __attribute__((__vector_size__(8)));
typedef float maca_float4 __attribute__((__vector_size__(16)));

union alignas(8) mma_bf16_fragment_t {
    maca_half4 vector;
    __nv_bfloat16 element[4];
};
union alignas(16) mma_float_fragment_t {
    maca_float4 vector;
    float element[4];
};

#define MMA_TILE 16
#define MMA_THREADS 64
#define HEAD_DIM 128
#define PAGE_SIZE 16
#define MMA_K_TILES (HEAD_DIM / MMA_TILE)

template <int NUM_KV_HEADS>
__global__ void single_page_test(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_data,
    const __nv_bfloat16* __restrict__ v_data,
    float* out_result,
    float* out_ref
) {
    constexpr int kNumHeads = 32;
    constexpr int kGroup = kNumHeads / NUM_KV_HEADS;
    constexpr int kThreadsPerHead = MMA_THREADS / kGroup;
    constexpr int kDimsPerThread = HEAD_DIM / kThreadsPerHead;
    constexpr float kScale = 0.08838834764831845f;

    __shared__ __align__(16) __nv_bfloat16 shared_k[PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(16) __nv_bfloat16 shared_v[PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(64) float shared_scores[kGroup][PAGE_SIZE];
    __shared__ float shared_page_alpha[kGroup];
    __shared__ float shared_merged_max[kGroup];
    __shared__ float shared_new_sum[kGroup];
    __shared__ float shared_running_max[kGroup];
    __shared__ float shared_running_sum[kGroup];
    __shared__ float shared_output_acc[kGroup][HEAD_DIM];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    const int a_row = lane & 0xf;
    const int a_k_off = ((lane >> 4) << 2);
    const int b_token = lane & 0xf;
    const int b_k_off = ((lane >> 4) << 2);
    const int c_row_base = ((lane >> 4) << 2);
    const int c_col = lane & 0xf;
    const int head_in_group = tid / kThreadsPerHead;
    const int thread_in_head = tid % kThreadsPerHead;
    const int dim_base = thread_in_head * kDimsPerThread;

    // Init shared state
    for (int i = tid; i < kGroup * HEAD_DIM; i += MMA_THREADS) {
        ((float*)shared_output_acc)[i] = 0.0f;
    }
    if (tid < kGroup) {
        shared_running_max[tid] = -CUDART_INF_F;
        shared_running_sum[tid] = 0.0f;
    }

    // Copy K and V to shared
    for (int i = tid; i < PAGE_SIZE * HEAD_DIM; i += MMA_THREADS) {
        shared_k[i] = k_data[i];
        shared_v[i] = v_data[i];
    }

    // Pre-load Q fragments
    const __nv_bfloat16 zero_bf16 = __float2bfloat16(0.0f);
    mma_bf16_fragment_t q_frag[MMA_K_TILES];
    for (int kt = 0; kt < MMA_K_TILES; ++kt) {
        const int k_base = kt * MMA_TILE + warp_id * 8;
        if (a_row < kGroup) {
            const int64_t q_base = static_cast<int64_t>(a_row) * HEAD_DIM;
            q_frag[kt].element[0] = q[q_base + k_base + 0 + a_k_off];
            q_frag[kt].element[1] = q[q_base + k_base + 1 + a_k_off];
            q_frag[kt].element[2] = q[q_base + k_base + 2 + a_k_off];
            q_frag[kt].element[3] = q[q_base + k_base + 3 + a_k_off];
        } else {
            for (int i = 0; i < 4; ++i) q_frag[kt].element[i] = zero_bf16;
        }
    }
    __syncthreads();

    // ---- QK MMA ----
    mma_float_fragment_t score_fragment;
    for (int i = 0; i < 4; ++i) score_fragment.element[i] = 0.0f;
    for (int kt = 0; kt < MMA_K_TILES; ++kt) {
        mma_bf16_fragment_t k_frag;
        const int k_base = kt * MMA_TILE + warp_id * 8 + b_k_off;
        k_frag.element[0] = shared_k[b_token * HEAD_DIM + k_base + 0];
        k_frag.element[1] = shared_k[b_token * HEAD_DIM + k_base + 1];
        k_frag.element[2] = shared_k[b_token * HEAD_DIM + k_base + 2];
        k_frag.element[3] = shared_k[b_token * HEAD_DIM + k_base + 3];
        score_fragment.vector = __builtin_mxc_mma_16x16x16bf16(
            q_frag[kt].vector, k_frag.vector, score_fragment.vector);
    }
    if (warp_id == 0) {
        for (int i = 0; i < 4; ++i) {
            const int row = c_row_base + i;
            if (row < kGroup) {
                shared_scores[row][c_col] = score_fragment.element[i] * kScale;
            }
        }
    }
    __syncthreads();

    // ---- Softmax ----
    if (tid < kGroup) {
        const int head = tid;
        float page_max = -CUDART_INF_F;
        for (int t = 0; t < PAGE_SIZE; ++t) {
            page_max = fmaxf(page_max, shared_scores[head][t]);
        }
        const float merged_max = fmaxf(shared_running_max[head], page_max);
        const float alpha = shared_running_sum[head] > 0.0f
            ? expf(shared_running_max[head] - merged_max) : 0.0f;
        float page_sum = 0.0f;
        for (int t = 0; t < PAGE_SIZE; ++t) {
            float prob = expf(shared_scores[head][t] - merged_max);
            shared_scores[head][t] = prob;
            page_sum += prob;
        }
        shared_page_alpha[tid] = alpha;
        shared_merged_max[tid] = merged_max;
        shared_new_sum[tid] = shared_running_sum[head] * alpha + page_sum;
    }
    __syncthreads();

    // ---- Rescale ----
    {
        const float alpha = shared_page_alpha[head_in_group];
        if (head_in_group < kGroup) {
            for (int d = 0; d < kDimsPerThread; ++d) {
                shared_output_acc[head_in_group][dim_base + d] *= alpha;
            }
        }
    }
    shared_running_max[head_in_group] = shared_merged_max[head_in_group];
    shared_running_sum[head_in_group] = shared_new_sum[head_in_group];
    __syncthreads();

    // ---- MMA P×V ----
    mma_float_fragment_t pv_acc;
    for (int kt = 0; kt < MMA_K_TILES; ++kt) {
        mma_bf16_fragment_t prob_frag;
        const int token_base = warp_id * 8 + a_k_off;
        if (a_row < kGroup) {
            prob_frag.element[0] = __float2bfloat16(shared_scores[a_row][token_base+0]);
            prob_frag.element[1] = __float2bfloat16(shared_scores[a_row][token_base+1]);
            prob_frag.element[2] = __float2bfloat16(shared_scores[a_row][token_base+2]);
            prob_frag.element[3] = __float2bfloat16(shared_scores[a_row][token_base+3]);
        } else {
            for (int i = 0; i < 4; ++i) prob_frag.element[i] = __float2bfloat16(0.0f);
        }
        mma_bf16_fragment_t v_frag;
        v_frag.element[0] = shared_v[(warp_id*8+b_k_off+0)*HEAD_DIM + kt*MMA_TILE + b_token];
        v_frag.element[1] = shared_v[(warp_id*8+b_k_off+1)*HEAD_DIM + kt*MMA_TILE + b_token];
        v_frag.element[2] = shared_v[(warp_id*8+b_k_off+2)*HEAD_DIM + kt*MMA_TILE + b_token];
        v_frag.element[3] = shared_v[(warp_id*8+b_k_off+3)*HEAD_DIM + kt*MMA_TILE + b_token];
        for (int i = 0; i < 4; ++i) pv_acc.element[i] = 0.0f;
        pv_acc.vector = __builtin_mxc_mma_16x16x16bf16(
            prob_frag.vector, v_frag.vector, pv_acc.vector);
        if (warp_id == 0) {
            for (int i = 0; i < 4; ++i) {
                const int row = c_row_base + i;
                if (row < kGroup) {
                    shared_output_acc[row][kt*MMA_TILE + c_col] += pv_acc.element[i];
                }
            }
        }
    }
    __syncthreads();

    // ---- Write MMA result ----
    if (head_in_group < kGroup) {
        const float inv_denom = shared_running_sum[head_in_group] > 0.0f
            ? 1.0f / shared_running_sum[head_in_group] : 0.0f;
        for (int d = 0; d < kDimsPerThread; ++d) {
            out_result[head_in_group * HEAD_DIM + dim_base + d] =
                shared_output_acc[head_in_group][dim_base + d] * inv_denom;
        }
    }

    // ---- Scalar reference ----
    if (head_in_group < kGroup) {
        // Reset running state for scalar
        float running_max_s = -CUDART_INF_F;
        float running_sum_s = 0.0f;
        float acc_s[kDimsPerThread];
        for (int d = 0; d < kDimsPerThread; ++d) acc_s[d] = 0.0f;

        // QK: same as MMA
        for (int head = 0; head < kGroup; ++head) {
            float scores[PAGE_SIZE];
            // Compute scores (from shared_k)
            int hh = head;
            (void)hh; // just for the thread doing its head only
        }
        // Actually, we can compute reference from already-computed shared_scores  
        // shared_scores already has probabilities, just do scalar P×V
        for (int t = 0; t < PAGE_SIZE; ++t) {
            const float prob = shared_scores[head_in_group][t];
            for (int d = 0; d < kDimsPerThread; ++d) {
                acc_s[d] += prob * __bfloat162float(shared_v[t * HEAD_DIM + dim_base + d]);
            }
        }
        for (int d = 0; d < kDimsPerThread; ++d) {
            out_ref[head_in_group * HEAD_DIM + dim_base + d] =
                acc_s[d] / shared_running_sum[head_in_group];
        }
    }
}

void launch_test(
    torch::Tensor out_mma, torch::Tensor out_ref,
    torch::Tensor q, torch::Tensor k, torch::Tensor v
) {
    single_page_test<4><<<1, 64>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        out_mma.data_ptr<float>(),
        out_ref.data_ptr<float>());
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_test", &launch_test);
}
"""

import tempfile
with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
    f.write(SRC)
    cu_path = f.name

mod = load(name="single_page_test", sources=[cu_path], verbose=True)

import numpy as np
kGroup = 8
rng = np.random.RandomState(12345)
q_host = rng.randn(kGroup, 128).astype(np.float32) * 0.1
k_host = rng.randn(16, 128).astype(np.float32) * 0.1
v_host = rng.randn(16, 128).astype(np.float32) * 0.1

q_t = torch.from_numpy(q_host).bfloat16().cuda()
k_t = torch.from_numpy(k_host).bfloat16().cuda()
v_t = torch.from_numpy(v_host).bfloat16().cuda()

out_mma = torch.zeros(kGroup * 128, dtype=torch.float32).cuda()
out_ref = torch.zeros(kGroup * 128, dtype=torch.float32).cuda()

mod.launch_test(out_mma, out_ref, q_t, k_t, v_t)
torch.cuda.synchronize()

mma_np = out_mma.cpu().numpy().reshape(kGroup, 128)
ref_np = out_ref.cpu().numpy().reshape(kGroup, 128)

has_nan = np.isnan(mma_np).any()
diff = np.abs(mma_np - ref_np).max()
print(f"MMA NaN: {has_nan}, Max diff MMA vs ScalarRef: {diff}")
print(f"MMA[0,:4]: {mma_np[0,:4]}")
print(f"Ref[0,:4]: {ref_np[0,:4]}")

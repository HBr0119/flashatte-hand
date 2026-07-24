#include <cuda_bf16.h>
#include <cuda_runtime.h>
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
#define PAGE_SIZE 16
#define HEAD_DIM 128

__global__ void pv_mma_diag(
    float* out_mma,
    float* out_scalar,
    const float* prob_matrix,
    const __nv_bfloat16* v_data
) {
    constexpr int kGroup = 8;
    constexpr int kThreadsPerHead = 8;
    constexpr int kDimsPerThread = 128 / kThreadsPerHead;

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

    __shared__ float shared_probs[kGroup][PAGE_SIZE];
    __shared__ __align__(16) __nv_bfloat16 shared_v[PAGE_SIZE * HEAD_DIM];

    for (int i = tid; i < kGroup * PAGE_SIZE; i += 64) {
        ((float*)shared_probs)[i] = prob_matrix[i];
    }
    for (int i = tid; i < PAGE_SIZE * HEAD_DIM; i += 64) {
        shared_v[i] = v_data[i];
    }
    __syncthreads();

    // --- MMA P×V ---
    __shared__ float mma_acc[kGroup][HEAD_DIM];
    for (int i = tid; i < kGroup * HEAD_DIM; i += 64) {
        ((float*)mma_acc)[i] = 0.0f;
    }
    __syncthreads();

    mma_float_fragment_t pv_acc;
    for (int kt = 0; kt < 8; ++kt) {
        mma_bf16_fragment_t prob_frag;
        const int token_base = warp_id * 8 + a_k_off;
        if (a_row < kGroup) {
            prob_frag.element[0] = __float2bfloat16(shared_probs[a_row][token_base + 0]);
            prob_frag.element[1] = __float2bfloat16(shared_probs[a_row][token_base + 1]);
            prob_frag.element[2] = __float2bfloat16(shared_probs[a_row][token_base + 2]);
            prob_frag.element[3] = __float2bfloat16(shared_probs[a_row][token_base + 3]);
        } else {
            for (int i = 0; i < 4; ++i)
                prob_frag.element[i] = __float2bfloat16(0.0f);
        }

        mma_bf16_fragment_t v_frag;
        v_frag.element[0] = shared_v[(warp_id * 8 + b_k_off + 0) * HEAD_DIM + kt * MMA_TILE + b_token];
        v_frag.element[1] = shared_v[(warp_id * 8 + b_k_off + 1) * HEAD_DIM + kt * MMA_TILE + b_token];
        v_frag.element[2] = shared_v[(warp_id * 8 + b_k_off + 2) * HEAD_DIM + kt * MMA_TILE + b_token];
        v_frag.element[3] = shared_v[(warp_id * 8 + b_k_off + 3) * HEAD_DIM + kt * MMA_TILE + b_token];

        for (int i = 0; i < 4; ++i) pv_acc.element[i] = 0.0f;

        pv_acc.vector = __builtin_mxc_mma_16x16x16bf16(
            prob_frag.vector, v_frag.vector, pv_acc.vector);

        if (warp_id == 0) {
            for (int i = 0; i < 4; ++i) {
                const int row = c_row_base + i;
                if (row < kGroup) {
                    mma_acc[row][kt * MMA_TILE + c_col] += pv_acc.element[i];
                }
            }
        }
    }
    __syncthreads();

    // --- Scalar P×V ---
    __shared__ float scalar_acc[kGroup][HEAD_DIM];
    for (int i = tid; i < kGroup * HEAD_DIM; i += 64) {
        ((float*)scalar_acc)[i] = 0.0f;
    }
    __syncthreads();

    for (int token = 0; token < PAGE_SIZE; ++token) {
        const float prob = shared_probs[head_in_group][token];
        if (head_in_group < kGroup) {
            for (int d = 0; d < kDimsPerThread; ++d) {
                scalar_acc[head_in_group][dim_base + d] +=
                    prob * __bfloat162float(shared_v[token * HEAD_DIM + dim_base + d]);
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < kGroup * HEAD_DIM; i += 64) {
        out_mma[i] = ((float*)mma_acc)[i];
        out_scalar[i] = ((float*)scalar_acc)[i];
    }
}

void launch_diag(torch::Tensor out_mma, torch::Tensor out_scalar,
                 torch::Tensor prob, torch::Tensor v_data) {
    pv_mma_diag<<<1, 64>>>(
        out_mma.data_ptr<float>(),
        out_scalar.data_ptr<float>(),
        prob.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(v_data.data_ptr()));
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_diag", &launch_diag);
}

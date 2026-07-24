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
#define PAGE_SIZE 16
#define HEAD_DIM 128
#define MMA_K_TILES (HEAD_DIM / MMA_TILE)  // 8

__global__ void integrated_mma_diag(
    float* out_mma_pv,
    float* out_scalar_pv,
    const __nv_bfloat16* q_data,      // [8 heads][HEAD_DIM]
    const __nv_bfloat16* k_data,      // [PAGE_SIZE][HEAD_DIM]
    const __nv_bfloat16* v_data       // [PAGE_SIZE][HEAD_DIM]
) {
    constexpr int kGroup = 8;    // GQA ratio (matching 4 KV-heads)
    constexpr int kThreadsPerHead = 8;
    constexpr int kDimsPerThread = 128 / kThreadsPerHead;
    constexpr float kScale = 0.08838834764831845f;
    constexpr int kThreads = 64;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    // WMMA lane mapping (same as v2/v3)
    const int a_row = lane & 0xf;
    const int a_k_off = ((lane >> 4) << 2);
    const int b_token = lane & 0xf;
    const int b_k_off = ((lane >> 4) << 2);
    const int c_row_base = ((lane >> 4) << 2);
    const int c_col = lane & 0xf;
    const int head_in_group = tid / kThreadsPerHead;
    const int thread_in_head = tid % kThreadsPerHead;
    const int dim_base = thread_in_head * kDimsPerThread;

    // Shared memory: K, V, scores, running state, output acc
    __shared__ __align__(16) __nv_bfloat16 shared_k[PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(16) __nv_bfloat16 shared_v[PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(64) float shared_scores[kGroup][PAGE_SIZE];
    __shared__ float shared_page_alpha[kGroup];
    __shared__ float shared_merged_max[kGroup];
    __shared__ float shared_new_sum[kGroup];
    __shared__ float shared_running_max[kGroup];
    __shared__ float shared_running_sum[kGroup];
    __shared__ float shared_mma_acc[kGroup][HEAD_DIM];  // MMA P×V results
    __shared__ float shared_scalar_acc[kGroup][HEAD_DIM]; // Scalar P×V results

    // Load K and V into shared memory (sync copy, like CP-async result)
    for (int i = tid; i < PAGE_SIZE * HEAD_DIM; i += kThreads) {
        shared_k[i] = k_data[i];
        shared_v[i] = v_data[i];
    }

#define ZERO_SHMEM(arr, count) \
    for (int i = tid; i < (count); i += kThreads) { ((float*)(arr))[i] = 0.0f; }

    ZERO_SHMEM(shared_mma_acc, kGroup * HEAD_DIM);
    ZERO_SHMEM(shared_scalar_acc, kGroup * HEAD_DIM);

    // Init running state
    if (tid < kGroup) {
        shared_running_max[tid] = -CUDART_INF_F;
        shared_running_sum[tid] = 0.0f;
    }
    __syncthreads();

    // ---- Pre-load Q into MMA fragments (same as v2/v3) ----
    const __nv_bfloat16 zero_bf16 = __float2bfloat16(0.0f);
    mma_bf16_fragment_t q_frag[MMA_K_TILES];

    for (int kt = 0; kt < MMA_K_TILES; ++kt) {
        const int k_base = kt * MMA_TILE + warp_id * 8;
        if (a_row < kGroup) {
            const int64_t q_base = static_cast<int64_t>(a_row) * HEAD_DIM;
            q_frag[kt].element[0] = q_data[q_base + k_base + 0 + a_k_off];
            q_frag[kt].element[1] = q_data[q_base + k_base + 1 + a_k_off];
            q_frag[kt].element[2] = q_data[q_base + k_base + 2 + a_k_off];
            q_frag[kt].element[3] = q_data[q_base + k_base + 3 + a_k_off];
        } else {
            for (int i = 0; i < 4; ++i)
                q_frag[kt].element[i] = zero_bf16;
        }
    }

    // ==================== STEP 1: QK MMA (same as v2/v3) ====================
    mma_float_fragment_t score_fragment;
    // Zero accumulator
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

    // Write scores to shared
    if (warp_id == 0) {
        for (int i = 0; i < 4; ++i) {
            const int row = c_row_base + i;
            if (row < kGroup) {
                shared_scores[row][c_col] = score_fragment.element[i] * kScale;
            }
        }
    }
    __syncthreads();

    // ==================== STEP 2: Online softmax ====================
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

    // Rescale existing accumulators
    {
        const float alpha = shared_page_alpha[head_in_group];
        if (head_in_group < kGroup) {
            for (int d = 0; d < kDimsPerThread; ++d) {
                shared_mma_acc[head_in_group][dim_base + d] *= alpha;
                shared_scalar_acc[head_in_group][dim_base + d] *= alpha;
            }
        }
    }

    shared_running_max[head_in_group] = shared_merged_max[head_in_group];
    shared_running_sum[head_in_group] = shared_new_sum[head_in_group];

    __syncthreads();

    // ==================== STEP 3a: MMA P×V ====================
    mma_float_fragment_t pv_acc;
    for (int kt = 0; kt < MMA_K_TILES; ++kt) {
        mma_bf16_fragment_t prob_frag;
        const int token_base = warp_id * 8 + a_k_off;
        if (a_row < kGroup) {
            prob_frag.element[0] = __float2bfloat16(shared_scores[a_row][token_base + 0]);
            prob_frag.element[1] = __float2bfloat16(shared_scores[a_row][token_base + 1]);
            prob_frag.element[2] = __float2bfloat16(shared_scores[a_row][token_base + 2]);
            prob_frag.element[3] = __float2bfloat16(shared_scores[a_row][token_base + 3]);
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
                    shared_mma_acc[row][kt * MMA_TILE + c_col] += pv_acc.element[i];
                }
            }
        }
    }
    __syncthreads();

    // ==================== STEP 3b: Scalar P×V (for comparison) ====================
    for (int token = 0; token < PAGE_SIZE; ++token) {
        const float prob = shared_scores[head_in_group][token];
        if (head_in_group < kGroup) {
            for (int d = 0; d < kDimsPerThread; ++d) {
                shared_scalar_acc[head_in_group][dim_base + d] +=
                    prob * __bfloat162float(shared_v[token * HEAD_DIM + dim_base + d]);
            }
        }
    }
    __syncthreads();

    // ---- Write both NORMALIZED results to global memory ----
    __syncthreads();
    if (head_in_group < kGroup) {
        const float inv_denom = shared_running_sum[head_in_group] > 0.0f
            ? 1.0f / shared_running_sum[head_in_group] : 0.0f;
        for (int d = 0; d < kDimsPerThread; ++d) {
            out_mma_pv[head_in_group * HEAD_DIM + dim_base + d] =
                shared_mma_acc[head_in_group][dim_base + d] * inv_denom;
            out_scalar_pv[head_in_group * HEAD_DIM + dim_base + d] =
                shared_scalar_acc[head_in_group][dim_base + d] * inv_denom;
        }
    }
}

void launch_integrated_diag(torch::Tensor out_mma, torch::Tensor out_scalar,
                             torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    integrated_mma_diag<<<1, 64>>>(
        out_mma.data_ptr<float>(),
        out_scalar.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()));
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_integrated_diag", &launch_integrated_diag);
}

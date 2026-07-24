/*
 * FlashAttention Paged GQA Decode — v8 (McFlashInfer-inspired warp-shuffle QK)
 *
 * Key differences from v5:
 *   - Warp-shuffle QK dot product (no MMA → lower register pressure)
 *   - GQA via blockDim.y: multiple Q heads per block sharing KV in shared memory
 *   - Online softmax (base-2 exp for precision)
 *   - Scalar P×V accumulation
 *
 * Block layout:
 *   gridDim.x = batch_size
 *   gridDim.y = num_kv_heads
 *   blockDim.x = WARP_SIZE = 32
 *   blockDim.y = group_size = num_heads / num_kv_heads
 *   blockDim.z = 1
 *
 * Each warp (blockDim.x threads) computes QK for one Q head.
 * All Q heads in a block share the same K/V in shared memory.
 */

#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <maca_bfloat16.h>
#include <mc_runtime.h>

// ─── Constants ─────────────────────────────────────────────────────────

#define HEAD_DIM 128
#define PAGE_SIZE 16
#define WARP_SIZE 32
#define VEC_SIZE (HEAD_DIM / WARP_SIZE)  // 4: each thread handles 4 head_dim elements
#define NUM_STAGES 2  // double-buffer shared memory stages

// ─── Device helpers ────────────────────────────────────────────────────

struct OnlineSoftmaxState {
    float o[VEC_SIZE];   // output accumulator (4 floats per thread)
    float m;             // running max logit
    float d;             // running denominator (sum-exp)
};

__device__ __forceinline__ void state_init(OnlineSoftmaxState& st) {
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) st.o[i] = 0.f;
    st.m = -CUDART_INF_F;
    st.d = 0.f;
}

__device__ __forceinline__ void state_normalize(OnlineSoftmaxState& st) {
    float inv_d = 1.f / st.d;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) st.o[i] *= inv_d;
}

// ─── Main kernel ───────────────────────────────────────────────────────

template <int GROUP_SIZE>
__global__ void paged_decode_gqa_kernel_v8(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int num_kv_heads)
{
    const int b = blockIdx.x;                      // batch index
    const int kv_head = blockIdx.y;                // KV head index
    const int qo_head = kv_head * GROUP_SIZE + threadIdx.y;  // QO head (0..31)

    const int seqlen = cache_seqlens[b];
    if (seqlen <= 0) return;
    const int valid_pages = (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;

    const int lane = threadIdx.x;                  // lane within warp (0..31)

    // ── Load Q vector (4 floats per thread) ───────────────────────
    float q_vec[VEC_SIZE];
    if (qo_head < 32) {
        const int64_t q_base = (static_cast<int64_t>(b) * 32 + qo_head) * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            q_vec[i] = __bfloat162float(q[q_base + lane * VEC_SIZE + i]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) q_vec[i] = 0.f;
    }

    // ── Shared memory ──────────────────────────────────────────────
    // Layout: [k_smem[STAGES][PAGE_SIZE][HEAD_DIM], v_smem[STAGES][PAGE_SIZE][HEAD_DIM]]
    extern __shared__ uint8_t smem_raw[];
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* v_smem = k_smem + NUM_STAGES * PAGE_SIZE * HEAD_DIM;

    OnlineSoftmaxState st;
    state_init(st);

    // ── Stride helpers ─────────────────────────────────────────────
    // Cache layout: (num_blocks, page_size, num_kv_heads, headdim) — NHD
    const int64_t page_stride = PAGE_SIZE * num_kv_heads * HEAD_DIM;
    const int64_t token_stride = num_kv_heads * HEAD_DIM;
    const int64_t block_row_base = static_cast<int64_t>(b) * blocks_per_batch;

    const float sm_scale = 1.f / sqrtf(static_cast<float>(HEAD_DIM));
    // sm_scale_log2 = sm_scale / ln(2) = sm_scale * log2(e)
    // exp2(score * sm_scale_log2) = exp(score * sm_scale)
    const float sm_scale_log2 = sm_scale * 1.4426950408889634f;

    // ── Preload first STAGES pages (sync) ──────────────────────────
    int stage_idx = 0;
    #pragma unroll
    for (int s = 0; s < NUM_STAGES && s < valid_pages; ++s) {
        const int physical_block = block_table[block_row_base + s];
        const int tok_count = (s == valid_pages - 1 && seqlen % PAGE_SIZE != 0)
                              ? seqlen % PAGE_SIZE : PAGE_SIZE;
        const int64_t page_base = static_cast<int64_t>(physical_block) * page_stride
                                  + kv_head * HEAD_DIM;

        for (int t = threadIdx.y; t < tok_count; t += GROUP_SIZE) {
            const int smem_idx = (stage_idx * PAGE_SIZE + t) * HEAD_DIM + lane * VEC_SIZE;
            const int64_t gmem_idx = page_base + t * token_stride + lane * VEC_SIZE;
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                k_smem[smem_idx + i] = k_cache_paged[gmem_idx + i];
                v_smem[smem_idx + i] = v_cache_paged[gmem_idx + i];
            }
        }
        stage_idx = (stage_idx + 1) % NUM_STAGES;
    }
    __syncthreads();

    // ── Main page loop (compute + pipeline) ────────────────────────
    for (int page = 0; page < valid_pages; ++page) {
        const int tok_count = (page == valid_pages - 1 && seqlen % PAGE_SIZE != 0)
                              ? seqlen % PAGE_SIZE : PAGE_SIZE;
        const int compute_stage = page % NUM_STAGES;

        // Process ALL tokens: ALL threads (all ty) compute QK for each token
        for (int tok = 0; tok < tok_count; ++tok) {
            const int smem_base = (compute_stage * PAGE_SIZE + tok) * HEAD_DIM;

            // ── Load K from shared, compute dot product ──────────
            float dot = 0.f;
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                float k_val = __bfloat162float(k_smem[smem_base + lane * VEC_SIZE + i]);
                dot += q_vec[i] * k_val;
            }

            // Warp shuffle reduction (all 32 lanes → full dot product)
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, offset);
            }

            // ── Online softmax ────────────────────────────────────
            float score = dot * sm_scale_log2;
            float m_prev = st.m;
            st.m = fmaxf(st.m, score);

            float o_scale = exp2f(m_prev - st.m);
            st.d *= o_scale;
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                st.o[i] *= o_scale;
            }

            float p = exp2f(score - st.m);
            st.d += p;

            // ── P×V accumulation ─────────────────────────────────
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                st.o[i] += p * __bfloat162float(v_smem[smem_base + lane * VEC_SIZE + i]);
            }
        }
        __syncthreads();

        // ── Load next page for pipeline ───────────────────────────
        int next_page = page + NUM_STAGES;
        if (next_page < valid_pages) {
            int next_stage = next_page % NUM_STAGES;
            const int physical_block = block_table[block_row_base + next_page];
            const int64_t page_base = static_cast<int64_t>(physical_block) * page_stride
                                      + kv_head * HEAD_DIM;
            const int next_tok = (next_page == valid_pages - 1 && seqlen % PAGE_SIZE != 0)
                                 ? seqlen % PAGE_SIZE : PAGE_SIZE;

            for (int t = threadIdx.y; t < next_tok; t += GROUP_SIZE) {
                const int smem_idx = (next_stage * PAGE_SIZE + t) * HEAD_DIM + lane * VEC_SIZE;
                const int64_t gmem_idx = page_base + t * token_stride + lane * VEC_SIZE;
                #pragma unroll
                for (int i = 0; i < VEC_SIZE; ++i) {
                    k_smem[smem_idx + i] = k_cache_paged[gmem_idx + i];
                    v_smem[smem_idx + i] = v_cache_paged[gmem_idx + i];
                }
            }
        }
        __syncthreads();
    }

    // ── Normalize and write output ─────────────────────────────────
    state_normalize(st);
    if (qo_head < 32) {
        const int64_t out_base = (static_cast<int64_t>(b) * 32 + qo_head) * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            output[out_base + lane * VEC_SIZE + i] = __float2bfloat16(st.o[i]);
        }
    }
}

// ─── Host dispatch ─────────────────────────────────────────────────────

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
    int64_t causal)
{
    (void)seqlen_k; (void)seqlen_q; (void)headdim; (void)page_block_size; (void)causal;

    const int blocks_per_batch = static_cast<int>(num_blocks / batch_size);
    const int group_size = static_cast<int>(num_heads / num_heads_k);
    const int nkh = static_cast<int>(num_heads_k);

    dim3 grid(static_cast<unsigned int>(batch_size),
              static_cast<unsigned int>(num_heads_k));
    dim3 block(WARP_SIZE, static_cast<unsigned int>(group_size), 1);
    // 2 stages × (K + V) × PAGE_SIZE × HEAD_DIM × sizeof(bf16)
    size_t smem_bytes = 2 * NUM_STAGES * PAGE_SIZE * HEAD_DIM * sizeof(__nv_bfloat16);

    if (group_size == 8) {
        paged_decode_gqa_kernel_v8<8><<<grid, block, smem_bytes>>>(
            q, k_cache_paged, v_cache_paged, output,
            cache_seqlens, block_table, blocks_per_batch, nkh);
    } else {
        paged_decode_gqa_kernel_v8<4><<<grid, block, smem_bytes>>>(
            q, k_cache_paged, v_cache_paged, output,
            cache_seqlens, block_table, blocks_per_batch, nkh);
    }
}

/*
 * v11 best/v10 hybrid: McFlashInfer BSM pipeline with best-style dispatch,
 * balanced Split-KV, leader LSE merge, shared-weight combine and tiny-GQA.
 * FlashAttention Paged GQA Decode — v10 (Full McFlashInfer restoration)
 *
 * Strict reimplementation of McFlashInfer BatchDecodeWithPagedKVCacheKernel:
 *   - 128-bit BSM async loads (load_128b_bsm / load_128b_bsm_pred)
 *   - bdx=16, vec_size=8 (8 bf16 per thread, 128-bit aligned)
 *   - Block: (16, GROUP_SIZE, BDZ) → BDZ=2 (G=8) or BDZ=4 (G=4)
 *   - Dual FMA ILP (__builtin_mxc_pk_fma_f32) for QK + P×V
 *   - LSE log2 softmax state + state_merge
 *   - 2-stage BSM double-buffer pipeline with cp_async_bsm_wait
 *   - Multi-block partition-KV + combine_kernel for long sequences
 */

#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <maca_bfloat16.h>
#include <mc_runtime.h>

// ─── Constants ────────────────────────────────────────────────────────────

#define HEAD_DIM 128
#define PAGE_SIZE 16
#define VEC_SIZE 8        // 8 bf16 = 128 bits per thread
#define BDX 16            // 16 threads in x-dim (McFlashInfer default)
#define NUM_STAGES 2      // double-buffer smem stages
#define MAX_SPLITS 64

// ─── BSM 128-bit helpers ─────────────────────────────────────────────────

__device__ __forceinline__ void bsm_load_128b(
    __nv_bfloat16* smem, const __nv_bfloat16* gmem)
{
    typedef __NATIVE_VECTOR__(4, int) VecType;
    auto* dst = (VecType*)smem;
    auto* src = (VecType*)gmem;
    __builtin_mxc_ldg_b128_bsm(dst, src, 0, -1, true, true, false, true);
}

__device__ __forceinline__ void bsm_load_128b_pred(
    __nv_bfloat16* smem, const __nv_bfloat16* gmem, bool pred)
{
    typedef __NATIVE_VECTOR__(4, int) VecType;
    auto* dst = (VecType*)smem;
    auto* src = (VecType*)gmem;
    __builtin_mxc_ldg_b128_bsm_predicator(
        dst, src, 0, true, true, false, true, pred, 1, MACA_ICMP_EQ);
}

template <int N>
__device__ __forceinline__ void bsm_wait() {
    __builtin_mxc_arrive_gvmcnt(N);
    __builtin_mxc_barrier_ex(4);
}

// ─── Dual FMA ILP ────────────────────────────────────────────────────────

// output[0] = a[0]*b[0] + c[0], output[1] = a[1]*b[1] + c[1]
__device__ __forceinline__ void fma_f32x2(float* out, const float* a,
                                           const float* b, const float* c) {
    typedef __NATIVE_VECTOR__(2, float) Float2;
    Float2 va = {a[0], a[1]};
    Float2 vb = {b[0], b[1]};
    Float2 vc = {c[0], c[1]};
    Float2 vo = __builtin_mxc_pk_fma_f32(va, vb, vc);
    *(Float2*)out = vo;
}

// output[0] = a[0]*scale + c, output[1] = a[1]*scale + c
__device__ __forceinline__ void fma_f32x2(float* out, const float* a,
                                           float scale, float c = 0.f) {
    typedef __NATIVE_VECTOR__(2, float) Float2;
    Float2 va = {a[0], a[1]};
    Float2 vb = {scale, scale};
    Float2 vc = {c, c};
    Float2 vo = __builtin_mxc_pk_fma_f32(va, vb, vc);
    *(Float2*)out = vo;
}

// ─── LSE log2 Softmax State ──────────────────────────────────────────────

struct SoftmaxState {
    float o[VEC_SIZE];   // 8 floats per thread
    float m_log2;        // running max logit (log2 domain)
    float d;             // running denominator (sum-exp)
};

__device__ __forceinline__ void state_init(SoftmaxState& st) {
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) st.o[i] = 0.f;
    st.m_log2 = -CUDART_INF_F;
    st.d = 0.f;
}

// Merge src (stored as o array + m + d) into dst (LSE log2 variant)
__device__ __forceinline__ void state_merge(SoftmaxState& dst,
                                             const float* src_o,
                                             float src_m_log2, float src_d) {
    float m_new = fmaxf(dst.m_log2, src_m_log2);
    float dst_scale = exp2f(dst.m_log2 - m_new);
    float src_scale = exp2f(src_m_log2 - m_new);
    dst.d = dst.d * dst_scale + src_d * src_scale;
    // Dual FMA rescaling for dst accumulator
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i += 2) {
        float scaled[2];
        fma_f32x2(scaled, &dst.o[i], dst_scale);
        dst.o[i]   = scaled[0];
        dst.o[i+1] = scaled[1];
    }
    // Accumulate src
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        dst.o[i] += src_o[i] * src_scale;
    }
    dst.m_log2 = m_new;
}

__device__ __forceinline__ float state_get_lse(const SoftmaxState& st) {
    return st.d > 0.f ? st.m_log2 + log2f(st.d) : -CUDART_INF_F;
}

__device__ __forceinline__ void state_normalize(SoftmaxState& st) {
    if (st.d <= 0.f) return;
    float inv_d = 1.f / st.d;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) st.o[i] *= inv_d;
}

__device__ __forceinline__ float subgroup_sum_16_v11(float value) {
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset, 16);
    }
    return value;
}

// Low-overhead direct path for KV <= 32. A block is one batch/KV-head;
// all query heads in the group reuse the same staged K/V page.
template <int GROUP_SIZE>
__global__ __launch_bounds__(BDX * GROUP_SIZE)
void paged_decode_tiny_gqa_kernel_v11(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int num_kv_heads) {
    __shared__ __align__(16) __nv_bfloat16 k_smem[PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(16) __nv_bfloat16 v_smem[PAGE_SIZE * HEAD_DIM];

    constexpr int THREADS = BDX * GROUP_SIZE;
    constexpr int PACKS_PER_TOKEN = HEAD_DIM / VEC_SIZE;
    constexpr float SCALE_LOG2 =
        0.08838834764831845f * 1.4426950408889634f;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int linear_tid = tx + BDX * ty;
    const int bkv = blockIdx.x;
    const int b = bkv / num_kv_heads;
    const int kv_head = bkv - b * num_kv_heads;
    const int q_head = kv_head * GROUP_SIZE + ty;
    const int64_t bh = static_cast<int64_t>(b) * 32 + q_head;
    const int64_t out_base = bh * HEAD_DIM + tx * VEC_SIZE;
    const int seqlen = cache_seqlens[b];

    if (seqlen <= 0) {
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) output[out_base + i] = __float2bfloat16(0.f);
        return;
    }

    const int32_t* block_row =
        block_table + static_cast<int64_t>(b) * blocks_per_batch;
    const int64_t token_stride = static_cast<int64_t>(num_kv_heads) * HEAD_DIM;
    const int64_t page_stride = static_cast<int64_t>(PAGE_SIZE) * token_stride;
    const int64_t head_offset = static_cast<int64_t>(kv_head) * HEAD_DIM;

    if (seqlen == 1) {
        const int physical_page = __ldg(block_row);
        const int64_t v_base = static_cast<int64_t>(physical_page) * page_stride +
                               head_offset + tx * VEC_SIZE;
        *reinterpret_cast<int4*>(output + out_base) =
            *reinterpret_cast<const int4*>(v_cache_paged + v_base);
        return;
    }

    float q_vec[VEC_SIZE];
    float o[VEC_SIZE];
    const int64_t q_base = bh * HEAD_DIM + tx * VEC_SIZE;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        q_vec[i] = __bfloat162float(q[q_base + i]);
        o[i] = 0.f;
    }

    float running_max = -CUDART_INF_F;
    float running_sum = 0.f;
    const int valid_pages = (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;

    for (int logical_page = 0; logical_page < valid_pages; ++logical_page) {
        const int token_begin = logical_page * PAGE_SIZE;
        int page_tokens = seqlen - token_begin;
        if (page_tokens > PAGE_SIZE) page_tokens = PAGE_SIZE;
        const int physical_page = __ldg(block_row + logical_page);
        const int64_t page_base = static_cast<int64_t>(physical_page) * page_stride + head_offset;
        const int pack_count = page_tokens * PACKS_PER_TOKEN;

        for (int pack = linear_tid; pack < pack_count; pack += THREADS) {
            const int token = pack / PACKS_PER_TOKEN;
            const int dim = (pack - token * PACKS_PER_TOKEN) * VEC_SIZE;
            const int64_t src = page_base + static_cast<int64_t>(token) * token_stride + dim;
            const int dst = token * HEAD_DIM + dim;
            *reinterpret_cast<int4*>(k_smem + dst) =
                *reinterpret_cast<const int4*>(k_cache_paged + src);
            *reinterpret_cast<int4*>(v_smem + dst) =
                *reinterpret_cast<const int4*>(v_cache_paged + src);
        }
        __syncthreads();

        #pragma unroll 1
        for (int token = 0; token < page_tokens; ++token) {
            const int smem_base = token * HEAD_DIM + tx * VEC_SIZE;
            float dot = 0.f;
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                dot = fmaf(q_vec[i], __bfloat162float(k_smem[smem_base + i]), dot);
            }
            const float score = subgroup_sum_16_v11(dot) * SCALE_LOG2;
            const float new_max = fmaxf(running_max, score);
            const float alpha = running_sum > 0.f ? exp2f(running_max - new_max) : 0.f;
            const float p = exp2f(score - new_max);
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                o[i] = fmaf(p, __bfloat162float(v_smem[smem_base + i]), alpha * o[i]);
            }
            running_sum = fmaf(alpha, running_sum, p);
            running_max = new_max;
        }
        __syncthreads();
    }

    const float inv = running_sum > 0.f ? 1.f / running_sum : 0.f;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        output[out_base + i] = __float2bfloat16(o[i] * inv);
    }
}

// ─── Partition Kernel ────────────────────────────────────────────────────

template <int GROUP_SIZE, int BDZ>
__global__ void paged_decode_partition_kernel_v11(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    float* __restrict__ partial_o,
    float* __restrict__ partial_lse,
    __nv_bfloat16* __restrict__ output_direct,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int num_kv_heads,
    int num_splits)
{
    static_assert(BDX * GROUP_SIZE * BDZ == 256, "Expected 256 threads");
    constexpr int STAGE_KV_ELEMS = BDZ * GROUP_SIZE * HEAD_DIM;  // bf16 per stage
    // Tokens per stage = BDZ * GROUP = exactly PAGE_SIZE (16)

    const int bkv = blockIdx.x;           // batch × kv_head
    const int split_idx = blockIdx.y;     // which KV split
    const int b = bkv / num_kv_heads;
    const int kv_head = bkv - b * num_kv_heads;
    const int q_head = kv_head * GROUP_SIZE + threadIdx.y;

    const int tx = threadIdx.x;  // 0..15
    const int ty = threadIdx.y;  // 0..GROUP_SIZE-1
    const int tz = threadIdx.z;  // 0..BDZ-1

    // ── Q load: each thread loads VEC_SIZE=8 bf16 (128-bit wide) ─────
    float q_vec[VEC_SIZE];
    if (q_head < 32) {
        const int64_t q_base = (static_cast<int64_t>(b) * 32 + q_head) * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            q_vec[i] = __bfloat162float(q[q_base + tx * VEC_SIZE + i]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) q_vec[i] = 0.f;
    }

    // ── Shared memory ─────────────────────────────────────────────────
    // Layout: [k_stage0][k_stage1][v_stage0][v_stage1]
    // Each stage: BDZ × GROUP × HEAD_DIM bf16
    extern __shared__ uint8_t smem_raw[];
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* v_smem = k_smem + NUM_STAGES * STAGE_KV_ELEMS;

    SoftmaxState st;
    state_init(st);

    // ── Page range for this split ─────────────────────────────────────
    const int seqlen = cache_seqlens[b];
    const int valid_pages = seqlen > 0 ? (seqlen + PAGE_SIZE - 1) / PAGE_SIZE : 0;
    const int first_page = static_cast<int>(
        static_cast<int64_t>(valid_pages) * split_idx / num_splits);
    const int last_page = static_cast<int>(
        static_cast<int64_t>(valid_pages) * (split_idx + 1) / num_splits);
    if (first_page >= last_page) {
        if (tz == 0 && q_head < 32) {
            if (num_splits <= 1) {
                const int64_t out_base =
                    (static_cast<int64_t>(b) * 32 + q_head) * HEAD_DIM + tx * VEC_SIZE;
                #pragma unroll
                for (int i = 0; i < VEC_SIZE; ++i) {
                    output_direct[out_base + i] = __float2bfloat16(0.f);
                }
            } else {
                const int64_t p_idx =
                    (static_cast<int64_t>(b) * 32 + q_head) * num_splits + split_idx;
                const int64_t p_base = p_idx * HEAD_DIM + tx * VEC_SIZE;
                #pragma unroll
                for (int i = 0; i < VEC_SIZE; ++i) partial_o[p_base + i] = 0.f;
                if (tx == 0) partial_lse[p_idx] = -CUDART_INF_F;
            }
        }
        return;
    }

    // ── Stride helpers ─────────────────────────────────────────────────
    const int64_t page_stride   = static_cast<int64_t>(PAGE_SIZE) * num_kv_heads * HEAD_DIM;
    const int64_t token_stride  = static_cast<int64_t>(num_kv_heads) * HEAD_DIM;
    const int64_t block_row_base = static_cast<int64_t>(b) * blocks_per_batch;
    const int32_t* block_row = block_table + block_row_base;

    const float sm_scale = 1.f / sqrtf(static_cast<float>(HEAD_DIM));
    const float sm_scale_log2 = sm_scale * 1.4426950408889634f;

    // Thread's token index within a BDZ×GROUP stage (0..15)
    const int token_in_stage = tz * GROUP_SIZE + ty;

    // ═══════════════════════════════════════════════════════════════════
    // Preload: issue BSM loads for stages 0 and 1, wait, then start loop
    // ═══════════════════════════════════════════════════════════════════

    // Helper lambda: load one page into stage s
    auto load_page = [&](int stage_s, int page_idx) {
        int phys = block_row[page_idx];
        int remaining = seqlen - page_idx * PAGE_SIZE;
        int ntok = (remaining < PAGE_SIZE) ? remaining : PAGE_SIZE;
        bool valid = (token_in_stage < ntok);
        int64_t page_base = static_cast<int64_t>(phys) * page_stride
                           + static_cast<int64_t>(kv_head) * HEAD_DIM;
        int64_t gmem_base = page_base
                           + static_cast<int64_t>(token_in_stage) * token_stride
                           + tx * VEC_SIZE;
        int smem_off = (stage_s * BDZ + tz) * GROUP_SIZE * HEAD_DIM
                      + ty * HEAD_DIM + tx * VEC_SIZE;
        bsm_load_128b_pred(k_smem + smem_off, k_cache_paged + gmem_base, valid);
        bsm_load_128b_pred(v_smem + smem_off, v_cache_paged + gmem_base, valid);
    };

    // Preload stages 0 and 1
    load_page(0, first_page);
    if (first_page + 1 < last_page) {
        load_page(1, first_page + 1);
    }
    bsm_wait<0>();
    __syncthreads();

    // ═══════════════════════════════════════════════════════════════════
    // Main pipeline loop: wait → compute → __syncthreads → issue next
    // ═══════════════════════════════════════════════════════════════════
    for (int page = first_page; page < last_page; ++page) {
        // Preload is relative to first_page: the first page is always stage 0.
        // Absolute page parity selects uninitialized data for odd split starts.
        int cs = (page - first_page) % NUM_STAGES;

        // Wait for compute stage data (BSM loads issued 2 iters ago)
        bsm_wait<2>();

        int remaining = seqlen - page * PAGE_SIZE;
        int page_tokens = (remaining < PAGE_SIZE) ? remaining : PAGE_SIZE;

        // ── QK + softmax for all tokens in this page ─────────────────
        float m_prev = st.m_log2;
        float scores[GROUP_SIZE];

        #pragma unroll
        for (int j = 0; j < GROUP_SIZE; ++j) {
            int tk = tz * GROUP_SIZE + j;
            float score;
            if (tk >= page_tokens) {
                score = -CUDART_INF_F;
            } else {
                int smem_k = (cs * BDZ + tz) * GROUP_SIZE * HEAD_DIM
                            + j * HEAD_DIM + tx * VEC_SIZE;

                // Dual FMA dot product (VEC_SIZE=8 → 4 dual-FMA iterations)
                float acc[2] = {0.f, 0.f};
                #pragma unroll
                for (int i = 0; i < VEC_SIZE; i += 2) {
                    float k_vals[2];
                    k_vals[0] = __bfloat162float(k_smem[smem_k + i]);
                    k_vals[1] = __bfloat162float(k_smem[smem_k + i + 1]);
                    fma_f32x2(acc, &q_vec[i], k_vals, acc);
                }
                float dot = acc[0] + acc[1];

                // Warp-shuffle: bdx=16 → 4 steps (8, 4, 2, 1)
                #pragma unroll
                for (int offset = BDX / 2; offset > 0; offset >>= 1) {
                    dot += __shfl_xor_sync(0xFFFFFFFF, dot, offset);
                }
                score = dot * sm_scale_log2;
            }
            scores[j] = score;
            st.m_log2 = fmaxf(st.m_log2, score);
        }

        // Rescale accumulator (skip on first page to avoid NaN from -inf - -inf)
        if (m_prev > -CUDART_INF_F) {
            float o_scale = exp2f(m_prev - st.m_log2);
            st.d *= o_scale;
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; i += 2) {
                float scaled[2];
                fma_f32x2(scaled, &st.o[i], o_scale);
                st.o[i]   = scaled[0];
                st.o[i+1] = scaled[1];
            }
        }

        // P×V accumulation
        #pragma unroll
        for (int j = 0; j < GROUP_SIZE; ++j) {
            int tk = tz * GROUP_SIZE + j;
            if (tk >= page_tokens) continue;

            float score = scores[j];
            float p = exp2f(score - st.m_log2);
            st.d += p;

            int smem_v = (cs * BDZ + tz) * GROUP_SIZE * HEAD_DIM
                        + j * HEAD_DIM + tx * VEC_SIZE;
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; i += 2) {
                float vv[2];
                vv[0] = __bfloat162float(v_smem[smem_v + i]);
                vv[1] = __bfloat162float(v_smem[smem_v + i + 1]);
                st.o[i]   += p * vv[0];
                st.o[i+1] += p * vv[1];
            }
        }

        __syncthreads();

        // Issue BSM loads for page+NUM_STAGES (into the stage we just consumed)
        int fut_page = page + NUM_STAGES;
        if (fut_page < last_page) {
            load_page(cs, fut_page);  // overwrite cs now that we're done consuming it
        }
    }

    bsm_wait<0>();
    __syncthreads();

    // ═══════════════════════════════════════════════════════════════════
    // Intra-block merge (if BDZ > 1) — reuse K smem as float storage
    // ═══════════════════════════════════════════════════════════════════
    {
        // Reuse the now-dead K/V staging area. The weight calculation is
        // performed once per query head instead of once per x-lane.
        float* merge_buf = reinterpret_cast<float*>(k_smem);
        constexpr int NUM_STATES = BDZ * GROUP_SIZE;
        constexpr int O_ELEMS = NUM_STATES * HEAD_DIM;
        constexpr int LSE_OFFSET = O_ELEMS;
        constexpr int WEIGHT_OFFSET = LSE_OFFSET + NUM_STATES;
        constexpr int INV_OFFSET = WEIGHT_OFFSET + NUM_STATES;
        constexpr int MERGED_LSE_OFFSET = INV_OFFSET + GROUP_SIZE;
        const int state_idx = tz * GROUP_SIZE + ty;

        const float local_inv = st.d > 0.f ? 1.f / st.d : 0.f;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            merge_buf[state_idx * HEAD_DIM + tx * VEC_SIZE + i] = st.o[i] * local_inv;
        }
        if (tx == 0) merge_buf[LSE_OFFSET + state_idx] = state_get_lse(st);
        __syncthreads();

        if (tz == 0 && tx == 0) {
            float max_lse = -CUDART_INF_F;
            #pragma unroll
            for (int z = 0; z < BDZ; ++z) {
                max_lse = fmaxf(max_lse,
                    merge_buf[LSE_OFFSET + z * GROUP_SIZE + ty]);
            }
            float denominator = 0.f;
            #pragma unroll
            for (int z = 0; z < BDZ; ++z) {
                const int idx = z * GROUP_SIZE + ty;
                const float lse = merge_buf[LSE_OFFSET + idx];
                const float w = lse == -CUDART_INF_F ? 0.f : exp2f(lse - max_lse);
                merge_buf[WEIGHT_OFFSET + idx] = w;
                denominator += w;
            }
            merge_buf[INV_OFFSET + ty] = denominator > 0.f ? 1.f / denominator : 0.f;
            merge_buf[MERGED_LSE_OFFSET + ty] = denominator > 0.f
                ? max_lse + log2f(denominator) : -CUDART_INF_F;
        }
        __syncthreads();

        if (tz == 0) {
            float merged[VEC_SIZE];
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) merged[i] = 0.f;
            #pragma unroll
            for (int z = 0; z < BDZ; ++z) {
                const int idx = z * GROUP_SIZE + ty;
                const float w = merge_buf[WEIGHT_OFFSET + idx];
                #pragma unroll
                for (int i = 0; i < VEC_SIZE; ++i) {
                    merged[i] = fmaf(w,
                        merge_buf[idx * HEAD_DIM + tx * VEC_SIZE + i], merged[i]);
                }
            }
            const float inv = merge_buf[INV_OFFSET + ty];
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) st.o[i] = merged[i] * inv;
            st.m_log2 = merge_buf[MERGED_LSE_OFFSET + ty];
            st.d = st.m_log2 == -CUDART_INF_F ? 0.f : 1.f;
        }
        __syncthreads();
    }

    // ═══════════════════════════════════════════════════════════════════
    // Normalize and write output
    // ═══════════════════════════════════════════════════════════════════
    if (tz == 0 && q_head < 32) {
        state_normalize(st);

        if (num_splits <= 1) {
            // Direct output
            int64_t out_base = (static_cast<int64_t>(b) * 32 + q_head) * HEAD_DIM;
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                output_direct[out_base + tx * VEC_SIZE + i] = __float2bfloat16(st.o[i]);
            }
        } else {
            // Partial output
            float lse = state_get_lse(st);
            int64_t p_idx = (static_cast<int64_t>(b) * 32 + q_head) * num_splits + split_idx;
            int64_t p_base = p_idx * HEAD_DIM;
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                partial_o[p_base + tx * VEC_SIZE + i] = st.o[i];
            }
            if (tx == 0) {
                partial_lse[p_idx] = lse;
            }
        }
    }
}

// ─── Combine Kernel (LSE log2 merge) ─────────────────────────────────────

__global__ void combine_lse_kernel_v11(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_lse,
    __nv_bfloat16* __restrict__ output,
    int num_splits)
{
    __shared__ float weights[MAX_SPLITS];
    __shared__ float inverse_denominator;
    const int tid = threadIdx.x;
    const int bh = blockIdx.x;

    if (num_splits <= 1) return;

    const int64_t first = static_cast<int64_t>(bh) * num_splits;
    if (tid == 0) {
        float max_lse = -CUDART_INF_F;
        for (int s = 0; s < num_splits; ++s) {
            max_lse = fmaxf(max_lse, partial_lse[first + s]);
        }
        float denominator = 0.f;
        for (int s = 0; s < num_splits; ++s) {
            const float lse = partial_lse[first + s];
            const float w = lse == -CUDART_INF_F ? 0.f : exp2f(lse - max_lse);
            weights[s] = w;
            denominator += w;
        }
        inverse_denominator = denominator > 0.f ? 1.f / denominator : 0.f;
    }
    __syncthreads();

    float acc = 0.f;
    for (int s = 0; s < num_splits; ++s) {
        const float w = weights[s];
        // Avoid reading an empty split: 0 * an uninitialized NaN is NaN.
        if (w != 0.f) {
            acc = fmaf(w, partial_o[(first + s) * HEAD_DIM + tid], acc);
        }
    }
    output[static_cast<int64_t>(bh) * HEAD_DIM + tid] =
        __float2bfloat16(acc * inverse_denominator);
}

// ─── Split heuristic ─────────────────────────────────────────────────────

static int choose_num_splits(int64_t batch_size, int seqlen) {
    if (seqlen <= 2048) return 1;
    if (batch_size <= 1) {
        if (seqlen >= 8192) return 64;
        if (seqlen >= 4096) return 32;
        return 16;
    }
    if (batch_size <= 8) {
        if (seqlen >= 32768) return 32;
        if (seqlen >= 8192) return 16;
        if (seqlen >= 4096) return 8;
        return 4;
    }
    if (batch_size <= 16) return seqlen >= 8192 ? 8 : 4;
    if (batch_size <= 32) return 4;
    return 2;
}

// ─── Global partition buffers ────────────────────────────────────────────

static float* g_partial_o = nullptr;
static float* g_partial_lse = nullptr;
static size_t g_partial_o_cap = 0;
static size_t g_partial_lse_cap = 0;

static bool ensure_partition_buffers(size_t o_elems, size_t lse_elems) {
    if (o_elems > g_partial_o_cap) {
        float* new_partial_o = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&new_partial_o),
                       o_elems * sizeof(float)) != cudaSuccess ||
            new_partial_o == nullptr) return false;
        if (g_partial_o) cudaFree(g_partial_o);
        g_partial_o = new_partial_o;
        g_partial_o_cap = o_elems;
    }
    if (lse_elems > g_partial_lse_cap) {
        float* new_partial_lse = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&new_partial_lse),
                       lse_elems * sizeof(float)) != cudaSuccess ||
            new_partial_lse == nullptr) return false;
        if (g_partial_lse) cudaFree(g_partial_lse);
        g_partial_lse = new_partial_lse;
        g_partial_lse_cap = lse_elems;
    }
    return g_partial_o != nullptr && g_partial_lse != nullptr;
}

// ─── Host dispatch ───────────────────────────────────────────────────────

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
    if (batch_size <= 0 || num_heads != 32 ||
        (num_heads_k != 4 && num_heads_k != 8) ||
        seqlen_q != 1 || headdim != HEAD_DIM ||
        page_block_size != PAGE_SIZE || causal != 0 || num_blocks <= 0) {
        return;
    }

    const int bs = static_cast<int>(batch_size);
    const int nh = static_cast<int>(num_heads);
    const int nkh = static_cast<int>(num_heads_k);
    const int group_size = nh / nkh;
    const int blocks_per_batch = static_cast<int>(num_blocks / batch_size);

    // The v10 pipeline wins once there is enough work to amortize its 256
    // threads and double buffer. Tiny KV uses the best-style light path.
    if (seqlen_k <= 32) {
        const dim3 tiny_grid(bs * nkh, 1, 1);
        if (group_size == 8) {
            const dim3 tiny_block(BDX, 8, 1);
            paged_decode_tiny_gqa_kernel_v11<8><<<tiny_grid, tiny_block>>>(
                q, k_cache_paged, v_cache_paged, output,
                cache_seqlens, block_table, blocks_per_batch, nkh);
        } else {
            const dim3 tiny_block(BDX, 4, 1);
            paged_decode_tiny_gqa_kernel_v11<4><<<tiny_grid, tiny_block>>>(
                q, k_cache_paged, v_cache_paged, output,
                cache_seqlens, block_table, blocks_per_batch, nkh);
        }
        return;
    }

    // BDZ: BDX(16) × GROUP × BDZ = 256  → GROUP=8→BDZ=2, GROUP=4→BDZ=4
    const int bdz = (group_size == 8) ? 2 : 4;

    dim3 block(BDX, static_cast<unsigned int>(group_size),
               static_cast<unsigned int>(bdz));

    // smem: 2 stages × (K+V) × BDZ × GROUP × HEAD_DIM × sizeof(bf16)
    // = 2 * 2 * bdz * group_size * 128 * 2 bytes = 4 * bdz * group_size * 256
    // Plus merge reuse of K smem (already included above)
    size_t smem_bytes = NUM_STAGES * 2 * bdz * group_size * HEAD_DIM * sizeof(__nv_bfloat16);

    int num_splits = choose_num_splits(batch_size, static_cast<int>(seqlen_k));
    const int max_pages = static_cast<int>((seqlen_k + PAGE_SIZE - 1) / PAGE_SIZE);
    if (num_splits > MAX_SPLITS) num_splits = MAX_SPLITS;
    if (num_splits > max_pages) num_splits = max_pages;
    if (num_splits > blocks_per_batch) num_splits = blocks_per_batch;
    if (num_splits < 1) num_splits = 1;

    if (num_splits <= 1) {
        // Direct mode
        dim3 grid(bs * nkh, 1);
        if (group_size == 8) {
            paged_decode_partition_kernel_v11<8, 2><<<grid, block, smem_bytes>>>(
                q, k_cache_paged, v_cache_paged,
                nullptr, nullptr, output,
                cache_seqlens, block_table,
                blocks_per_batch, nkh, 1);
        } else {
            paged_decode_partition_kernel_v11<4, 4><<<grid, block, smem_bytes>>>(
                q, k_cache_paged, v_cache_paged,
                nullptr, nullptr, output,
                cache_seqlens, block_table,
                blocks_per_batch, nkh, 1);
        }
        return;
    }

    // Partition mode
    const int total_q_heads = bs * nh;
    size_t o_elems   = static_cast<size_t>(total_q_heads) * num_splits * HEAD_DIM;
    size_t lse_elems = static_cast<size_t>(total_q_heads) * num_splits;

    if (!ensure_partition_buffers(o_elems, lse_elems)) {
        // Fallback: direct
        num_splits = 1;
        dim3 grid(bs * nkh, 1);
        if (group_size == 8) {
            paged_decode_partition_kernel_v11<8, 2><<<grid, block, smem_bytes>>>(
                q, k_cache_paged, v_cache_paged,
                nullptr, nullptr, output,
                cache_seqlens, block_table,
                blocks_per_batch, nkh, 1);
        } else {
            paged_decode_partition_kernel_v11<4, 4><<<grid, block, smem_bytes>>>(
                q, k_cache_paged, v_cache_paged,
                nullptr, nullptr, output,
                cache_seqlens, block_table,
                blocks_per_batch, nkh, 1);
        }
        return;
    }

    // Launch partition kernel
    dim3 part_grid(bs * nkh, num_splits);
    if (group_size == 8) {
        paged_decode_partition_kernel_v11<8, 2><<<part_grid, block, smem_bytes>>>(
            q, k_cache_paged, v_cache_paged,
            g_partial_o, g_partial_lse, output,
            cache_seqlens, block_table,
            blocks_per_batch, nkh, num_splits);
    } else {
        paged_decode_partition_kernel_v11<4, 4><<<part_grid, block, smem_bytes>>>(
            q, k_cache_paged, v_cache_paged,
            g_partial_o, g_partial_lse, output,
            cache_seqlens, block_table,
            blocks_per_batch, nkh, num_splits);
    }

    // Launch combine kernel
    combine_lse_kernel_v11<<<total_q_heads, HEAD_DIM>>>(
        g_partial_o, g_partial_lse, output, num_splits);
}

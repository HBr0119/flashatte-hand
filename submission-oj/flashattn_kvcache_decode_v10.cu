/*
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

// ─── Partition Kernel ────────────────────────────────────────────────────

template <int GROUP_SIZE, int BDZ>
__global__ void paged_decode_partition_kernel_v10(
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
    if (seqlen <= 0) return;
    const int valid_pages = (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;
    const int pages_per_split = (valid_pages + num_splits - 1) / num_splits;
    const int first_page = split_idx * pages_per_split;
    int last_page = first_page + pages_per_split;
    if (last_page > valid_pages) last_page = valid_pages;
    if (first_page >= last_page) {
        if (tz == 0 && tx == 0 && q_head < 32) {
            partial_lse[(static_cast<int64_t>(b) * 32 + q_head) * num_splits + split_idx] = -CUDART_INF_F;
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
        int cs = page % NUM_STAGES;  // compute stage

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
    if (BDZ > 1) {
        float* merge_buf = reinterpret_cast<float*>(k_smem);  // reuse K smem
        int warp_idx = tz * GROUP_SIZE + ty;

        // Store partial state
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            merge_buf[warp_idx * HEAD_DIM + tx * VEC_SIZE + i] = st.o[i];
        }
        // Store m/d after all o values: BDZ*GROUP*HEAD_DIM + offset
        merge_buf[BDZ * GROUP_SIZE * HEAD_DIM + warp_idx * 2]     = st.m_log2;
        merge_buf[BDZ * GROUP_SIZE * HEAD_DIM + warp_idx * 2 + 1] = st.d;
        __syncthreads();

        // Start with tz=0 state
        int widx0 = ty;  // tz=0, GROUP=ty
        st.m_log2 = merge_buf[BDZ * GROUP_SIZE * HEAD_DIM + widx0 * 2];
        st.d      = merge_buf[BDZ * GROUP_SIZE * HEAD_DIM + widx0 * 2 + 1];
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            st.o[i] = merge_buf[widx0 * HEAD_DIM + tx * VEC_SIZE + i];
        }

        // Merge other tz layers
        #pragma unroll
        for (int z = 1; z < BDZ; ++z) {
            int widx_z = z * GROUP_SIZE + ty;
            float mz = merge_buf[BDZ * GROUP_SIZE * HEAD_DIM + widx_z * 2];
            float dz = merge_buf[BDZ * GROUP_SIZE * HEAD_DIM + widx_z * 2 + 1];
            float* oz = merge_buf + widx_z * HEAD_DIM + tx * VEC_SIZE;
            state_merge(st, oz, mz, dz);
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

__global__ void combine_lse_kernel_v10(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_lse,
    __nv_bfloat16* __restrict__ output,
    int num_splits)
{
    const int tid = threadIdx.x;
    const int bh = blockIdx.x;

    if (num_splits <= 1) return;

    // Find max LSE
    float max_lse = -CUDART_INF_F;
    for (int s = 0; s < num_splits; ++s) {
        float lse = partial_lse[static_cast<int64_t>(bh) * num_splits + s];
        if (lse > max_lse) max_lse = lse;
    }

    // Compute weights
    float denominator = 0.f;
    float acc = 0.f;
    for (int s = 0; s < num_splits; ++s) {
        float lse = partial_lse[static_cast<int64_t>(bh) * num_splits + s];
        float w = (lse == -CUDART_INF_F) ? 0.f : exp2f(lse - max_lse);
        denominator += w;
        acc += w * partial_o[(static_cast<int64_t>(bh) * num_splits + s) * HEAD_DIM + tid];
    }

    float inv_denom = (denominator > 0.f) ? 1.f / denominator : 0.f;
    output[static_cast<int64_t>(bh) * HEAD_DIM + tid] = __float2bfloat16(acc * inv_denom);
}

// ─── Split heuristic ─────────────────────────────────────────────────────

static int choose_num_splits(int64_t batch_size, int seqlen) {
    if (seqlen <= 2048) return 1;

    if (seqlen <= 8192) {
        if (batch_size <= 1) return 4;
        if (batch_size <= 8) return 2;
        return 1;
    }

    if (seqlen <= 32768) {
        if (batch_size <= 1) return 32;
        if (batch_size <= 4) return 16;
        if (batch_size <= 8) return 8;
        return 4;
    }

    // Very long: ~4 pages per split
    int pages = (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;
    int splits = pages / 4;
    if (splits > MAX_SPLITS) splits = MAX_SPLITS;
    if (splits < 2) splits = 2;
    return splits;
}

// ─── Global partition buffers ────────────────────────────────────────────

static float* g_partial_o = nullptr;
static float* g_partial_lse = nullptr;
static size_t g_partial_o_cap = 0;
static size_t g_partial_lse_cap = 0;

static bool ensure_partition_buffers(size_t o_elems, size_t lse_elems) {
    if (o_elems > g_partial_o_cap) {
        if (g_partial_o) cudaFree(g_partial_o);
        if (cudaMalloc(&g_partial_o, o_elems * sizeof(float)) != cudaSuccess) return false;
        g_partial_o_cap = o_elems;
    }
    if (lse_elems > g_partial_lse_cap) {
        if (g_partial_lse) cudaFree(g_partial_lse);
        if (cudaMalloc(&g_partial_lse, lse_elems * sizeof(float)) != cudaSuccess) return false;
        g_partial_lse_cap = lse_elems;
    }
    return true;
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
    (void)seqlen_q; (void)headdim; (void)page_block_size; (void)causal;

    const int bs = static_cast<int>(batch_size);
    const int nh = static_cast<int>(num_heads);
    const int nkh = static_cast<int>(num_heads_k);
    const int group_size = nh / nkh;
    const int blocks_per_batch = static_cast<int>(num_blocks / batch_size);

    // BDZ: BDX(16) × GROUP × BDZ = 256  → GROUP=8→BDZ=2, GROUP=4→BDZ=4
    const int bdz = (group_size == 8) ? 2 : 4;

    dim3 block(BDX, static_cast<unsigned int>(group_size),
               static_cast<unsigned int>(bdz));

    // smem: 2 stages × (K+V) × BDZ × GROUP × HEAD_DIM × sizeof(bf16)
    // = 2 * 2 * bdz * group_size * 128 * 2 bytes = 4 * bdz * group_size * 256
    // Plus merge reuse of K smem (already included above)
    size_t smem_bytes = NUM_STAGES * 2 * bdz * group_size * HEAD_DIM * sizeof(__nv_bfloat16);

    int num_splits = choose_num_splits(batch_size, static_cast<int>(seqlen_k));

    if (num_splits <= 1) {
        // Direct mode
        dim3 grid(bs * nkh, 1);
        if (group_size == 8) {
            paged_decode_partition_kernel_v10<8, 2><<<grid, block, smem_bytes>>>(
                q, k_cache_paged, v_cache_paged,
                nullptr, nullptr, output,
                cache_seqlens, block_table,
                blocks_per_batch, nkh, 1);
        } else {
            paged_decode_partition_kernel_v10<4, 4><<<grid, block, smem_bytes>>>(
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
            paged_decode_partition_kernel_v10<8, 2><<<grid, block, smem_bytes>>>(
                q, k_cache_paged, v_cache_paged,
                nullptr, nullptr, output,
                cache_seqlens, block_table,
                blocks_per_batch, nkh, 1);
        } else {
            paged_decode_partition_kernel_v10<4, 4><<<grid, block, smem_bytes>>>(
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
        paged_decode_partition_kernel_v10<8, 2><<<part_grid, block, smem_bytes>>>(
            q, k_cache_paged, v_cache_paged,
            g_partial_o, g_partial_lse, output,
            cache_seqlens, block_table,
            blocks_per_batch, nkh, num_splits);
    } else {
        paged_decode_partition_kernel_v10<4, 4><<<part_grid, block, smem_bytes>>>(
            q, k_cache_paged, v_cache_paged,
            g_partial_o, g_partial_lse, output,
            cache_seqlens, block_table,
            blocks_per_batch, nkh, num_splits);
    }

    // Launch combine kernel
    combine_lse_kernel_v10<<<total_q_heads, HEAD_DIM>>>(
        g_partial_o, g_partial_lse, output, num_splits);
}

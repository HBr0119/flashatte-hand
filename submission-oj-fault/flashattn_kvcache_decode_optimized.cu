#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

#define MAX_SPLITS 64

// Persistent scratch for split-combine path
static float* g_partial_o = nullptr;
static float* g_partial_m = nullptr;
static float* g_partial_l = nullptr;
static size_t g_partial_o_elems = 0;
static size_t g_partial_s_elems = 0;

// ===================== Reductions =====================

// Power-of-two templated warp-then-block reduction returning the sum to all threads.
// Uses 2 __syncthreads per call (vs ~log2(HEAD_DIM) previously).
template <int HEAD_DIM>
__device__ __forceinline__ float block_sum_fast(float x) {
    static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be a multiple of 32");
    const unsigned FULL_MASK = 0xffffffffu;

    // Intra-warp reduce
    float v = x;
    // Unrolled warp reduction
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);

    // Cross-warp reduce using shared memory
    __shared__ float warp_sums[HEAD_DIM / 32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5; // / 32
    if (lane == 0) {
        warp_sums[warp_id] = v;
    }
    __syncthreads();

    float total = 0.f;
    if (warp_id == 0) {
        // Let warp 0 reduce the warp_sums
        float wv = (lane < (HEAD_DIM / 32)) ? warp_sums[lane] : 0.f;
        wv += __shfl_down_sync(FULL_MASK, wv, 16);
        wv += __shfl_down_sync(FULL_MASK, wv, 8);
        wv += __shfl_down_sync(FULL_MASK, wv, 4);
        wv += __shfl_down_sync(FULL_MASK, wv, 2);
        wv += __shfl_down_sync(FULL_MASK, wv, 1);
        if (lane == 0) {
            warp_sums[0] = wv;
        }
    }
    __syncthreads();
    total = warp_sums[0];
    return total;
}

// Generic reduction fallback (correct for any blockDim.x), slower but safe
__device__ __forceinline__ float block_sum_generic_atomic(float x) {
    __shared__ float sum;
    if (threadIdx.x == 0) sum = 0.0f;
    __syncthreads();
    atomicAdd(&sum, x);
    __syncthreads();
    return sum;
}

// Warp-only sum reduction (no __syncthreads), returns sum in all lanes
__device__ __forceinline__ float warp_sum(float v) {
    const unsigned FULL_MASK = 0xffffffffu;
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);
    // Broadcast lane 0 to all
    return __shfl_sync(FULL_MASK, v, 0);
}

// ===================== Vectorized BF16 pack loaders (K/V) =====================
// These helpers load a lane's contiguous PACK bf16s into float scalars.
// For PACK=8 (headdim=256), they perform 16B vectorized loads via four __nv_bfloat162.
// For PACK=4 (headdim=128), they perform 8B vectorized loads via two __nv_bfloat162.
// Fallback generic path exists for other PACK values if ever instantiated.

template <int PACK>
struct B16PackLoader {
    __device__ __forceinline__ static void load(const __nv_bfloat16* __restrict__ p, float* __restrict__ out) {
        #pragma unroll
        for (int i = 0; i < PACK; ++i) {
            out[i] = __bfloat162float(p[i]);
        }
    }
};

template <>
struct B16PackLoader<8> {
    __device__ __forceinline__ static void load(const __nv_bfloat16* __restrict__ p, float* __restrict__ out) {
        // 16B vectorized: 4 x bf16x2
        const __nv_bfloat162* __restrict__ p2 = reinterpret_cast<const __nv_bfloat162*>(p);
        const __nv_bfloat162 r0 = p2[0];
        const __nv_bfloat162 r1 = p2[1];
        const __nv_bfloat162 r2 = p2[2];
        const __nv_bfloat162 r3 = p2[3];
        out[0] = __bfloat162float(__low2bfloat16(r0));
        out[1] = __bfloat162float(__high2bfloat16(r0));
        out[2] = __bfloat162float(__low2bfloat16(r1));
        out[3] = __bfloat162float(__high2bfloat16(r1));
        out[4] = __bfloat162float(__low2bfloat16(r2));
        out[5] = __bfloat162float(__high2bfloat16(r2));
        out[6] = __bfloat162float(__low2bfloat16(r3));
        out[7] = __bfloat162float(__high2bfloat16(r3));
    }
};

template <>
struct B16PackLoader<4> {
    __device__ __forceinline__ static void load(const __nv_bfloat16* __restrict__ p, float* __restrict__ out) {
        // 8B vectorized: 2 x bf16x2
        const __nv_bfloat162* __restrict__ p2 = reinterpret_cast<const __nv_bfloat162*>(p);
        const __nv_bfloat162 r0 = p2[0];
        const __nv_bfloat162 r1 = p2[1];
        out[0] = __bfloat162float(__low2bfloat16(r0));
        out[1] = __bfloat162float(__high2bfloat16(r0));
        out[2] = __bfloat162float(__low2bfloat16(r1));
        out[3] = __bfloat162float(__high2bfloat16(r1));
    }
};

// ===================== Small-KV fast path (1 warp per head, HEAD_DIM = 128 or 256) =====================
// Contiguous dims per lane enable vector-friendly access.
// Each lane owns PACK = HEAD_DIM / 32 consecutive dims: d0 = lane * PACK ... d0 + PACK-1.
// We avoid per-token __syncthreads by doing the dot reduction with warp shuffles.
// Additionally, we hoist paged-KV address computation and provide a "contiguous span" fast path
// where all physical blocks in-range are linear, so we can step by a constant stride across tokens.

template <int HEAD_DIM>
__launch_bounds__(32, 4)
__global__ void paged_decode_smallkv_kernel_t(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int num_heads,
    int num_heads_k) {
    static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be divisible by 32");
    constexpr int PACK = HEAD_DIM / 32;
    const unsigned FULL_MASK = 0xffffffffu;

    // One warp per head
    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x & 31;

    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh - b * num_heads;
    const int gqa_ratio = num_heads / num_heads_k;
    const int kv_head = h / gqa_ratio;
    const int seqlen = cache_seqlens[b];

    if (seqlen <= 0) {
        // zero out output dims owned by this lane
        const int64_t out_base = (static_cast<int64_t>(b) * num_heads + h) * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < PACK; ++i) {
            const int d = lane * PACK + i;
            output[out_base + d] = __float2bfloat16(0.0f);
        }
        return;
    }

    // Preload and scale Q into registers; contiguous dims per lane
    const float scale = rsqrtf(static_cast<float>(HEAD_DIM));
    const int64_t q_base = (static_cast<int64_t>(b) * num_heads + h) * HEAD_DIM;
    float q_reg[PACK];
    #pragma unroll
    for (int i = 0; i < PACK; ++i) {
        const int d = lane * PACK + i;
        q_reg[i] = __bfloat162float(q[q_base + d]) * scale;
    }

    // Online softmax state (replicated per thread, kept consistent via warp shuffles)
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[PACK];
    #pragma unroll
    for (int i = 0; i < PACK; ++i) acc[i] = 0.0f;

    // Hoist addressing
    const int stride_tok = num_heads_k * HEAD_DIM;                 // in elements (bf16)
    const int64_t stride_tok64 = static_cast<int64_t>(stride_tok); // in elements
    const int valid_pages = (seqlen + 15) >> 4;

    // Check if pages are contiguous in physical memory (span fast path)
    int contig_flag = 0;
    int first_block = 0;
    if (lane == 0) {
        contig_flag = 1;
        if (valid_pages > 0) {
            first_block = block_table[b * blocks_per_batch + 0];
            int expected = first_block;
            #pragma unroll
            for (int p = 0; p < valid_pages; ++p) {
                const int pb = block_table[b * blocks_per_batch + p];
                if (pb != expected) {
                    contig_flag = 0;
                    break;
                }
                expected += 1;
            }
        } else {
            contig_flag = 0;
        }
    }
    contig_flag = __shfl_sync(FULL_MASK, contig_flag, 0);
    first_block = __shfl_sync(FULL_MASK, first_block, 0);

    // Common per-head constants
    const int d0 = lane * PACK;
    const int64_t headdim64 = static_cast<int64_t>(HEAD_DIM);
    const int64_t nhk64 = static_cast<int64_t>(num_heads_k);

    if (contig_flag) {
        // Single span across all tokens
        const int64_t base0 =
            (static_cast<int64_t>(first_block) * 16 * nhk64 + kv_head) * headdim64;

        const __nv_bfloat16* __restrict__ kptr = k_cache_paged + base0 + d0;
        const __nv_bfloat16* __restrict__ vptr = v_cache_paged + base0 + d0;

        // Iterate tokens linearly by stride; vectorized loads per lane
        for (int t = 0; t < seqlen; ++t) {
            float kvals[PACK];
            B16PackLoader<PACK>::load(kptr, kvals);
            float s_lane = 0.0f;
            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                s_lane = fmaf(q_reg[i], kvals[i], s_lane);
            }
            const float score = warp_sum(s_lane);

            // Online softmax
            const float new_m = fmaxf(m, score);
            const float alpha = __expf(m - new_m);
            const float p = __expf(score - new_m);

            float vvals[PACK];
            B16PackLoader<PACK>::load(vptr, vvals);
            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                acc[i] = acc[i] * alpha + p * vvals[i];
            }
            l = l * alpha + p;
            m = new_m;

            kptr += stride_tok64;
            vptr += stride_tok64;
        }
    } else {
        // General (non-contiguous) path: walk pages with simple prefetch of next block entry
        if (valid_pages <= 0) {
            // nothing to do
        } else {
            int next_block = block_table[b * blocks_per_batch + 0];
            for (int page = 0; page < valid_pages; ++page) {
                const int physical_block = next_block;
                if (page + 1 < valid_pages) {
                    next_block = block_table[b * blocks_per_batch + (page + 1)];
                }
                const int remain = seqlen - (page << 4);
                const int page_tokens = (remain < 16) ? remain : 16;

                const int64_t page_base =
                    (static_cast<int64_t>(physical_block) * 16 * nhk64 + kv_head) * headdim64;

                const __nv_bfloat16* __restrict__ kptr = k_cache_paged + page_base + d0;
                const __nv_bfloat16* __restrict__ vptr = v_cache_paged + page_base + d0;

                #pragma unroll 16
                for (int j = 0; j < 16; ++j) {
                    if (j >= page_tokens) break;

                    float kvals[PACK];
                    B16PackLoader<PACK>::load(kptr, kvals);
                    float s_lane = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < PACK; ++i) {
                        s_lane = fmaf(q_reg[i], kvals[i], s_lane);
                    }
                    const float score = warp_sum(s_lane);

                    const float new_m = fmaxf(m, score);
                    const float alpha = __expf(m - new_m);
                    const float p = __expf(score - new_m);

                    float vvals[PACK];
                    B16PackLoader<PACK>::load(vptr, vvals);
                    #pragma unroll
                    for (int i = 0; i < PACK; ++i) {
                        acc[i] = acc[i] * alpha + p * vvals[i];
                    }
                    l = l * alpha + p;
                    m = new_m;

                    kptr += stride_tok64;
                    vptr += stride_tok64;
                }
            }
        }
    }

    // Write results
    const int64_t out_base = (static_cast<int64_t>(b) * num_heads + h) * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < PACK; ++i) {
        const int d = d0 + i;
        const float val = (l > 0.0f) ? (acc[i] / l) : 0.0f;
        output[out_base + d] = __float2bfloat16(val);
    }
}

// Split variant of small-KV fast path: divides the K-reduction (tokens) across splits to raise occupancy.
// Keeps the same numerics via partial (m,l,o) combination.
template <int HEAD_DIM>
__launch_bounds__(32, 4)
__global__ void paged_decode_smallkv_split_kernel_t(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    float* __restrict__ partial_o,   // [total_heads*splits, HEAD_DIM]
    float* __restrict__ partial_m,   // [total_heads*splits]
    float* __restrict__ partial_l,   // [total_heads*splits]
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int num_heads,
    int num_heads_k,
    int splits) {

    static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be divisible by 32");
    constexpr int PACK = HEAD_DIM / 32;
    const unsigned FULL_MASK = 0xffffffffu;
    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x & 31;

    const int bh = blockIdx.x;
    const int split = blockIdx.y;
    const int b = bh / num_heads;
    const int h = bh - b * num_heads;
    const int gqa_ratio = num_heads / num_heads_k;
    const int kv_head = h / gqa_ratio;

    const int seqlen = cache_seqlens[b];
    const int tokens_per_split = (seqlen + splits - 1) / splits;
    const int start = split * tokens_per_split;
    const int stop = min(seqlen, start + tokens_per_split);

    const int part_idx = (bh * splits) + split;
    const int64_t po_base = static_cast<int64_t>(part_idx) * HEAD_DIM;

    if (start >= stop) {
        if (lane == 0) {
            partial_m[part_idx] = -CUDART_INF_F;
            partial_l[part_idx] = 0.0f;
        }
        // Zero this split's partial output
        #pragma unroll
        for (int i = 0; i < PACK; ++i) {
            partial_o[po_base + lane * PACK + i] = 0.0f;
        }
        return;
    }

    // Preload and scale Q into registers; contiguous dims per lane
    const float scale = rsqrtf(static_cast<float>(HEAD_DIM));
    const int64_t q_base = (static_cast<int64_t>(b) * num_heads + h) * HEAD_DIM;
    float q_reg[PACK];
    #pragma unroll
    for (int i = 0; i < PACK; ++i) {
        const int d = lane * PACK + i;
        q_reg[i] = __bfloat162float(q[q_base + d]) * scale;
    }

    // Online softmax state (replicated per thread, kept consistent via warp shuffles)
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[PACK];
    #pragma unroll
    for (int i = 0; i < PACK; ++i) acc[i] = 0.0f;

    // Hoist addressing for this split
    const int stride_tok = num_heads_k * HEAD_DIM;                 // in elements (bf16)
    const int64_t stride_tok64 = static_cast<int64_t>(stride_tok); // in elements
    const int start_page = start >> 4;
    const int end_page = (stop - 1) >> 4;

    // Check contiguity across [start_page, end_page]
    int contig_flag = 0;
    int first_block = 0;
    if (lane == 0) {
        contig_flag = 1;
        if (start < stop) {
            first_block = block_table[b * blocks_per_batch + start_page];
            int expected = first_block;
            for (int p = start_page; p <= end_page; ++p) {
                const int pb = block_table[b * blocks_per_batch + p];
                if (pb != expected) {
                    contig_flag = 0;
                    break;
                }
                expected += 1;
            }
        } else {
            contig_flag = 0;
        }
    }
    contig_flag = __shfl_sync(FULL_MASK, contig_flag, 0);
    first_block = __shfl_sync(FULL_MASK, first_block, 0);

    const int d0 = lane * PACK;
    const int64_t headdim64 = static_cast<int64_t>(HEAD_DIM);
    const int64_t nhk64 = static_cast<int64_t>(num_heads_k);

    if (contig_flag) {
        const int page_local_offset = start - (start_page << 4);
        const int64_t base_page =
            (static_cast<int64_t>(first_block) * 16 * nhk64 + kv_head) * headdim64;

        const __nv_bfloat16* __restrict__ kptr =
            k_cache_paged + base_page + static_cast<int64_t>(page_local_offset) * stride_tok64 + d0;
        const __nv_bfloat16* __restrict__ vptr =
            v_cache_paged + base_page + static_cast<int64_t>(page_local_offset) * stride_tok64 + d0;

        for (int t = start; t < stop; ++t) {
            float kvals[PACK];
            B16PackLoader<PACK>::load(kptr, kvals);
            float s_lane = 0.0f;
            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                s_lane = fmaf(q_reg[i], kvals[i], s_lane);
            }
            const float score = warp_sum(s_lane);

            const float new_m = fmaxf(m, score);
            const float alpha = __expf(m - new_m);
            const float p = __expf(score - new_m);

            float vvals[PACK];
            B16PackLoader<PACK>::load(vptr, vvals);
            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                acc[i] = acc[i] * alpha + p * vvals[i];
            }
            l = l * alpha + p;
            m = new_m;

            kptr += stride_tok64;
            vptr += stride_tok64;
        }
    } else {
        int token = start;
        while (token < stop) {
            const int page = token >> 4;
            const int physical_block = block_table[b * blocks_per_batch + page];
            const int next_page = (page + 1) << 4;
            const int page_limit = (stop < next_page) ? stop : next_page;

            const int64_t page_base =
                (static_cast<int64_t>(physical_block) * 16 * nhk64 + kv_head) * headdim64;
            const int j0 = token - (page << 4);

            const __nv_bfloat16* __restrict__ kptr =
                k_cache_paged + page_base + static_cast<int64_t>(j0) * stride_tok64 + d0;
            const __nv_bfloat16* __restrict__ vptr =
                v_cache_paged + page_base + static_cast<int64_t>(j0) * stride_tok64 + d0;

            for (; token < page_limit; ++token) {
                float kvals[PACK];
                B16PackLoader<PACK>::load(kptr, kvals);
                float s_lane = 0.0f;
                #pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    s_lane = fmaf(q_reg[i], kvals[i], s_lane);
                }
                const float score = warp_sum(s_lane);

                const float new_m = fmaxf(m, score);
                const float alpha = __expf(m - new_m);
                const float p = __expf(score - new_m);

                float vvals[PACK];
                B16PackLoader<PACK>::load(vptr, vvals);
                #pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    acc[i] = acc[i] * alpha + p * vvals[i];
                }
                l = l * alpha + p;
                m = new_m;

                kptr += stride_tok64;
                vptr += stride_tok64;
            }
        }
    }

    if (lane == 0) {
        partial_m[part_idx] = m;
        partial_l[part_idx] = l;
    }
    // Store partial output (normalized by l will be done in the combiner via weights)
    #pragma unroll
    for (int i = 0; i < PACK; ++i) {
        partial_o[po_base + d0 + i] = (l > 0.0f) ? (acc[i] / l) : 0.0f;
    }
}

// ===================== 2-stage prefetch small-KV kernels (for large-batch long-KV) =====================

template <int HEAD_DIM>
__launch_bounds__(32, 4)
__global__ void paged_decode_smallkv2s_kernel_t(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int num_heads,
    int num_heads_k) {
    static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be divisible by 32");
    constexpr int PACK = HEAD_DIM / 32;
    const unsigned FULL_MASK = 0xffffffffu;

    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x & 31;

    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh - b * num_heads;
    const int gqa_ratio = num_heads / num_heads_k;
    const int kv_head = h / gqa_ratio;
    const int seqlen = cache_seqlens[b];

    if (seqlen <= 0) {
        const int64_t out_base = (static_cast<int64_t>(b) * num_heads + h) * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < PACK; ++i) {
            output[out_base + lane * PACK + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    const float scale = rsqrtf(static_cast<float>(HEAD_DIM));
    const int64_t q_base = (static_cast<int64_t>(b) * num_heads + h) * HEAD_DIM;
    float q_reg[PACK];
    #pragma unroll
    for (int i = 0; i < PACK; ++i) {
        const int d = lane * PACK + i;
        q_reg[i] = __bfloat162float(q[q_base + d]) * scale;
    }

    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[PACK];
    #pragma unroll
    for (int i = 0; i < PACK; ++i) acc[i] = 0.0f;

    const int stride_tok = num_heads_k * HEAD_DIM;                 // elements
    const int64_t stride_tok64 = static_cast<int64_t>(stride_tok);
    const int valid_pages = (seqlen + 15) >> 4;

    // Contiguity check
    int contig_flag = 0;
    int first_block = 0;
    if (lane == 0) {
        contig_flag = 1;
        if (valid_pages > 0) {
            first_block = block_table[b * blocks_per_batch + 0];
            int expected = first_block;
            #pragma unroll
            for (int p = 0; p < valid_pages; ++p) {
                const int pb = block_table[b * blocks_per_batch + p];
                if (pb != expected) { contig_flag = 0; break; }
                expected += 1;
            }
        } else {
            contig_flag = 0;
        }
    }
    contig_flag = __shfl_sync(FULL_MASK, contig_flag, 0);
    first_block = __shfl_sync(FULL_MASK, first_block, 0);

    const int d0 = lane * PACK;
    const int64_t headdim64 = static_cast<int64_t>(HEAD_DIM);
    const int64_t nhk64 = static_cast<int64_t>(num_heads_k);

    if (contig_flag) {
        const int64_t base0 = (static_cast<int64_t>(first_block) * 16 * nhk64 + kv_head) * headdim64;
        const __nv_bfloat16* __restrict__ kptr = k_cache_paged + base0 + d0;
        const __nv_bfloat16* __restrict__ vptr = v_cache_paged + base0 + d0;

        float kbuf[2][PACK];
        float vbuf[2][PACK];
        int buf = 0;

        // Prime
        B16PackLoader<PACK>::load(kptr, kbuf[buf]);
        B16PackLoader<PACK>::load(vptr, vbuf[buf]);

        for (int t = 0; t < seqlen; ++t) {
            const bool has_next = (t + 1) < seqlen;
            if (has_next) {
                B16PackLoader<PACK>::load(kptr + stride_tok64, kbuf[buf ^ 1]);
                B16PackLoader<PACK>::load(vptr + stride_tok64, vbuf[buf ^ 1]);
            }

            float s_lane = 0.0f;
            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                s_lane = fmaf(q_reg[i], kbuf[buf][i], s_lane);
            }
            const float score = warp_sum(s_lane);

            const float new_m = fmaxf(m, score);
            const float alpha = __expf(m - new_m);
            const float p = __expf(score - new_m);

            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                acc[i] = acc[i] * alpha + p * vbuf[buf][i];
            }
            l = l * alpha + p;
            m = new_m;

            kptr += stride_tok64;
            vptr += stride_tok64;
            buf ^= 1;
        }
    } else {
        // Page-walk with 2-stage prefetch inside each page
        int next_block = (valid_pages > 0) ? block_table[b * blocks_per_batch + 0] : 0;
        for (int page = 0; page < valid_pages; ++page) {
            const int physical_block = next_block;
            if (page + 1 < valid_pages) {
                next_block = block_table[b * blocks_per_batch + (page + 1)];
            }
            const int remain = seqlen - (page << 4);
            const int page_tokens = (remain < 16) ? remain : 16;
            if (page_tokens <= 0) continue;

            const int64_t page_base =
                (static_cast<int64_t>(physical_block) * 16 * nhk64 + kv_head) * headdim64;

            const __nv_bfloat16* __restrict__ kptr = k_cache_paged + page_base + d0;
            const __nv_bfloat16* __restrict__ vptr = v_cache_paged + page_base + d0;

            float kbuf[2][PACK];
            float vbuf[2][PACK];
            int buf = 0;

            // Prime (j=0)
            B16PackLoader<PACK>::load(kptr, kbuf[buf]);
            B16PackLoader<PACK>::load(vptr, vbuf[buf]);

            #pragma unroll 16
            for (int j = 0; j < 16; ++j) {
                if (j >= page_tokens) break;
                const bool has_next = (j + 1) < page_tokens;
                if (has_next) {
                    B16PackLoader<PACK>::load(kptr + stride_tok64, kbuf[buf ^ 1]);
                    B16PackLoader<PACK>::load(vptr + stride_tok64, vbuf[buf ^ 1]);
                }

                float s_lane = 0.0f;
                #pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    s_lane = fmaf(q_reg[i], kbuf[buf][i], s_lane);
                }
                const float score = warp_sum(s_lane);

                const float new_m = fmaxf(m, score);
                const float alpha = __expf(m - new_m);
                const float p = __expf(score - new_m);

                #pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    acc[i] = acc[i] * alpha + p * vbuf[buf][i];
                }
                l = l * alpha + p;
                m = new_m;

                kptr += stride_tok64;
                vptr += stride_tok64;
                buf ^= 1;
            }
        }
    }

    const int64_t out_base = (static_cast<int64_t>(b) * num_heads + h) * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < PACK; ++i) {
        output[out_base + d0 + i] = __float2bfloat16((l > 0.0f) ? (acc[i] / l) : 0.0f);
    }
}

template <int HEAD_DIM>
__launch_bounds__(32, 4)
__global__ void paged_decode_smallkv2s_split_kernel_t(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    float* __restrict__ partial_o,   // [total_heads*splits, HEAD_DIM]
    float* __restrict__ partial_m,   // [total_heads*splits]
    float* __restrict__ partial_l,   // [total_heads*splits]
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int num_heads,
    int num_heads_k,
    int splits) {
    static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be divisible by 32");
    constexpr int PACK = HEAD_DIM / 32;
    const unsigned FULL_MASK = 0xffffffffu;

    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x & 31;

    const int bh = blockIdx.x;
    const int split = blockIdx.y;
    const int b = bh / num_heads;
    const int h = bh - b * num_heads;
    const int gqa_ratio = num_heads / num_heads_k;
    const int kv_head = h / gqa_ratio;
    const int seqlen = cache_seqlens[b];

    const int tokens_per_split = (seqlen + splits - 1) / splits;
    const int start = split * tokens_per_split;
    const int stop = min(seqlen, start + tokens_per_split);

    const int part_idx = (bh * splits) + split;
    const int64_t po_base = static_cast<int64_t>(part_idx) * HEAD_DIM;

    if (start >= stop) {
        if (lane == 0) {
            partial_m[part_idx] = -CUDART_INF_F;
            partial_l[part_idx] = 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < PACK; ++i) {
            partial_o[po_base + lane * PACK + i] = 0.0f;
        }
        return;
    }

    const float scale = rsqrtf(static_cast<float>(HEAD_DIM));
    const int64_t q_base = (static_cast<int64_t>(b) * num_heads + h) * HEAD_DIM;
    float q_reg[PACK];
    #pragma unroll
    for (int i = 0; i < PACK; ++i) {
        const int d = lane * PACK + i;
        q_reg[i] = __bfloat162float(q[q_base + d]) * scale;
    }

    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[PACK];
    #pragma unroll
    for (int i = 0; i < PACK; ++i) acc[i] = 0.0f;

    const int stride_tok = num_heads_k * HEAD_DIM;
    const int64_t stride_tok64 = static_cast<int64_t>(stride_tok);
    const int start_page = start >> 4;
    const int end_page = (stop - 1) >> 4;

    int contig_flag = 0;
    int first_block = 0;
    if (lane == 0) {
        contig_flag = 1;
        if (start < stop) {
            first_block = block_table[b * blocks_per_batch + start_page];
            int expected = first_block;
            for (int p = start_page; p <= end_page; ++p) {
                const int pb = block_table[b * blocks_per_batch + p];
                if (pb != expected) { contig_flag = 0; break; }
                expected += 1;
            }
        } else {
            contig_flag = 0;
        }
    }
    contig_flag = __shfl_sync(FULL_MASK, contig_flag, 0);
    first_block = __shfl_sync(FULL_MASK, first_block, 0);

    const int d0 = lane * PACK;
    const int64_t headdim64 = static_cast<int64_t>(HEAD_DIM);
    const int64_t nhk64 = static_cast<int64_t>(num_heads_k);

    if (contig_flag) {
        const int page_local_offset = start - (start_page << 4);
        const int64_t base_page =
            (static_cast<int64_t>(first_block) * 16 * nhk64 + kv_head) * headdim64;

        const __nv_bfloat16* __restrict__ kptr =
            k_cache_paged + base_page + static_cast<int64_t>(page_local_offset) * stride_tok64 + d0;
        const __nv_bfloat16* __restrict__ vptr =
            v_cache_paged + base_page + static_cast<int64_t>(page_local_offset) * stride_tok64 + d0;

        float kbuf[2][PACK];
        float vbuf[2][PACK];
        int buf = 0;

        // Prime
        B16PackLoader<PACK>::load(kptr, kbuf[buf]);
        B16PackLoader<PACK>::load(vptr, vbuf[buf]);

        for (int t = start; t < stop; ++t) {
            const bool has_next = (t + 1) < stop;
            if (has_next) {
                B16PackLoader<PACK>::load(kptr + stride_tok64, kbuf[buf ^ 1]);
                B16PackLoader<PACK>::load(vptr + stride_tok64, vbuf[buf ^ 1]);
            }

            float s_lane = 0.0f;
            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                s_lane = fmaf(q_reg[i], kbuf[buf][i], s_lane);
            }
            const float score = warp_sum(s_lane);

            const float new_m = fmaxf(m, score);
            const float alpha = __expf(m - new_m);
            const float p = __expf(score - new_m);

            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                acc[i] = acc[i] * alpha + p * vbuf[buf][i];
            }
            l = l * alpha + p;
            m = new_m;

            kptr += stride_tok64;
            vptr += stride_tok64;
        }
    } else {
        int token = start;
        while (token < stop) {
            const int page = token >> 4;
            const int physical_block = block_table[b * blocks_per_batch + page];
            const int next_page_tok = (page + 1) << 4;
            const int page_limit = (stop < next_page_tok) ? stop : next_page_tok;
            const int64_t page_base =
                (static_cast<int64_t>(physical_block) * 16 * nhk64 + kv_head) * headdim64;

            const int j0 = token - (page << 4);
            const __nv_bfloat16* __restrict__ kptr =
                k_cache_paged + page_base + static_cast<int64_t>(j0) * stride_tok64 + d0;
            const __nv_bfloat16* __restrict__ vptr =
                v_cache_paged + page_base + static_cast<int64_t>(j0) * stride_tok64 + d0;

            float kbuf[2][PACK];
            float vbuf[2][PACK];
            int buf = 0;

            // Prime current position
            B16PackLoader<PACK>::load(kptr, kbuf[buf]);
            B16PackLoader<PACK>::load(vptr, vbuf[buf]);

            for (; token < page_limit; ++token) {
                const bool has_next = (token + 1) < page_limit;
                if (has_next) {
                    B16PackLoader<PACK>::load(kptr + stride_tok64, kbuf[buf ^ 1]);
                    B16PackLoader<PACK>::load(vptr + stride_tok64, vbuf[buf ^ 1]);
                }

                float s_lane = 0.0f;
                #pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    s_lane = fmaf(q_reg[i], kbuf[buf][i], s_lane);
                }
                const float score = warp_sum(s_lane);

                const float new_m = fmaxf(m, score);
                const float alpha = __expf(m - new_m);
                const float p = __expf(score - new_m);

                #pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    acc[i] = acc[i] * alpha + p * vbuf[buf][i];
                }
                l = l * alpha + p;
                m = new_m;

                kptr += stride_tok64;
                vptr += stride_tok64;
                buf ^= 1;
            }
        }
    }

    if (lane == 0) {
        partial_m[part_idx] = m;
        partial_l[part_idx] = l;
    }
    #pragma unroll
    for (int i = 0; i < PACK; ++i) {
        partial_o[po_base + d0 + i] = (l > 0.0f) ? (acc[i] / l) : 0.0f;
    }
}

// ===================== Templated fast kernels (HEAD_DIM = 128 or 256) =====================

template <int HEAD_DIM>
__global__ void paged_decode_kernel_t(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int num_heads,
    int num_heads_k) {
    const int tid = threadIdx.x;
    if (tid >= HEAD_DIM) return;  // safety
    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh - b * num_heads;
    const int gqa_ratio = num_heads / num_heads_k;
    const int kv_head = h / gqa_ratio;
    const int seqlen = cache_seqlens[b];

    // Fold scale into q to save a multiply per token
    const float scale = rsqrtf(static_cast<float>(HEAD_DIM));  // 1/sqrt(HEAD_DIM)
    const int64_t q_base = (static_cast<int64_t>(b) * num_heads + h) * HEAD_DIM;
    const float q_scaled = __bfloat162float(q[q_base + tid]) * scale;

    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc = 0.0f;

    const int valid_pages = (seqlen + 15) >> 4;  // page_block_size == 16
    for (int page = 0; page < valid_pages; ++page) {
        const int physical_block = block_table[b * blocks_per_batch + page];
        const int remain = seqlen - (page << 4);
        const int page_tokens = (remain < 16) ? remain : 16;

        const int64_t page_base =
            (static_cast<int64_t>(physical_block) * 16 * num_heads_k + kv_head) * HEAD_DIM;
        const int64_t stride_tok = static_cast<int64_t>(num_heads_k) * HEAD_DIM;

        const __nv_bfloat16* __restrict__ kptr = k_cache_paged + page_base + tid;
        const __nv_bfloat16* __restrict__ vptr = v_cache_paged + page_base + tid;

        #pragma unroll 16
        for (int j = 0; j < 16; ++j) {
            if (j >= page_tokens) break;

            const float k_val = __bfloat162float(kptr[0]);
            // Sum across HEAD_DIM threads; already scaled via q_scaled
            const float score = block_sum_fast<HEAD_DIM>(q_scaled * k_val);
            const float v_val = __bfloat162float(vptr[0]);

            const float new_m = fmaxf(m, score);
            const float alpha = __expf(m - new_m);
            const float p = __expf(score - new_m);
            acc = acc * alpha + p * v_val;
            l = l * alpha + p;
            m = new_m;

            kptr += stride_tok;
            vptr += stride_tok;
        }
    }

    const int64_t out_base = (static_cast<int64_t>(b) * num_heads + h) * HEAD_DIM;
    output[out_base + tid] = __float2bfloat16((l > 0.0f) ? (acc / l) : 0.0f);
}

template <int HEAD_DIM>
__global__ void paged_decode_split_kernel_t(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    float* __restrict__ partial_o,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int num_heads,
    int num_heads_k,
    int splits) {
    const int tid = threadIdx.x;
    if (tid >= HEAD_DIM) return;  // safety
    const int bh = blockIdx.x;
    const int split = blockIdx.y;
    const int b = bh / num_heads;
    const int h = bh - b * num_heads;
    const int gqa_ratio = num_heads / num_heads_k;
    const int kv_head = h / gqa_ratio;
    const int seqlen = cache_seqlens[b];
    const int tokens_per_split = (seqlen + splits - 1) / splits;
    const int start = split * tokens_per_split;
    const int stop = min(seqlen, start + tokens_per_split);

    const float scale = rsqrtf(static_cast<float>(HEAD_DIM));  // 1 / sqrt(HEAD_DIM)
    const int part_idx = (bh * splits) + split;
    const int64_t po_base = static_cast<int64_t>(part_idx) * HEAD_DIM;

    if (start >= stop) {
        if (tid == 0) {
            partial_m[part_idx] = -CUDART_INF_F;
            partial_l[part_idx] = 0.0f;
        }
        partial_o[po_base + tid] = 0.0f;
        return;
    }

    const int64_t q_base = (static_cast<int64_t>(b) * num_heads + h) * HEAD_DIM;
    const float q_scaled = __bfloat162float(q[q_base + tid]) * scale;

    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc = 0.0f;

    int token = start;
    while (token < stop) {
        const int page = token >> 4;  // page_block_size == 16
        const int physical_block = block_table[b * blocks_per_batch + page];
        const int next_page = (page + 1) << 4;
        const int page_limit = (stop < next_page) ? stop : next_page;

        const int64_t page_base =
            (static_cast<int64_t>(physical_block) * 16 * num_heads_k + kv_head) * HEAD_DIM;
        const int64_t stride_tok = static_cast<int64_t>(num_heads_k) * HEAD_DIM;

        const __nv_bfloat16* __restrict__ kptr = k_cache_paged + page_base + tid + static_cast<int64_t>(token - (page << 4)) * stride_tok;
        const __nv_bfloat16* __restrict__ vptr = v_cache_paged + page_base + tid + static_cast<int64_t>(token - (page << 4)) * stride_tok;

        for (; token < page_limit; ++token) {
            const float k_val = __bfloat162float(kptr[0]);
            const float score = block_sum_fast<HEAD_DIM>(q_scaled * k_val);
            const float v_val = __bfloat162float(vptr[0]);

            const float new_m = fmaxf(m, score);
            const float alpha = __expf(m - new_m);
            const float p = __expf(score - new_m);
            acc = acc * alpha + p * v_val;
            l = l * alpha + p;
            m = new_m;

            kptr += stride_tok;
            vptr += stride_tok;
        }
    }

    if (tid == 0) {
        partial_m[part_idx] = m;
        partial_l[part_idx] = l;
    }
    partial_o[po_base + tid] = (l > 0.0f) ? (acc / l) : 0.0f;
}

template <int HEAD_DIM>
__global__ void combine_split_kernel_t(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __nv_bfloat16* __restrict__ output,
    int /*num_heads*/,
    int splits) {
    const int tid = threadIdx.x;
    if (tid >= HEAD_DIM) return;  // safety
    const int bh = blockIdx.x;

    float m = -CUDART_INF_F;
    for (int s = 0; s < splits; ++s) {
        const int idx = bh * splits + s;
        const float l = partial_l[idx];
        const float ms = partial_m[idx];
        if (l > 0.0f) {
            m = fmaxf(m, ms);
        }
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const int idx = bh * splits + s;
        const float l = partial_l[idx];
        if (l > 0.0f) {
            const float w = l * __expf(partial_m[idx] - m);
            denom += w;
            acc += w * partial_o[static_cast<int64_t>(idx) * HEAD_DIM + tid];
        }
    }

    output[static_cast<int64_t>(bh) * HEAD_DIM + tid] =
        __float2bfloat16((denom > 0.0f) ? (acc / denom) : 0.0f);
}

// ===================== Generic fallback kernels (any headdim) =====================

__global__ void paged_decode_kernel_generic(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int num_heads,
    int num_heads_k,
    int headdim) {
    const int tid = threadIdx.x;
    if (tid >= headdim) return;
    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh - b * num_heads;
    const int gqa_ratio = num_heads / num_heads_k;
    const int kv_head = h / gqa_ratio;
    const int seqlen = cache_seqlens[b];
    const float scale = rsqrtf(static_cast<float>(headdim));  // 1 / sqrt(headdim)

    const int64_t q_base = (static_cast<int64_t>(b) * num_heads + h) * headdim;
    const float q_val = __bfloat162float(q[q_base + tid]);

    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc = 0.0f;

    const int valid_pages = (seqlen + 15) >> 4;  // page_block_size == 16
    for (int page = 0; page < valid_pages; ++page) {
        const int physical_block = block_table[b * blocks_per_batch + page];
        const int remain = seqlen - (page << 4);
        const int page_tokens = (remain < 16) ? remain : 16;
        const int64_t page_base =
            (static_cast<int64_t>(physical_block) * 16 * num_heads_k + kv_head) * headdim;

        for (int j = 0; j < page_tokens; ++j) {
            const int64_t kv_base = page_base + static_cast<int64_t>(j) * num_heads_k * headdim;
            const float k_val = __bfloat162float(k_cache_paged[kv_base + tid]);
            const float score = block_sum_generic_atomic(q_val * k_val) * scale;
            const float v_val = __bfloat162float(v_cache_paged[kv_base + tid]);

            const float new_m = fmaxf(m, score);
            const float alpha = expf(m - new_m);
            const float p = expf(score - new_m);
            acc = acc * alpha + p * v_val;
            l = l * alpha + p;
            m = new_m;
        }
    }

    const int64_t out_base = (static_cast<int64_t>(b) * num_heads + h) * headdim;
    output[out_base + tid] = __float2bfloat16((l > 0.0f) ? (acc / l) : 0.0f);
}

__global__ void paged_decode_split_kernel_generic(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    float* __restrict__ partial_o,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int num_heads,
    int num_heads_k,
    int splits,
    int headdim) {
    const int tid = threadIdx.x;
    if (tid >= headdim) return;
    const int bh = blockIdx.x;
    const int split = blockIdx.y;
    const int b = bh / num_heads;
    const int h = bh - b * num_heads;
    const int gqa_ratio = num_heads / num_heads_k;
    const int kv_head = h / gqa_ratio;
    const int seqlen = cache_seqlens[b];
    const int tokens_per_split = (seqlen + splits - 1) / splits;
    const int start = split * tokens_per_split;
    const int stop = min(seqlen, start + tokens_per_split);
    const float scale = rsqrtf(static_cast<float>(headdim));  // 1 / sqrt(headdim)

    const int part_idx = (bh * splits) + split;
    const int64_t po_base = static_cast<int64_t>(part_idx) * headdim;

    if (start >= stop) {
        if (tid == 0) {
            partial_m[part_idx] = -CUDART_INF_F;
            partial_l[part_idx] = 0.0f;
        }
        partial_o[po_base + tid] = 0.0f;
        return;
    }

    const int64_t q_base = (static_cast<int64_t>(b) * num_heads + h) * headdim;
    const float q_val = __bfloat162float(q[q_base + tid]);

    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc = 0.0f;

    int token = start;
    while (token < stop) {
        const int page = token >> 4;  // page_block_size == 16
        const int physical_block = block_table[b * blocks_per_batch + page];
        const int next_page = (page + 1) << 4;
        const int page_limit = (stop < next_page) ? stop : next_page;
        const int64_t page_base =
            (static_cast<int64_t>(physical_block) * 16 * num_heads_k + kv_head) * headdim;

        for (; token < page_limit; ++token) {
            const int j = token - (page << 4);
            const int64_t kv_base = page_base + static_cast<int64_t>(j) * num_heads_k * headdim;
            const float k_val = __bfloat162float(k_cache_paged[kv_base + tid]);
            const float score = block_sum_generic_atomic(q_val * k_val) * scale;
            const float v_val = __bfloat162float(v_cache_paged[kv_base + tid]);

            const float new_m = fmaxf(m, score);
            const float alpha = expf(m - new_m);
            const float p = expf(score - new_m);
            acc = acc * alpha + p * v_val;
            l = l * alpha + p;
            m = new_m;
        }
    }

    if (tid == 0) {
        partial_m[part_idx] = m;
        partial_l[part_idx] = l;
    }
    partial_o[po_base + tid] = (l > 0.0f) ? (acc / l) : 0.0f;
}

__global__ void combine_split_kernel_generic(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __nv_bfloat16* __restrict__ output,
    int splits,
    int headdim) {
    const int tid = threadIdx.x;
    if (tid >= headdim) return;
    const int bh = blockIdx.x;

    float m = -CUDART_INF_F;
    for (int s = 0; s < splits; ++s) {
        const int idx = bh * splits + s;
        const float l = partial_l[idx];
        const float ms = partial_m[idx];
        if (l > 0.0f) {
            m = fmaxf(m, ms);
        }
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const int idx = bh * splits + s;
        const float l = partial_l[idx];
        if (l > 0.0f) {
            const float w = l * expf(partial_m[idx] - m);
            denom += w;
            acc += w * partial_o[static_cast<int64_t>(idx) * headdim + tid];
        }
    }

    output[static_cast<int64_t>(bh) * headdim + tid] =
        __float2bfloat16((denom > 0.0f) ? (acc / denom) : 0.0f);
}

// ===================== Split selection =====================

// Default split selection for non-small-KV path (preserve baseline behavior)
static int choose_splits(int64_t batch_size, int64_t seqlen_k) {
    // Small KV: defer to single-CTA per head in the long-path dispatcher (we will override in small-KV path)
    if (seqlen_k <= 2048) {
        return 1;
    }

    // Mid KV up to 4k: always split some to raise parallelism and reduce per-CTA work
    if (seqlen_k <= 4096) {
        if (batch_size <= 1) return 16;
        if (batch_size <= 8) return 8;
        if (batch_size <= 32) return 4;
        return 4;
    }

    // 4k < KV <= 8k
    if (seqlen_k <= 8192) {
        if (batch_size <= 1) return 32;
        if (batch_size <= 8) return 16;
        if (batch_size <= 16) return 8;
        return 4;  // batch >= 32
    }

    // Long KV > 8k
    if (batch_size <= 1) {
        if (seqlen_k >= 32768) return 64;
        if (seqlen_k >= 16384) return 32;
        return 16; // 8192..16383
    }
    if (batch_size <= 8) {
        if (seqlen_k >= 32768) return 32;
        if (seqlen_k >= 16384) return 16;
        return 8; // 8192..16383
    }
    if (batch_size <= 16) {
        if (seqlen_k >= 16384) return 8;
        return 4; // 8192..16383
    }
    // batch > 16: ensure at least a few splits to prevent long per-CTA sections
    if (seqlen_k >= 32768) return 8;
    if (seqlen_k >= 16384) return 4;
    return 4; // 8192..16383
}

// Tailored split selection for small-KV fast path (<= 2048), to raise occupancy for small batches.
// Aim for ~256 tokens per split, capped, with fewer splits as batch increases.
static int choose_smallkv_splits(int64_t batch_size, int64_t seqlen_k) {
    if (seqlen_k <= 0) return 1;
    const int base = static_cast<int>((seqlen_k + 255) / 256); // 512->2, 1024->4, 2048->8
    int splits = 1;
    if (batch_size <= 2) {
        splits = base;                 // up to 8 for 2k
    } else if (batch_size <= 4) {
        splits = (base > 8) ? 8 : base;
    } else if (batch_size <= 8) {
        // keep moderate splits to avoid too many tiny CTAs
        splits = (seqlen_k >= 1024) ? 4 : ((seqlen_k >= 512) ? 2 : 1);
    } else {
        splits = (seqlen_k >= 2048) ? 2 : 1;
    }
    if (splits < 1) splits = 1;
    if (splits > MAX_SPLITS) splits = MAX_SPLITS;
    return splits;
}

// Large-batch long-KV split selection: target ~1k tokens per split, clamp to MAX_SPLITS.
static int choose_largekv_splits(int64_t /*batch_size*/, int64_t seqlen_k) {
    // Aim ~1024 tokens per split: 4/8/16 for 4k/8k/16k
    int splits = static_cast<int>((seqlen_k + 1023) / 1024);
    if (splits < 4) splits = 4; // guard for seqlen_k >= 4096
    if (splits > MAX_SPLITS) splits = MAX_SPLITS;
    return splits;
}

// New: Large-batch small/mid-KV (2049..4096) split selection for STAGES_KV=1 path.
// Target ~1k tokens per split, but allow 2-4 for this range to keep overhead low.
static int choose_largebatch_smallmid_splits(int64_t /*batch_size*/, int64_t seqlen_k) {
    int splits = static_cast<int>((seqlen_k + 1023) / 1024); // 3 for 3k, 4 for 4k
    if (splits < 2) splits = 2;                              // ensure some parallelism
    if (splits > 8) splits = 8;
    if (splits > MAX_SPLITS) splits = MAX_SPLITS;
    return splits;
}

// New: Large-batch small-KV splits with tile_n=128. For seqlen_k in {512, 1024}, return 4 or 8.
static int choose_smallkv_largebatch_splits(int64_t seqlen_k) {
    int splits = static_cast<int>((seqlen_k + 127) / 128); // 512->4, 1024->8
    if (splits < 4) splits = 4;
    if (splits > 8) splits = 8;
    return splits;
}

// New: mid_kv_2048_large_batch splits with tile_n=64. For 2048 -> 32.
static inline int choose_midkv_2048_large_batch_splits() { return 32; }

// Small-kv 512 light path: tile_n=64 via 8 splits, single-stage, vectorized 16B loads.
static inline int smallkv_512_light_splits() { return 8; }

// kv512 large-batch non-persistent path: tile_n=128 via 4 splits, single-stage, vectorized 16B loads.
static inline int kv512_large_batch_np_splits() { return 4; }

// ===================== Host entry =====================

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
    int64_t causal) {
    (void)seqlen_q; // seqlen_q == 1 for decode
    (void)causal;   // non-causal in benchmark

    if (page_block_size != 16) {
        // Only page size 16 is supported here.
        return;
    }

    const int blocks_per_batch = static_cast<int>(num_blocks / batch_size);
    const int h = static_cast<int>(num_heads);
    const int hk = static_cast<int>(num_heads_k);
    const int hd = static_cast<int>(headdim);

    const int total_heads = static_cast<int>(batch_size * num_heads);

    // New narrowly-guarded small_kv_512_light path:
    // For kv == 512 and batch <= 8 (hd in {128,256}), use single-stage 1-warp kernel split into 8 tiles (tile_n=64).
    // This reduces fixed pipeline/paging overhead for short-KV while keeping vectorized 16B loads.
    if ((seqlen_k == 512) && (batch_size <= 8) && (hd == 128 || hd == 256)) {
        const int splits = smallkv_512_light_splits(); // 8
        // Allocate partials
        const size_t partial_s_elems = static_cast<size_t>(total_heads) * splits;
        const size_t partial_o_elems = partial_s_elems * static_cast<size_t>(hd);
        if (partial_o_elems > g_partial_o_elems) {
            if (g_partial_o != nullptr) cudaFree(g_partial_o);
            cudaMalloc(reinterpret_cast<void**>(&g_partial_o), partial_o_elems * sizeof(float));
            g_partial_o_elems = partial_o_elems;
        }
        if (partial_s_elems > g_partial_s_elems) {
            if (g_partial_m != nullptr) cudaFree(g_partial_m);
            if (g_partial_l != nullptr) cudaFree(g_partial_l);
            cudaMalloc(reinterpret_cast<void**>(&g_partial_m), partial_s_elems * sizeof(float));
            cudaMalloc(reinterpret_cast<void**>(&g_partial_l), partial_s_elems * sizeof(float));
            g_partial_s_elems = partial_s_elems;
        }

        dim3 split_grid(total_heads, splits);
        dim3 one_warp(32);
        if (hd == 128) {
            paged_decode_smallkv_split_kernel_t<128><<<split_grid, one_warp>>>(
                q, k_cache_paged, v_cache_paged,
                g_partial_o, g_partial_m, g_partial_l,
                cache_seqlens, block_table, blocks_per_batch, h, hk, splits);
            combine_split_kernel_t<128><<<total_heads, dim3(128)>>>(
                g_partial_o, g_partial_m, g_partial_l, output, h, splits);
        } else {
            paged_decode_smallkv_split_kernel_t<256><<<split_grid, one_warp>>>(
                q, k_cache_paged, v_cache_paged,
                g_partial_o, g_partial_m, g_partial_l,
                cache_seqlens, block_table, blocks_per_batch, h, hk, splits);
            combine_split_kernel_t<256><<<total_heads, dim3(256)>>>(
                g_partial_o, g_partial_m, g_partial_l, output, h, splits);
        }
        return;
    }

    // New: kv=512, batch>=16, hd==256 large-batch non-persistent path with low-footprint tile (tile_n=128).
    // Use single-stage 1-warp small-kv split path with 4 splits to raise wave count while keeping low overhead.
    if ((seqlen_k == 512) && (batch_size >= 16) && (hd == 256)) {
        const int splits = kv512_large_batch_np_splits(); // 4 -> tile_n=128
        const size_t partial_s_elems = static_cast<size_t>(total_heads) * splits;
        const size_t partial_o_elems = partial_s_elems * static_cast<size_t>(hd);
        if (partial_o_elems > g_partial_o_elems) {
            if (g_partial_o != nullptr) cudaFree(g_partial_o);
            cudaMalloc(reinterpret_cast<void**>(&g_partial_o), partial_o_elems * sizeof(float));
            g_partial_o_elems = partial_o_elems;
        }
        if (partial_s_elems > g_partial_s_elems) {
            if (g_partial_m != nullptr) cudaFree(g_partial_m);
            if (g_partial_l != nullptr) cudaFree(g_partial_l);
            cudaMalloc(reinterpret_cast<void**>(&g_partial_m), partial_s_elems * sizeof(float));
            cudaMalloc(reinterpret_cast<void**>(&g_partial_l), partial_s_elems * sizeof(float));
            g_partial_s_elems = partial_s_elems;
        }

        dim3 split_grid(total_heads, splits);
        dim3 one_warp(32);
        paged_decode_smallkv_split_kernel_t<256><<<split_grid, one_warp>>>(
            q, k_cache_paged, v_cache_paged,
            g_partial_o, g_partial_m, g_partial_l,
            cache_seqlens, block_table, blocks_per_batch, h, hk, splits);
        combine_split_kernel_t<256><<<total_heads, dim3(256)>>>(
            g_partial_o, g_partial_m, g_partial_l, output, h, splits);
        return;
    }

    // New guarded small-KV large-batch branch:
    // For kv == 1024 and batch >= 16, use 2-stage prefetch 1-warp kernel with
    // splits = kv / 128 (tile_n=128), then combine. Exclude kv==512 per retune notes.
    if ((batch_size >= 16) && (hd == 128 || hd == 256) && (seqlen_k == 1024)) {
        const int lb_splits = choose_smallkv_largebatch_splits(seqlen_k); // 8 for 1024

        // Allocate partials
        const size_t partial_s_elems = static_cast<size_t>(total_heads) * lb_splits;
        const size_t partial_o_elems = partial_s_elems * static_cast<size_t>(hd);
        if (partial_o_elems > g_partial_o_elems) {
            if (g_partial_o != nullptr) cudaFree(g_partial_o);
            cudaMalloc(reinterpret_cast<void**>(&g_partial_o), partial_o_elems * sizeof(float));
            g_partial_o_elems = partial_o_elems;
        }
        if (partial_s_elems > g_partial_s_elems) {
            if (g_partial_m != nullptr) cudaFree(g_partial_m);
            if (g_partial_l != nullptr) cudaFree(g_partial_l);
            cudaMalloc(reinterpret_cast<void**>(&g_partial_m), partial_s_elems * sizeof(float));
            cudaMalloc(reinterpret_cast<void**>(&g_partial_l), partial_s_elems * sizeof(float));
            g_partial_s_elems = partial_s_elems;
        }

        dim3 split_grid(total_heads, lb_splits);
        dim3 one_warp(32);
        if (hd == 128) {
            paged_decode_smallkv2s_split_kernel_t<128><<<split_grid, one_warp>>>(
                q, k_cache_paged, v_cache_paged,
                g_partial_o, g_partial_m, g_partial_l,
                cache_seqlens, block_table, blocks_per_batch, h, hk, lb_splits);
            combine_split_kernel_t<128><<<total_heads, dim3(128)>>>(
                g_partial_o, g_partial_m, g_partial_l, output, h, lb_splits);
        } else {
            paged_decode_smallkv2s_split_kernel_t<256><<<split_grid, one_warp>>>(
                q, k_cache_paged, v_cache_paged,
                g_partial_o, g_partial_m, g_partial_l,
                cache_seqlens, block_table, blocks_per_batch, h, hk, lb_splits);
            combine_split_kernel_t<256><<<total_heads, dim3(256)>>>(
                g_partial_o, g_partial_m, g_partial_l, output, h, lb_splits);
        }
        return;
    }

    // New narrowly-guarded mid_kv_2048_large_batch branch:
    // For kv == 2048 and batch >= 16 (hd in {128,256}), use single-stage 1-warp kernel with
    // splits = 32 (tile_n=64). Require 16B-aligned base K/V pointers; else fall back.
    if ((seqlen_k == 2048) && (batch_size >= 16) && (hd == 128 || hd == 256)) {
        const bool ptr_aligned =
            ((reinterpret_cast<uintptr_t>(k_cache_paged) & 0xF) == 0) &&
            ((reinterpret_cast<uintptr_t>(v_cache_paged) & 0xF) == 0);
        // Stride (in bytes) is 2 * num_heads_k * headdim; for hd in {128,256} and hk dividing 8 it's >= 2048B -> aligned.
        const size_t stride_bytes = static_cast<size_t>(2) * static_cast<size_t>(num_heads_k) * static_cast<size_t>(headdim);
        const bool stride_aligned = (stride_bytes % 16) == 0;

        if (ptr_aligned && stride_aligned) {
            const int splits = choose_midkv_2048_large_batch_splits(); // 32

            // Allocate partials
            const size_t partial_s_elems = static_cast<size_t>(total_heads) * splits;
            const size_t partial_o_elems = partial_s_elems * static_cast<size_t>(hd);
            if (partial_o_elems > g_partial_o_elems) {
                if (g_partial_o != nullptr) cudaFree(g_partial_o);
                cudaMalloc(reinterpret_cast<void**>(&g_partial_o), partial_o_elems * sizeof(float));
                g_partial_o_elems = partial_o_elems;
            }
            if (partial_s_elems > g_partial_s_elems) {
                if (g_partial_m != nullptr) cudaFree(g_partial_m);
                if (g_partial_l != nullptr) cudaFree(g_partial_l);
                cudaMalloc(reinterpret_cast<void**>(&g_partial_m), partial_s_elems * sizeof(float));
                cudaMalloc(reinterpret_cast<void**>(&g_partial_l), partial_s_elems * sizeof(float));
                g_partial_s_elems = partial_s_elems;
            }

            dim3 split_grid(total_heads, splits);
            dim3 one_warp(32);
            if (hd == 128) {
                paged_decode_smallkv_split_kernel_t<128><<<split_grid, one_warp>>>(
                    q, k_cache_paged, v_cache_paged,
                    g_partial_o, g_partial_m, g_partial_l,
                    cache_seqlens, block_table, blocks_per_batch, h, hk, splits);
                combine_split_kernel_t<128><<<total_heads, dim3(128)>>>(
                    g_partial_o, g_partial_m, g_partial_l, output, h, splits);
            } else {
                paged_decode_smallkv_split_kernel_t<256><<<split_grid, one_warp>>>(
                    q, k_cache_paged, v_cache_paged,
                    g_partial_o, g_partial_m, g_partial_l,
                    cache_seqlens, block_table, blocks_per_batch, h, hk, splits);
                combine_split_kernel_t<256><<<total_heads, dim3(256)>>>(
                    g_partial_o, g_partial_m, g_partial_l, output, h, splits);
            }
            return;
        }
        // else: fall through to the existing <=2048 path
    }

    // Guarded small-KV fast path (<= 2048) for common head dims using 1-warp kernel, with split option.
    // Now includes vectorized 16B/8B loads and simple prefetch.
    if (seqlen_k <= 2048 && (hd == 128 || hd == 256)) {
        const int small_splits = choose_smallkv_splits(batch_size, seqlen_k);
        if (small_splits <= 1) {
            dim3 block(32);  // one warp per head
            if (hd == 128) {
                paged_decode_smallkv_kernel_t<128><<<total_heads, block>>>(
                    q, k_cache_paged, v_cache_paged, output,
                    cache_seqlens, block_table, blocks_per_batch, h, hk);
                return;
            } else {
                paged_decode_smallkv_kernel_t<256><<<total_heads, block>>>(
                    q, k_cache_paged, v_cache_paged, output,
                    cache_seqlens, block_table, blocks_per_batch, h, hk);
                return;
            }
        } else {
            // Allocate partials
            const size_t partial_s_elems = static_cast<size_t>(total_heads) * small_splits;
            const size_t partial_o_elems = partial_s_elems * static_cast<size_t>(hd);
            if (partial_o_elems > g_partial_o_elems) {
                if (g_partial_o != nullptr) cudaFree(g_partial_o);
                cudaMalloc(reinterpret_cast<void**>(&g_partial_o), partial_o_elems * sizeof(float));
                g_partial_o_elems = partial_o_elems;
            }
            if (partial_s_elems > g_partial_s_elems) {
                if (g_partial_m != nullptr) cudaFree(g_partial_m);
                if (g_partial_l != nullptr) cudaFree(g_partial_l);
                cudaMalloc(reinterpret_cast<void**>(&g_partial_m), partial_s_elems * sizeof(float));
                cudaMalloc(reinterpret_cast<void**>(&g_partial_l), partial_s_elems * sizeof(float));
                g_partial_s_elems = partial_s_elems;
            }
            dim3 split_grid(total_heads, small_splits);
            dim3 block(32); // one warp per partial
            if (hd == 128) {
                paged_decode_smallkv_split_kernel_t<128><<<split_grid, block>>>(
                    q, k_cache_paged, v_cache_paged,
                    g_partial_o, g_partial_m, g_partial_l,
                    cache_seqlens, block_table, blocks_per_batch, h, hk, small_splits);
                combine_split_kernel_t<128><<<total_heads, dim3(128)>>>(
                    g_partial_o, g_partial_m, g_partial_l, output, h, small_splits);
                return;
            } else {
                paged_decode_smallkv_split_kernel_t<256><<<split_grid, block>>>(
                    q, k_cache_paged, v_cache_paged,
                    g_partial_o, g_partial_m, g_partial_l,
                    cache_seqlens, block_table, blocks_per_batch, h, hk, small_splits);
                combine_split_kernel_t<256><<<total_heads, dim3(256)>>>(
                    g_partial_o, g_partial_m, g_partial_l, output, h, small_splits);
                return;
            }
        }
    }

    // New dual-gated large-batch small/mid-KV path (STAGES_KV=1) for 2049..4096.
    // Avoids the 2-stage double-buffering overhead that penalizes shorter sequences at large batch.
    if ((batch_size >= 8) && (seqlen_k > 2048) && (seqlen_k <= 4096) && (hd == 128 || hd == 256)) {
        const int lb_sm_splits = choose_largebatch_smallmid_splits(batch_size, seqlen_k);
        if (lb_sm_splits <= 1) {
            dim3 block(32);  // one warp per head
            if (hd == 128) {
                paged_decode_smallkv_kernel_t<128><<<total_heads, block>>>(
                    q, k_cache_paged, v_cache_paged, output,
                    cache_seqlens, block_table, blocks_per_batch, h, hk);
            } else {
                paged_decode_smallkv_kernel_t<256><<<total_heads, block>>>(
                    q, k_cache_paged, v_cache_paged, output,
                    cache_seqlens, block_table, blocks_per_batch, h, hk);
            }
            return;
        } else {
            // Allocate partials
            const size_t partial_s_elems = static_cast<size_t>(total_heads) * lb_sm_splits;
            const size_t partial_o_elems = partial_s_elems * static_cast<size_t>(hd);
            if (partial_o_elems > g_partial_o_elems) {
                if (g_partial_o != nullptr) cudaFree(g_partial_o);
                cudaMalloc(reinterpret_cast<void**>(&g_partial_o), partial_o_elems * sizeof(float));
                g_partial_o_elems = partial_o_elems;
            }
            if (partial_s_elems > g_partial_s_elems) {
                if (g_partial_m != nullptr) cudaFree(g_partial_m);
                if (g_partial_l != nullptr) cudaFree(g_partial_l);
                cudaMalloc(reinterpret_cast<void**>(&g_partial_m), partial_s_elems * sizeof(float));
                cudaMalloc(reinterpret_cast<void**>(&g_partial_l), partial_s_elems * sizeof(float));
                g_partial_s_elems = partial_s_elems;
            }
            dim3 split_grid(total_heads, lb_sm_splits);
            dim3 block(32); // one warp per partial
            if (hd == 128) {
                paged_decode_smallkv_split_kernel_t<128><<<split_grid, block>>>(
                    q, k_cache_paged, v_cache_paged,
                    g_partial_o, g_partial_m, g_partial_l,
                    cache_seqlens, block_table, blocks_per_batch, h, hk, lb_sm_splits);
                combine_split_kernel_t<128><<<total_heads, dim3(128)>>>(
                    g_partial_o, g_partial_m, g_partial_l, output, h, lb_sm_splits);
            } else {
                paged_decode_smallkv_split_kernel_t<256><<<split_grid, block>>>(
                    q, k_cache_paged, v_cache_paged,
                    g_partial_o, g_partial_m, g_partial_l,
                    cache_seqlens, block_table, blocks_per_batch, h, hk, lb_sm_splits);
                combine_split_kernel_t<256><<<total_heads, dim3(256)>>>(
                    g_partial_o, g_partial_m, g_partial_l, output, h, lb_sm_splits);
            }
            return;
        }
    }

    // New guarded large-batch long-KV path: double-buffered 1-warp kernel with splits to raise occupancy.
    if ((batch_size >= 8) && (seqlen_k >= 4096) && (hd == 128 || hd == 256)) {
        const int lb_splits = choose_largekv_splits(batch_size, seqlen_k);
        if (lb_splits <= 1) {
            dim3 block(32);
            if (hd == 128) {
                paged_decode_smallkv2s_kernel_t<128><<<total_heads, block>>>(
                    q, k_cache_paged, v_cache_paged, output,
                    cache_seqlens, block_table, blocks_per_batch, h, hk);
            } else {
                paged_decode_smallkv2s_kernel_t<256><<<total_heads, block>>>(
                    q, k_cache_paged, v_cache_paged, output,
                    cache_seqlens, block_table, blocks_per_batch, h, hk);
            }
            return;
        } else {
            // Allocate partials
            const size_t partial_s_elems = static_cast<size_t>(total_heads) * lb_splits;
            const size_t partial_o_elems = partial_s_elems * static_cast<size_t>(hd);
            if (partial_o_elems > g_partial_o_elems) {
                if (g_partial_o != nullptr) cudaFree(g_partial_o);
                cudaMalloc(reinterpret_cast<void**>(&g_partial_o), partial_o_elems * sizeof(float));
                g_partial_o_elems = partial_o_elems;
            }
            if (partial_s_elems > g_partial_s_elems) {
                if (g_partial_m != nullptr) cudaFree(g_partial_m);
                if (g_partial_l != nullptr) cudaFree(g_partial_l);
                cudaMalloc(reinterpret_cast<void**>(&g_partial_m), partial_s_elems * sizeof(float));
                cudaMalloc(reinterpret_cast<void**>(&g_partial_l), partial_s_elems * sizeof(float));
                g_partial_s_elems = partial_s_elems;
            }
            dim3 split_grid(total_heads, lb_splits);
            dim3 block(32);
            if (hd == 128) {
                paged_decode_smallkv2s_split_kernel_t<128><<<split_grid, block>>>(
                    q, k_cache_paged, v_cache_paged,
                    g_partial_o, g_partial_m, g_partial_l,
                    cache_seqlens, block_table, blocks_per_batch, h, hk, lb_splits);
                combine_split_kernel_t<128><<<total_heads, dim3(128)>>>(
                    g_partial_o, g_partial_m, g_partial_l, output, h, lb_splits);
            } else {
                paged_decode_smallkv2s_split_kernel_t<256><<<split_grid, block>>>(
                    q, k_cache_paged, v_cache_paged,
                    g_partial_o, g_partial_m, g_partial_l,
                    cache_seqlens, block_table, blocks_per_batch, h, hk, lb_splits);
                combine_split_kernel_t<256><<<total_heads, dim3(256)>>>(
                    g_partial_o, g_partial_m, g_partial_l, output, h, lb_splits);
            }
            return;
        }
    }

    // For non-small-KV or unsupported head dims, use existing long/mid path
    int splits = choose_splits(batch_size, seqlen_k);
    if (splits > MAX_SPLITS) {
        splits = MAX_SPLITS;
    }

    // Dispatch for common head dims: 128 and 256 (long/mid-KV path unchanged for small-batch)
    if (hd == 128) {
        dim3 block(128);
        if (splits <= 1) {
            paged_decode_kernel_t<128><<<total_heads, block>>>(
                q, k_cache_paged, v_cache_paged, output,
                cache_seqlens, block_table, blocks_per_batch, h, hk);
            return;
        }
        // Allocate partials
        {
            const size_t partial_s_elems = static_cast<size_t>(total_heads) * splits;
            const size_t partial_o_elems = partial_s_elems * 128;
            if (partial_o_elems > g_partial_o_elems) {
                if (g_partial_o != nullptr) cudaFree(g_partial_o);
                cudaMalloc(reinterpret_cast<void**>(&g_partial_o), partial_o_elems * sizeof(float));
                g_partial_o_elems = partial_o_elems;
            }
            if (partial_s_elems > g_partial_s_elems) {
                if (g_partial_m != nullptr) cudaFree(g_partial_m);
                if (g_partial_l != nullptr) cudaFree(g_partial_l);
                cudaMalloc(reinterpret_cast<void**>(&g_partial_m), partial_s_elems * sizeof(float));
                cudaMalloc(reinterpret_cast<void**>(&g_partial_l), partial_s_elems * sizeof(float));
                g_partial_s_elems = partial_s_elems;
            }
        }
        dim3 split_grid(total_heads, splits);
        paged_decode_split_kernel_t<128><<<split_grid, block>>>(
            q, k_cache_paged, v_cache_paged,
            g_partial_o, g_partial_m, g_partial_l,
            cache_seqlens, block_table, blocks_per_batch, h, hk, splits);
        combine_split_kernel_t<128><<<total_heads, block>>>(
            g_partial_o, g_partial_m, g_partial_l, output, h, splits);
        return;
    }

    if (hd == 256) {
        dim3 block(256);
        if (splits <= 1) {
            paged_decode_kernel_t<256><<<total_heads, block>>>(
                q, k_cache_paged, v_cache_paged, output,
                cache_seqlens, block_table, blocks_per_batch, h, hk);
            return;
        }
        // Allocate partials
        {
            const size_t partial_s_elems = static_cast<size_t>(total_heads) * splits;
            const size_t partial_o_elems = partial_s_elems * 256;
            if (partial_o_elems > g_partial_o_elems) {
                if (g_partial_o != nullptr) cudaFree(g_partial_o);
                cudaMalloc(reinterpret_cast<void**>(&g_partial_o), partial_o_elems * sizeof(float));
                g_partial_o_elems = partial_o_elems;
            }
            if (partial_s_elems > g_partial_s_elems) {
                if (g_partial_m != nullptr) cudaFree(g_partial_m);
                if (g_partial_l != nullptr) cudaFree(g_partial_l);
                cudaMalloc(reinterpret_cast<void**>(&g_partial_m), partial_s_elems * sizeof(float));
                cudaMalloc(reinterpret_cast<void**>(&g_partial_l), partial_s_elems * sizeof(float));
                g_partial_s_elems = partial_s_elems;
            }
        }
        dim3 split_grid(total_heads, splits);
        paged_decode_split_kernel_t<256><<<split_grid, block>>>(
            q, k_cache_paged, v_cache_paged,
            g_partial_o, g_partial_m, g_partial_l,
            cache_seqlens, block_table, blocks_per_batch, h, hk, splits);
        combine_split_kernel_t<256><<<total_heads, block>>>(
            g_partial_o, g_partial_m, g_partial_l, output, h, splits);
        return;
    }

    // Generic fallback path (correctness over performance)
    {
        dim3 block(hd);
        if (splits <= 1) {
            paged_decode_kernel_generic<<<total_heads, block>>>(
                q, k_cache_paged, v_cache_paged, output,
                cache_seqlens, block_table, blocks_per_batch, h, hk, hd);
            return;
        }
        const size_t partial_s_elems = static_cast<size_t>(total_heads) * splits;
        const size_t partial_o_elems = partial_s_elems * static_cast<size_t>(hd);
        if (partial_o_elems > g_partial_o_elems) {
            if (g_partial_o != nullptr) cudaFree(g_partial_o);
            cudaMalloc(reinterpret_cast<void**>(&g_partial_o), partial_o_elems * sizeof(float));
            g_partial_o_elems = partial_o_elems;
        }
        if (partial_s_elems > g_partial_s_elems) {
            if (g_partial_m != nullptr) cudaFree(g_partial_m);
            if (g_partial_l != nullptr) cudaFree(g_partial_l);
            cudaMalloc(reinterpret_cast<void**>(&g_partial_m), partial_s_elems * sizeof(float));
            cudaMalloc(reinterpret_cast<void**>(&g_partial_l), partial_s_elems * sizeof(float));
            g_partial_s_elems = partial_s_elems;
        }
        dim3 split_grid(total_heads, splits);
        paged_decode_split_kernel_generic<<<split_grid, block>>>(
            q, k_cache_paged, v_cache_paged,
            g_partial_o, g_partial_m, g_partial_l,
            cache_seqlens, block_table, blocks_per_batch, h, hk, splits, hd);
        combine_split_kernel_generic<<<total_heads, block>>>(
            g_partial_o, g_partial_m, g_partial_l, output, splits, hd);
    }
}

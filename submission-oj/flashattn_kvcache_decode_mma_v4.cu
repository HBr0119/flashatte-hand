#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mcflashinfer/cp_async.cuh>
#include <mcflashinfer/utils.cuh>
#include <stddef.h>
#include <stdint.h>

/*
 * cu-bridge does not expose these fragment types because __MACA_ARCH__
 * is not set. They are required by the native C500 MMA builtins.
 */
typedef _Float16 maca_half4 __attribute__((__vector_size__(8)));
typedef float maca_float4 __attribute__((__vector_size__(16)));

#define HEAD_DIM 128
#define PAGE_SIZE 16
#define MMA_TILE 16
#define MMA_K_TILES (HEAD_DIM / MMA_TILE)
#define MMA_THREADS 64
#define WARP_SIZE_ 32
#define MAX_SPLITS 64

union alignas(8) mma_bf16_fragment_t {
    maca_half4 vector;
    __nv_bfloat16 element[4];
};

union alignas(16) mma_float_fragment_t {
    maca_float4 vector;
    float element[4];
};

#define MID_CHUNK_PAGES 32
#define MID_TILE_PAGES 2
#define MID_TILE_TOKENS (PAGE_SIZE * MID_TILE_PAGES)

#define NATIVE_PIPELINE_STAGES 2
#define NATIVE_MIN_CHUNK_PAGES 16
#define NATIVE_LONG_MIN_CHUNK_PAGES 32

static float* g_partial_o = nullptr;
static float* g_partial_lse = nullptr;
static size_t g_partial_o_elems = 0;
static size_t g_partial_lse_elems = 0;

struct alignas(16) float8_t {
    float value[8];
};

union alignas(16) packed_bf16x8_t {
    int4 bits;
    uint32_t words[4];
};

union packed_bf16x2_t {
    uint32_t bits;
    __nv_bfloat162 value;
};

__device__ __forceinline__ float subgroup_sum_16(float value) {
    constexpr unsigned kMask = 0xffffffffu;
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(kMask, value, offset, 16);
    }
    return value;
}

__device__ __forceinline__ float subgroup_max_16(float value) {
    constexpr unsigned kMask = 0xffffffffu;
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        value = fmaxf(
            value,
            __shfl_xor_sync(kMask, value, offset, 16));
    }
    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
    constexpr unsigned kMask = 0xffffffffu;
#pragma unroll
    for (int offset = WARP_SIZE_ / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(kMask, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum_128_shared(float value) {
    __shared__ float reduce_buffer[HEAD_DIM];

    reduce_buffer[threadIdx.x] = value;
    __syncthreads();

#pragma unroll
    for (int stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce_buffer[threadIdx.x] += reduce_buffer[threadIdx.x + stride];
        }
        __syncthreads();
    }

    return reduce_buffer[0];
}

__device__ __forceinline__ float block_sum_128_warp(float value) {
    __shared__ float warp_sums[4];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    value = warp_sum(value);

    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        float sum = lane < 4 ? warp_sums[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) {
            warp_sums[0] = sum;
        }
    }
    __syncthreads();

    return warp_sums[0];
}

__device__ __forceinline__ float2 load_bf16x2_as_float2(
    const __nv_bfloat16* ptr) {
    return __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162*>(ptr));
}

__device__ __forceinline__ float8_t load_bf16x8_as_float8(
    const __nv_bfloat16* ptr) {
    packed_bf16x8_t packed;
    packed.bits = *reinterpret_cast<const int4*>(ptr);

    packed_bf16x2_t p0;
    packed_bf16x2_t p1;
    packed_bf16x2_t p2;
    packed_bf16x2_t p3;

    p0.bits = packed.words[0];
    p1.bits = packed.words[1];
    p2.bits = packed.words[2];
    p3.bits = packed.words[3];

    const float2 f0 = __bfloat1622float2(p0.value);
    const float2 f1 = __bfloat1622float2(p1.value);
    const float2 f2 = __bfloat1622float2(p2.value);
    const float2 f3 = __bfloat1622float2(p3.value);

    float8_t result;
    result.value[0] = f0.x;
    result.value[1] = f0.y;
    result.value[2] = f1.x;
    result.value[3] = f1.y;
    result.value[4] = f2.x;
    result.value[5] = f2.y;
    result.value[6] = f3.x;
    result.value[7] = f3.y;
    return result;
}

__device__ __forceinline__ float dot_float8(
    const float8_t& lhs,
    const float8_t& rhs) {
    float result = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        result = fmaf(lhs.value[i], rhs.value[i], result);
    }
    return result;
}

__device__ __forceinline__ void zero_float8(float8_t& value) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        value.value[i] = 0.0f;
    }
}

__device__ __forceinline__ void scale_float8(
    float8_t& value,
    float scale) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        value.value[i] = fmaf(scale, value.value[i], 0.0f);
    }
}

__device__ __forceinline__ void fma_float8(
    float8_t& accumulator,
    const float8_t& value,
    float weight) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        accumulator.value[i] =
            fmaf(weight, value.value[i], accumulator.value[i]);
    }
}

__device__ __forceinline__ void store_float8(
    float* ptr,
    const float8_t& value) {
    *reinterpret_cast<float4*>(ptr) =
        make_float4(
            value.value[0],
            value.value[1],
            value.value[2],
            value.value[3]);

    *reinterpret_cast<float4*>(ptr + 4) =
        make_float4(
            value.value[4],
            value.value[5],
            value.value[6],
            value.value[7]);
}

__device__ __forceinline__ float8_t load_float8(const float* ptr) {
    const float4 lo = *reinterpret_cast<const float4*>(ptr);
    const float4 hi = *reinterpret_cast<const float4*>(ptr + 4);

    float8_t result;
    result.value[0] = lo.x;
    result.value[1] = lo.y;
    result.value[2] = lo.z;
    result.value[3] = lo.w;
    result.value[4] = hi.x;
    result.value[5] = hi.y;
    result.value[6] = hi.z;
    result.value[7] = hi.w;
    return result;
}

__device__ __forceinline__ void async_wait_all() {
    flashinfer::cp_async_bsm_wait<0>();
}

__global__ __launch_bounds__(HEAD_DIM)
void paged_decode_kernel(
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
    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh - b * num_heads;
    const int gqa_ratio = num_heads / num_heads_k;
    const int kv_head = h / gqa_ratio;
    const int seqlen = cache_seqlens[b];

    constexpr float kScale = 0.08838834764831845f;

    const int64_t q_base =
        (static_cast<int64_t>(b) * num_heads + h) * HEAD_DIM;
    const float q_value = __bfloat162float(q[q_base + tid]);

    float running_max = -CUDART_INF_F;
    float running_sum = 0.0f;
    float accumulator = 0.0f;

    const int32_t* block_row =
        block_table + static_cast<int64_t>(b) * blocks_per_batch;
    const int64_t token_stride =
        static_cast<int64_t>(num_heads_k) * HEAD_DIM;
    const int64_t page_stride =
        static_cast<int64_t>(PAGE_SIZE) * token_stride;
    const int64_t head_offset =
        static_cast<int64_t>(kv_head) * HEAD_DIM;

    const int valid_pages =
        (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;

    for (int page = 0; page < valid_pages; ++page) {
        const int physical_block = block_row[page];
        const int remaining = seqlen - page * PAGE_SIZE;
        const int page_tokens =
            remaining < PAGE_SIZE ? remaining : PAGE_SIZE;

        const int64_t page_base =
            static_cast<int64_t>(physical_block) * page_stride +
            head_offset;

        const __nv_bfloat16* k_ptr =
            k_cache_paged + page_base + tid;
        const __nv_bfloat16* v_ptr =
            v_cache_paged + page_base + tid;

        for (int token = 0; token < page_tokens; ++token) {
            const float k_value = __bfloat162float(*k_ptr);
            const float dot = q_value * k_value;
            const float score =
                (seqlen <= 32
                     ? block_sum_128_shared(dot)
                     : block_sum_128_warp(dot)) *
                kScale;
            const float v_value = __bfloat162float(*v_ptr);

            const float new_max = fmaxf(running_max, score);
            const float alpha = expf(running_max - new_max);
            const float probability = expf(score - new_max);

            accumulator =
                accumulator * alpha + probability * v_value;
            running_sum =
                running_sum * alpha + probability;
            running_max = new_max;

            k_ptr += token_stride;
            v_ptr += token_stride;
        }
    }

    const int64_t output_base =
        (static_cast<int64_t>(b) * num_heads + h) * HEAD_DIM;

    output[output_base + tid] =
        __float2bfloat16(
            running_sum > 0.0f
                ? accumulator / running_sum
                : 0.0f);
}

/*
 * Pre-load Q values into MMA A fragments (col-major: row + k * 16).
 * Called once per block before the page loop. All 64 threads participate.
 */
template <int GROUP>
__device__ __forceinline__ void initialize_q_fragments(
    const __nv_bfloat16* __restrict__ q,
    mma_bf16_fragment_t (&q_fragment)[MMA_K_TILES],
    int b,
    int kv_head) {
    constexpr int kNumHeads = 32;
    const int tid = threadIdx.x;
    const int fragment_linear_base = tid * 4;
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);

#pragma unroll
    for (int kt = 0; kt < MMA_K_TILES; ++kt) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int linear = fragment_linear_base + i;
            const int query_in_group = linear & 15;
            const int k_in_tile = linear >> 4;

            if (query_in_group < GROUP) {
                const int query_head =
                    kv_head * GROUP + query_in_group;
                const int64_t q_offset =
                    (static_cast<int64_t>(b) * kNumHeads +
                     query_head) *
                        HEAD_DIM +
                    kt * MMA_TILE +
                    k_in_tile;
                q_fragment[kt].element[i] = q[q_offset];
            } else {
                q_fragment[kt].element[i] = zero;
            }
        }
    }
}

__device__ __forceinline__ void zero_mma_accumulator(
    mma_float_fragment_t& fragment) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        fragment.element[i] = 0.0f;
    }
}

template <int NUM_KV_HEADS>
__global__ __launch_bounds__((32 / NUM_KV_HEADS / 2) * WARP_SIZE_)
void paged_decode_gqa_page_softmax_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch) {
    static_assert(
        NUM_KV_HEADS == 4 || NUM_KV_HEADS == 8,
        "Unsupported KV-head count");

    constexpr int kNumHeads = 32;
    constexpr int kGroup = kNumHeads / NUM_KV_HEADS;
    constexpr int kWarps = kGroup / 2;
    constexpr int kThreads = kWarps * WARP_SIZE_;
    constexpr int kPacksPerToken = HEAD_DIM / 8;
    constexpr int64_t kTokenStride =
        static_cast<int64_t>(NUM_KV_HEADS) * HEAD_DIM;
    constexpr int64_t kPageStride =
        static_cast<int64_t>(PAGE_SIZE) * kTokenStride;
    constexpr float kScale = 0.08838834764831845f;
    constexpr unsigned kMask = 0xffffffffu;

    __shared__ __align__(16)
        __nv_bfloat16 shared_k[PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(16)
        __nv_bfloat16 shared_v[PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(64)
        float shared_score_or_probability[kGroup][PAGE_SIZE];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int half = lane >> 4;
    const int sublane = lane & 15;
    const int query_in_group = warp * 2 + half;

    const int bkv = blockIdx.x;
    const int b = bkv / NUM_KV_HEADS;
    const int kv_head = bkv - b * NUM_KV_HEADS;
    const int query_head =
        kv_head * kGroup + query_in_group;

    const int seqlen = cache_seqlens[b];
    const int valid_pages =
        (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;

    const int d0 = sublane * 2;
    const int d1 = d0 + 32;
    const int d2 = d0 + 64;
    const int d3 = d0 + 96;

    const int64_t q_base =
        (static_cast<int64_t>(b) * kNumHeads + query_head) *
        HEAD_DIM;

    const float2 q_pair0 =
        load_bf16x2_as_float2(q + q_base + d0);
    const float2 q_pair1 =
        load_bf16x2_as_float2(q + q_base + d1);
    const float2 q_pair2 =
        load_bf16x2_as_float2(q + q_base + d2);
    const float2 q_pair3 =
        load_bf16x2_as_float2(q + q_base + d3);

    float acc0x = 0.0f;
    float acc0y = 0.0f;
    float acc1x = 0.0f;
    float acc1y = 0.0f;
    float acc2x = 0.0f;
    float acc2y = 0.0f;
    float acc3x = 0.0f;
    float acc3y = 0.0f;

    float running_max = -CUDART_INF_F;
    float running_sum = 0.0f;

    const int32_t* block_row =
        block_table + static_cast<int64_t>(b) * blocks_per_batch;
    const int64_t head_offset =
        static_cast<int64_t>(kv_head) * HEAD_DIM;

    float* score_row =
        shared_score_or_probability[query_in_group];

    for (int page = 0; page < valid_pages; ++page) {
        const int physical_block = block_row[page];
        const int remaining = seqlen - page * PAGE_SIZE;
        const int page_tokens =
            remaining < PAGE_SIZE ? remaining : PAGE_SIZE;

        const int64_t page_base =
            static_cast<int64_t>(physical_block) * kPageStride +
            head_offset;
        const int pack_count =
            page_tokens * kPacksPerToken;

        for (int pack = tid;
             pack < pack_count;
             pack += kThreads) {
            const int token = pack / kPacksPerToken;
            const int pack_in_token =
                pack - token * kPacksPerToken;
            const int dim = pack_in_token * 8;

            const int64_t global_offset =
                page_base +
                static_cast<int64_t>(token) * kTokenStride +
                dim;
            const int shared_offset =
                token * HEAD_DIM + dim;

            flashinfer::cp_async::load_128b_bsm(
                shared_k + shared_offset,
                k_cache_paged + global_offset);

            flashinfer::cp_async::load_128b_bsm(
                shared_v + shared_offset,
                v_cache_paged + global_offset);
        }

        async_wait_all();
        __syncthreads();

#pragma unroll 1
        for (int token = 0; token < page_tokens; ++token) {
            const int shared_base = token * HEAD_DIM;

            const float2 k_pair0 =
                load_bf16x2_as_float2(
                    shared_k + shared_base + d0);
            const float2 k_pair1 =
                load_bf16x2_as_float2(
                    shared_k + shared_base + d1);
            const float2 k_pair2 =
                load_bf16x2_as_float2(
                    shared_k + shared_base + d2);
            const float2 k_pair3 =
                load_bf16x2_as_float2(
                    shared_k + shared_base + d3);

            float dot =
                q_pair0.x * k_pair0.x +
                q_pair0.y * k_pair0.y +
                q_pair1.x * k_pair1.x +
                q_pair1.y * k_pair1.y +
                q_pair2.x * k_pair2.x +
                q_pair2.y * k_pair2.y +
                q_pair3.x * k_pair3.x +
                q_pair3.y * k_pair3.y;

            dot = subgroup_sum_16(dot);

            if (sublane == 0) {
                score_row[token] = dot * kScale;
            }
        }

        __syncwarp(kMask);

        const float lane_score =
            sublane < page_tokens
                ? score_row[sublane]
                : -CUDART_INF_F;

        const float page_max =
            subgroup_max_16(lane_score);

        float merged_max = 0.0f;
        float alpha = 0.0f;

        if (sublane == 0) {
            merged_max = fmaxf(running_max, page_max);
            alpha = expf(running_max - merged_max);
        }

        merged_max =
            __shfl_sync(kMask, merged_max, 0, 16);
        alpha =
            __shfl_sync(kMask, alpha, 0, 16);

        const float probability =
            sublane < page_tokens
                ? expf(lane_score - merged_max)
                : 0.0f;

        if (sublane < page_tokens) {
            score_row[sublane] = probability;
        }

        const float page_sum =
            subgroup_sum_16(probability);

        if (sublane == 0) {
            running_sum =
                running_sum * alpha + page_sum;
            running_max = merged_max;
        }

        acc0x = fmaf(alpha, acc0x, 0.0f);
        acc0y = fmaf(alpha, acc0y, 0.0f);
        acc1x = fmaf(alpha, acc1x, 0.0f);
        acc1y = fmaf(alpha, acc1y, 0.0f);
        acc2x = fmaf(alpha, acc2x, 0.0f);
        acc2y = fmaf(alpha, acc2y, 0.0f);
        acc3x = fmaf(alpha, acc3x, 0.0f);
        acc3y = fmaf(alpha, acc3y, 0.0f);

        __syncwarp(kMask);

#pragma unroll 1
        for (int token = 0; token < page_tokens; ++token) {
            const float probability_value = score_row[token];
            const int shared_base = token * HEAD_DIM;

            const float2 v_pair0 =
                load_bf16x2_as_float2(
                    shared_v + shared_base + d0);
            const float2 v_pair1 =
                load_bf16x2_as_float2(
                    shared_v + shared_base + d1);
            const float2 v_pair2 =
                load_bf16x2_as_float2(
                    shared_v + shared_base + d2);
            const float2 v_pair3 =
                load_bf16x2_as_float2(
                    shared_v + shared_base + d3);

            acc0x =
                fmaf(probability_value, v_pair0.x, acc0x);
            acc0y =
                fmaf(probability_value, v_pair0.y, acc0y);
            acc1x =
                fmaf(probability_value, v_pair1.x, acc1x);
            acc1y =
                fmaf(probability_value, v_pair1.y, acc1y);
            acc2x =
                fmaf(probability_value, v_pair2.x, acc2x);
            acc2y =
                fmaf(probability_value, v_pair2.y, acc2y);
            acc3x =
                fmaf(probability_value, v_pair3.x, acc3x);
            acc3y =
                fmaf(probability_value, v_pair3.y, acc3y);
        }

        if (page + 1 < valid_pages) {
            __syncthreads();
        }
    }

    const float denominator =
        __shfl_sync(kMask, running_sum, 0, 16);
    const float inverse =
        denominator > 0.0f ? 1.0f / denominator : 0.0f;

    const int64_t output_base =
        (static_cast<int64_t>(b) * kNumHeads + query_head) *
        HEAD_DIM;

    output[output_base + d0] =
        __float2bfloat16(acc0x * inverse);
    output[output_base + d0 + 1] =
        __float2bfloat16(acc0y * inverse);
    output[output_base + d1] =
        __float2bfloat16(acc1x * inverse);
    output[output_base + d1 + 1] =
        __float2bfloat16(acc1y * inverse);
    output[output_base + d2] =
        __float2bfloat16(acc2x * inverse);
    output[output_base + d2 + 1] =
        __float2bfloat16(acc2y * inverse);
    output[output_base + d3] =
        __float2bfloat16(acc3x * inverse);
    output[output_base + d3 + 1] =
        __float2bfloat16(acc3y * inverse);
}

/*
 * MMA-accelerated GQA page-softmax kernel (v4: 64-thread, WMMA lane mapping).
 *
 * Fragment lane mapping from MACA LLVM WMMA header:
 *   load_matrix_sync / store_matrix_sync for 16x16x16 bf16:
 *
 *   A (Q, matrix_a RowMajor): row = lane & 0xf, k_off = ((lane>>4)<<2)
 *   B (K, matrix_b RowMajor): k_off = ((lane>>4)<<2), token = lane & 0xf
 *   C (output, RowMajor):   row_base = ((lane>>4)<<2), col = lane & 0xf
 *
 * The MMA builtin requires all 64 threads in the block to call it.
 * Warp 1 (lanes 32-63) provides zero fragments and also calls MMA.
 * Warp 0 (lanes 0-31) handles all actual data.
 *
 * Per K-tile: 2 MMA calls (k 0-7, k 8-15), accumulated over MMA_K_TILES.
 *
 * Page iterator, CP-async, online softmax, and scalar P*V are preserved
 * from the baseline kernel verbatim.
 */
template <int NUM_KV_HEADS>
__global__ __launch_bounds__(MMA_THREADS)
void paged_decode_gqa_mma_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch) {
    static_assert(
        NUM_KV_HEADS == 4 || NUM_KV_HEADS == 8,
        "Unsupported KV-head count");

    constexpr int kNumHeads = 32;
    constexpr int kGroup = kNumHeads / NUM_KV_HEADS;
    constexpr int kThreadsPerHead = MMA_THREADS / kGroup;
    constexpr int kDimsPerThread = HEAD_DIM / kThreadsPerHead;
    constexpr int kPacksPerToken = HEAD_DIM / 8;
    constexpr int64_t kTokenStride =
        static_cast<int64_t>(NUM_KV_HEADS) * HEAD_DIM;
    constexpr int64_t kPageStride =
        static_cast<int64_t>(PAGE_SIZE) * kTokenStride;
    constexpr float kScale = 0.08838834764831845f;

    __shared__ __align__(16)
        __nv_bfloat16 shared_k[PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(16)
        __nv_bfloat16 shared_v[PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(64)
        float shared_scores[kGroup][PAGE_SIZE];
    __shared__ float shared_page_alpha[kGroup];
    __shared__ float shared_merged_max[kGroup];
    __shared__ float shared_new_sum[kGroup];
    __shared__ float shared_running_max[kGroup];
    __shared__ float shared_running_sum[kGroup];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int head_in_group = tid / kThreadsPerHead;
    const int thread_in_head = tid % kThreadsPerHead;
    const int dim_base = thread_in_head * kDimsPerThread;

    const int bkv = blockIdx.x;
    const int b = bkv / NUM_KV_HEADS;
    const int kv_head = bkv - b * NUM_KV_HEADS;

    const int seqlen = cache_seqlens[b];
    const int valid_pages =
        (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;

    const int32_t* block_row =
        block_table + static_cast<int64_t>(b) * blocks_per_batch;
    const int64_t head_offset =
        static_cast<int64_t>(kv_head) * HEAD_DIM;

    // -------- WMMA lane mapping (64-thread MMA verified) ----------
    // Warp 0 (threads 0-31):  covers k-dimensions 0..7
    // Warp 1 (threads 32-63): covers k-dimensions 8..15
    // Both warps produce the same full 16×16 output; single MMA call per K-tile.
    // A: row = lane & 0xf, k_off = ((lane>>4)<<2)
    // B: k_off = ((lane>>4)<<2), token = lane & 0xf
    // C: row_base = ((lane>>4)<<2), col = lane & 0xf
    const int a_row = lane & 0xf;
    const int a_k_off = ((lane >> 4) << 2);
    const int b_token = lane & 0xf;
    const int b_k_off = ((lane >> 4) << 2);
    const int c_row_base = ((lane >> 4) << 2);
    const int c_col = lane & 0xf;
    const int warp_id = tid >> 5;

    // -------- Pre-load Q into MMA fragments ----------
    // Each warp loads different k-dims: warp0 k=0..7, warp1 k=8..15
    const __nv_bfloat16 zero_bf16 = __float2bfloat16(0.0f);
    mma_bf16_fragment_t q_frag[MMA_K_TILES];

#pragma unroll
    for (int kt = 0; kt < MMA_K_TILES; ++kt) {
        const int k_base = kt * MMA_TILE + warp_id * 8;
        if (a_row < kGroup) {
            const int query_head =
                kv_head * kGroup + a_row;
            const int64_t q_base =
                (static_cast<int64_t>(b) * kNumHeads +
                 query_head) *
                HEAD_DIM;

            q_frag[kt].element[0] =
                q[q_base + k_base + 0 + a_k_off];
            q_frag[kt].element[1] =
                q[q_base + k_base + 1 + a_k_off];
            q_frag[kt].element[2] =
                q[q_base + k_base + 2 + a_k_off];
            q_frag[kt].element[3] =
                q[q_base + k_base + 3 + a_k_off];
        } else {
            q_frag[kt].element[0] = zero_bf16;
            q_frag[kt].element[1] = zero_bf16;
            q_frag[kt].element[2] = zero_bf16;
            q_frag[kt].element[3] = zero_bf16;
        }
    }

    __shared__ float shared_output_acc[kGroup][HEAD_DIM];

    /* Zero shared output accumulator */
    for (int i = tid; i < kGroup * HEAD_DIM; i += MMA_THREADS) {
        ((float*)shared_output_acc)[i] = 0.0f;
    }

    /* Initialize per-head running max/sum */
    if (tid < kGroup) {
        shared_running_max[tid] = -CUDART_INF_F;
        shared_running_sum[tid] = 0.0f;
    }
    __syncthreads();

    mma_float_fragment_t score_fragment;

    for (int page = 0; page < valid_pages; ++page) {
        const int physical_block = block_row[page];
        const int remaining = seqlen - page * PAGE_SIZE;
        const int page_tokens =
            remaining < PAGE_SIZE ? remaining : PAGE_SIZE;

        const int64_t page_base =
            static_cast<int64_t>(physical_block) * kPageStride +
            head_offset;
        const int pack_count =
            page_tokens * kPacksPerToken;

        /* ---- Sync load K and V (diagnostic: replace CP-async) ---- */
        for (int pack = tid;
             pack < pack_count;
             pack += MMA_THREADS) {
            const int token = pack / kPacksPerToken;
            const int pack_in_token =
                pack - token * kPacksPerToken;
            const int dim = pack_in_token * 8;

            const int64_t global_offset =
                page_base +
                static_cast<int64_t>(token) * kTokenStride +
                dim;
            const int shared_offset =
                token * HEAD_DIM + dim;

            *reinterpret_cast<int4*>(shared_k + shared_offset) =
                *reinterpret_cast<const int4*>(k_cache_paged + global_offset);

            *reinterpret_cast<int4*>(shared_v + shared_offset) =
                *reinterpret_cast<const int4*>(v_cache_paged + global_offset);
        }

        __syncthreads();

        /* ---- Zero unused V tokens to prevent MMA NaN (0 * NaN = NaN) ---- */
        for (int t = page_tokens + tid; t < PAGE_SIZE; t += MMA_THREADS) {
            for (int d = 0; d < HEAD_DIM; d += 8) {
                *reinterpret_cast<int4*>(shared_v + t * HEAD_DIM + d) =
                    make_int4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        /* ---- MMA dot(Q,K): single call per K-tile (64 threads) ---- */
        /* Warp 0 contributes k=0..7, Warp 1 contributes k=8..15 */
        zero_mma_accumulator(score_fragment);

#pragma unroll
        for (int kt = 0; kt < MMA_K_TILES; ++kt) {
            mma_bf16_fragment_t k_frag;

            const int k_base =
                kt * MMA_TILE + warp_id * 8 + b_k_off;
            k_frag.element[0] =
                shared_k[b_token * HEAD_DIM + k_base + 0];
            k_frag.element[1] =
                shared_k[b_token * HEAD_DIM + k_base + 1];
            k_frag.element[2] =
                shared_k[b_token * HEAD_DIM + k_base + 2];
            k_frag.element[3] =
                shared_k[b_token * HEAD_DIM + k_base + 3];

            score_fragment.vector =
                __builtin_mxc_mma_16x16x16bf16(
                    q_frag[kt].vector,
                    k_frag.vector,
                    score_fragment.vector);
        }

        /* Extract scores: C[row_base + i][col] for i=0..3 */
        /* Both warps hold the same complete result; only warp0 writes */
        if (warp_id == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = c_row_base + i;
                if (row < kGroup) {
                    shared_scores[row][c_col] =
                        score_fragment.element[i] * kScale;
                }
            }
        }

        __syncthreads();

        /* ---- Online softmax (one thread per query head) ---- */
        if (tid < kGroup) {
            const int head = tid;
            float page_max = -CUDART_INF_F;

#pragma unroll
            for (int t = 0; t < PAGE_SIZE; ++t) {
                if (t < page_tokens) {
                    page_max = fmaxf(
                        page_max,
                        shared_scores[head][t]);
                }
            }

            const float merged_max =
                fmaxf(shared_running_max[head], page_max);
            const float alpha =
                shared_running_sum[head] > 0.0f
                    ? expf(shared_running_max[head] - merged_max)
                    : 0.0f;

            float page_sum = 0.0f;
#pragma unroll
            for (int t = 0; t < PAGE_SIZE; ++t) {
                float prob = 0.0f;
                if (t < page_tokens) {
                    prob = expf(
                        shared_scores[head][t] -
                        merged_max);
                }
                shared_scores[head][t] = prob;
                page_sum += prob;
            }

            shared_page_alpha[tid] = alpha;
            shared_merged_max[tid] = merged_max;
            shared_new_sum[tid] =
                shared_running_sum[head] * alpha + page_sum;
        }

        __syncthreads();

        /* ---- Rescale accumulated output by alpha ---- */
        {
            const float alpha = shared_page_alpha[head_in_group];
            if (head_in_group < kGroup) {
#pragma unroll
                for (int d = 0; d < kDimsPerThread; ++d) {
                    shared_output_acc[head_in_group][dim_base + d] *= alpha;
                }
            }
        }

        shared_running_max[head_in_group] =
            shared_merged_max[head_in_group];
        shared_running_sum[head_in_group] =
            shared_new_sum[head_in_group];

        __syncthreads();

        /* ---- MMA P×V: prob × V → accumulate to shared_output_acc ---- */
        {
            mma_float_fragment_t pv_acc;

#pragma unroll
            for (int kt = 0; kt < MMA_K_TILES; ++kt) {
                /* A fragment: prob[head][token] from shared_scores */
                mma_bf16_fragment_t prob_frag;
                const int token_base = warp_id * 8 + a_k_off;
                if (a_row < kGroup) {
                    prob_frag.element[0] =
                        __float2bfloat16(shared_scores[a_row][token_base + 0]);
                    prob_frag.element[1] =
                        __float2bfloat16(shared_scores[a_row][token_base + 1]);
                    prob_frag.element[2] =
                        __float2bfloat16(shared_scores[a_row][token_base + 2]);
                    prob_frag.element[3] =
                        __float2bfloat16(shared_scores[a_row][token_base + 3]);
                } else {
                    for (int i = 0; i < 4; ++i)
                        prob_frag.element[i] = __float2bfloat16(0.0f);
                }

                /* B fragment: V[token][dim] from shared_v */
                mma_bf16_fragment_t v_frag;
                v_frag.element[0] =
                    shared_v[(warp_id * 8 + b_k_off + 0) * HEAD_DIM + kt * MMA_TILE + b_token];
                v_frag.element[1] =
                    shared_v[(warp_id * 8 + b_k_off + 1) * HEAD_DIM + kt * MMA_TILE + b_token];
                v_frag.element[2] =
                    shared_v[(warp_id * 8 + b_k_off + 2) * HEAD_DIM + kt * MMA_TILE + b_token];
                v_frag.element[3] =
                    shared_v[(warp_id * 8 + b_k_off + 3) * HEAD_DIM + kt * MMA_TILE + b_token];

                /* Zero accumulator before each MMA */
                for (int i = 0; i < 4; ++i) pv_acc.element[i] = 0.0f;

                /* MMA: C[head][kt_dims] = prob × V */
                pv_acc.vector =
                    __builtin_mxc_mma_16x16x16bf16(
                        prob_frag.vector,
                        v_frag.vector,
                        pv_acc.vector);

                /* Accumulate into shared_output_acc (warp0 only, both warps have same result) */
                if (warp_id == 0) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int row = c_row_base + i;
                        if (row < kGroup) {
                            shared_output_acc[row][kt * MMA_TILE + c_col] +=
                                pv_acc.element[i];
                        }
                    }
                }
            }
        }

        if (page + 1 < valid_pages) {
            __syncthreads();
        }
    }

    /* ---- Write output directly from shared_output_acc ---- */
    __syncthreads();
    if (head_in_group < kGroup) {
        const int query_head =
            kv_head * kGroup + head_in_group;
        const int64_t output_base =
            (static_cast<int64_t>(b) * kNumHeads +
             query_head) *
            HEAD_DIM;

        const float inv_denom =
            shared_running_sum[head_in_group] > 0.0f
                ? 1.0f / shared_running_sum[head_in_group]
                : 0.0f;

#pragma unroll
        for (int d = 0; d < kDimsPerThread; d += 2) {
            output[output_base + dim_base + d] =
                __float2bfloat16(
                    shared_output_acc[head_in_group][dim_base + d] *
                    inv_denom);
            output[output_base + dim_base + d + 1] =
                __float2bfloat16(
                    shared_output_acc[head_in_group][dim_base + d + 1] *
                    inv_denom);
        }
    }
}

template <int GROUP, int BDZ>
__device__ __forceinline__ void stage_page_sync(
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    __nv_bfloat16* __restrict__ shared_k,
    __nv_bfloat16* __restrict__ shared_v,
    int physical_block,
    int page_tokens,
    int kv_head) {
    constexpr int kNumHeads = 32;
    constexpr int kNumKVHeads = kNumHeads / GROUP;
    constexpr int kThreads = 16 * GROUP * BDZ;
    constexpr int kPacksPerToken = HEAD_DIM / 8;
    constexpr int64_t kTokenStride =
        static_cast<int64_t>(kNumKVHeads) * HEAD_DIM;
    constexpr int64_t kPageStride =
        static_cast<int64_t>(PAGE_SIZE) * kTokenStride;

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);
    const int linear_thread =
        tx + 16 * (ty + GROUP * tz);

    const int pack_count =
        page_tokens * kPacksPerToken;
    const int64_t page_base =
        static_cast<int64_t>(physical_block) * kPageStride +
        static_cast<int64_t>(kv_head) * HEAD_DIM;

    for (int pack = linear_thread;
         pack < pack_count;
         pack += kThreads) {
        const int token = pack >> 4;
        const int dim = (pack & 15) * 8;

        const int64_t global_offset =
            page_base +
            static_cast<int64_t>(token) * kTokenStride +
            dim;
        const int shared_offset =
            token * HEAD_DIM + dim;

        *reinterpret_cast<int4*>(
            shared_k + shared_offset) =
            *reinterpret_cast<const int4*>(
                k_cache_paged + global_offset);

        *reinterpret_cast<int4*>(
            shared_v + shared_offset) =
            *reinterpret_cast<const int4*>(
                v_cache_paged + global_offset);
    }
}

template <int GROUP, int BDZ>
__device__ __forceinline__ void finish_partition_state(
    float8_t accumulator,
    float running_max_log2,
    float running_sum,
    float* __restrict__ merge_o,
    float* __restrict__ merge_md,
    float* __restrict__ partial_o,
    float* __restrict__ partial_lse,
    int partial_index) {
    constexpr unsigned kMask = 0xffffffffu;

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);
    const int dimension_base = tx * 8;
    const int64_t partial_base =
        static_cast<int64_t>(partial_index) * HEAD_DIM;

    const float local_denominator =
        __shfl_sync(kMask, running_sum, 0, 16);
    const float local_max_log2 =
        __shfl_sync(kMask, running_max_log2, 0, 16);

    const int state_index = tz * GROUP + ty;
    const int64_t merge_base =
        static_cast<int64_t>(state_index) * HEAD_DIM;

    const float inverse_local =
        local_denominator > 0.0f
            ? 1.0f / local_denominator
            : 0.0f;

    scale_float8(accumulator, inverse_local);
    store_float8(
        merge_o + merge_base + dimension_base,
        accumulator);

    if (tx == 0) {
        merge_md[state_index * 2] =
            local_denominator > 0.0f
                ? local_max_log2 + log2f(local_denominator)
                : -CUDART_INF_F;
        merge_md[state_index * 2 + 1] = 0.0f;
    }

    __syncthreads();

    if (tz == 0) {
        float max_lse_log2 = -CUDART_INF_F;
        float weight_denominator = 0.0f;

        if (tx == 0) {
#pragma unroll
            for (int z = 0; z < BDZ; ++z) {
                const int index = z * GROUP + ty;
                max_lse_log2 =
                    fmaxf(
                        max_lse_log2,
                        merge_md[index * 2]);
            }

#pragma unroll
            for (int z = 0; z < BDZ; ++z) {
                const int index = z * GROUP + ty;
                const float lse_log2 =
                    merge_md[index * 2];
                const float weight =
                    lse_log2 == -CUDART_INF_F
                        ? 0.0f
                        : exp2f(lse_log2 - max_lse_log2);

                merge_md[index * 2 + 1] = weight;
                weight_denominator += weight;
            }
        }

        max_lse_log2 =
            __shfl_sync(kMask, max_lse_log2, 0, 16);
        weight_denominator =
            __shfl_sync(kMask, weight_denominator, 0, 16);

        __syncwarp(kMask);

        float8_t merged_output;
        zero_float8(merged_output);

#pragma unroll
        for (int z = 0; z < BDZ; ++z) {
            const int index = z * GROUP + ty;
            const float weight =
                merge_md[index * 2 + 1];
            const int64_t source_base =
                static_cast<int64_t>(index) * HEAD_DIM;

            const float8_t source =
                load_float8(
                    merge_o +
                    source_base +
                    dimension_base);

            fma_float8(merged_output, source, weight);
        }

        const float inverse_weight =
            weight_denominator > 0.0f
                ? 1.0f / weight_denominator
                : 0.0f;

        scale_float8(merged_output, inverse_weight);
        store_float8(
            partial_o + partial_base + dimension_base,
            merged_output);

        if (tx == 0) {
            partial_lse[partial_index] =
                weight_denominator > 0.0f
                    ? max_lse_log2 +
                        log2f(weight_denominator)
                    : -CUDART_INF_F;
        }
    }
}

template <int GROUP, int BDZ>
__global__ __launch_bounds__(16 * GROUP * BDZ)
void paged_decode_partition_baseline_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    float* __restrict__ partial_o,
    float* __restrict__ partial_lse,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int splits) {
    static_assert(
        GROUP == 4 || GROUP == 8,
        "Unsupported GQA group");
    static_assert(
        16 * GROUP * BDZ == 256,
        "Expected 256 threads");

    __shared__ __align__(16)
        __nv_bfloat16 k_stage[2][PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(16)
        __nv_bfloat16 v_stage[2][PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(64)
        float score_or_merge[BDZ][GROUP][PAGE_SIZE];

    constexpr unsigned kMask = 0xffffffffu;
    constexpr int kNumHeads = 32;
    constexpr int kNumKVHeads = kNumHeads / GROUP;
    constexpr float kScaleLog2 =
        0.08838834764831845f *
        1.4426950408889634074f;

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);
    const int split = static_cast<int>(blockIdx.y);

    const int bkv = static_cast<int>(blockIdx.x);
    const int b = bkv / kNumKVHeads;
    const int kv_head = bkv - b * kNumKVHeads;
    const int query_head = kv_head * GROUP + ty;
    const int bh = b * kNumHeads + query_head;

    const int seqlen = cache_seqlens[b];
    const int valid_pages =
        (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;

    const int first_page =
        static_cast<int>(
            (static_cast<int64_t>(valid_pages) * split) /
            splits);
    const int last_page =
        static_cast<int>(
            (static_cast<int64_t>(valid_pages) * (split + 1)) /
            splits);

    const int partial_index = bh * splits + split;
    const int64_t partial_base =
        static_cast<int64_t>(partial_index) * HEAD_DIM;
    const int dimension_base = tx * 8;

    if (first_page >= last_page) {
        if (tz == 0) {
            if (tx == 0) {
                partial_lse[partial_index] = -CUDART_INF_F;
            }

            float8_t zero;
            zero_float8(zero);
            store_float8(
                partial_o + partial_base + dimension_base,
                zero);
        }
        return;
    }

    const int64_t q_base =
        (static_cast<int64_t>(b) * kNumHeads + query_head) *
        HEAD_DIM;
    const float8_t q_vector =
        load_bf16x8_as_float8(
            q + q_base + dimension_base);

    float8_t accumulator;
    zero_float8(accumulator);

    float running_max_log2 = -CUDART_INF_F;
    float running_sum = 0.0f;

    const int32_t* block_row =
        block_table + static_cast<int64_t>(b) * blocks_per_batch;

    int current_stage = 0;
    int current_tokens = 0;

    {
        const int physical = block_row[first_page];
        const int remaining =
            seqlen - first_page * PAGE_SIZE;
        current_tokens =
            remaining < PAGE_SIZE ? remaining : PAGE_SIZE;

        stage_page_sync<GROUP, BDZ>(
            k_cache_paged,
            v_cache_paged,
            k_stage[current_stage],
            v_stage[current_stage],
            physical,
            current_tokens,
            kv_head);
    }

    __syncthreads();

    for (int page = first_page; page < last_page; ++page) {
        const int next_page = page + 1;
        const int next_stage = current_stage ^ 1;

        if (next_page < last_page) {
            const int physical = block_row[next_page];
            const int remaining =
                seqlen - next_page * PAGE_SIZE;
            const int next_tokens =
                remaining < PAGE_SIZE ? remaining : PAGE_SIZE;

            stage_page_sync<GROUP, BDZ>(
                k_cache_paged,
                v_cache_paged,
                k_stage[next_stage],
                v_stage[next_stage],
                physical,
                next_tokens,
                kv_head);
        }

        float* score_row = score_or_merge[tz][ty];
        int worker_token_count = 0;

#pragma unroll 1
        for (int token = tz;
             token < current_tokens;
             token += BDZ) {
            const float8_t k_vector =
                load_bf16x8_as_float8(
                    k_stage[current_stage] +
                    token * HEAD_DIM +
                    dimension_base);

            float score =
                subgroup_sum_16(
                    dot_float8(q_vector, k_vector));

            if (tx == 0) {
                score_row[worker_token_count] =
                    score * kScaleLog2;
            }
            ++worker_token_count;
        }

        __syncwarp(kMask);

        const float lane_score =
            tx < worker_token_count
                ? score_row[tx]
                : -CUDART_INF_F;

        const float tile_max =
            subgroup_max_16(lane_score);

        float merged_max = 0.0f;
        float alpha = 0.0f;

        if (tx == 0) {
            merged_max =
                fmaxf(running_max_log2, tile_max);
            alpha =
                running_sum > 0.0f
                    ? exp2f(running_max_log2 - merged_max)
                    : 0.0f;
        }

        merged_max =
            __shfl_sync(kMask, merged_max, 0, 16);
        alpha =
            __shfl_sync(kMask, alpha, 0, 16);

        const float probability =
            tx < worker_token_count
                ? exp2f(lane_score - merged_max)
                : 0.0f;

        if (tx < worker_token_count) {
            score_row[tx] = probability;
        }

        const float tile_sum =
            subgroup_sum_16(probability);

        if (tx == 0) {
            running_sum =
                running_sum * alpha + tile_sum;
            running_max_log2 = merged_max;
        }

        scale_float8(accumulator, alpha);
        __syncwarp(kMask);

        int probability_index = 0;

#pragma unroll 1
        for (int token = tz;
             token < current_tokens;
             token += BDZ) {
            const float probability_value =
                score_row[probability_index++];

            const float8_t v_vector =
                load_bf16x8_as_float8(
                    v_stage[current_stage] +
                    token * HEAD_DIM +
                    dimension_base);

            fma_float8(
                accumulator,
                v_vector,
                probability_value);
        }

        __syncthreads();

        current_stage = next_stage;

        if (next_page < last_page) {
            const int remaining =
                seqlen - next_page * PAGE_SIZE;
            current_tokens =
                remaining < PAGE_SIZE ? remaining : PAGE_SIZE;
        }
    }

    float* merge_o =
        reinterpret_cast<float*>(&k_stage[0][0]);
    float* merge_md =
        &score_or_merge[0][0][0];

    finish_partition_state<GROUP, BDZ>(
        accumulator,
        running_max_log2,
        running_sum,
        merge_o,
        merge_md,
        partial_o,
        partial_lse,
        partial_index);
}

template <int GROUP, int BDZ>
__device__ __forceinline__ void async_stage_mid_tile(
    const __nv_bfloat16* __restrict__ cache,
    __nv_bfloat16* __restrict__ shared_tile,
    const int32_t* __restrict__ block_row,
    int first_page,
    int last_page,
    int seqlen,
    int kv_head) {
    constexpr int kNumHeads = 32;
    constexpr int kNumKVHeads = kNumHeads / GROUP;
    constexpr int kThreads = 16 * GROUP * BDZ;
    constexpr int kPacksPerPage =
        PAGE_SIZE * (HEAD_DIM / 8);
    constexpr int64_t kTokenStride =
        static_cast<int64_t>(kNumKVHeads) * HEAD_DIM;
    constexpr int64_t kPageStride =
        static_cast<int64_t>(PAGE_SIZE) * kTokenStride;

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);
    const int linear_thread =
        tx + 16 * (ty + GROUP * tz);

#pragma unroll
    for (int page_in_tile = 0;
         page_in_tile < MID_TILE_PAGES;
         ++page_in_tile) {
        const int logical_page =
            first_page + page_in_tile;

        if (logical_page < last_page) {
            const int physical_block =
                block_row[logical_page];
            const int remaining =
                seqlen - logical_page * PAGE_SIZE;
            const int page_tokens =
                remaining < PAGE_SIZE ? remaining : PAGE_SIZE;

            const int64_t page_base =
                static_cast<int64_t>(physical_block) *
                    kPageStride +
                static_cast<int64_t>(kv_head) * HEAD_DIM;

            for (int pack = linear_thread;
                 pack < kPacksPerPage;
                 pack += kThreads) {
                const int token = pack >> 4;
                const int dim = (pack & 15) * 8;
                const bool valid = token < page_tokens;

                const int64_t global_offset =
                    page_base +
                    static_cast<int64_t>(token) *
                        kTokenStride +
                    dim;
                const int shared_offset =
                    page_in_tile * PAGE_SIZE * HEAD_DIM +
                    token * HEAD_DIM +
                    dim;

                flashinfer::cp_async::load_128b_bsm_pred(
                    shared_tile + shared_offset,
                    cache + global_offset,
                    valid);
            }
        }
    }
}

template <int GROUP, int BDZ>
__global__ __launch_bounds__(16 * GROUP * BDZ)
void paged_decode_mid_async_chunk_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    float* __restrict__ partial_o,
    float* __restrict__ partial_lse,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int num_chunks) {
    static_assert(
        GROUP == 4 || GROUP == 8,
        "Unsupported GQA group");
    static_assert(
        16 * GROUP * BDZ == 256,
        "Expected 256 threads");
    static_assert(
        MID_TILE_TOKENS / BDZ <= 16,
        "Worker tile exceeds subgroup");

    __shared__ __align__(16)
        __nv_bfloat16
        k_stage[2][MID_TILE_TOKENS * HEAD_DIM];
    __shared__ __align__(16)
        __nv_bfloat16
        v_stage[MID_TILE_TOKENS * HEAD_DIM];
    __shared__ __align__(64)
        float score_or_merge[BDZ][GROUP][16];

    constexpr unsigned kMask = 0xffffffffu;
    constexpr int kNumHeads = 32;
    constexpr int kNumKVHeads = kNumHeads / GROUP;
    constexpr float kScaleLog2 =
        0.08838834764831845f *
        1.4426950408889634074f;

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);
    const int chunk = static_cast<int>(blockIdx.y);

    const int bkv = static_cast<int>(blockIdx.x);
    const int b = bkv / kNumKVHeads;
    const int kv_head = bkv - b * kNumKVHeads;
    const int query_head = kv_head * GROUP + ty;
    const int bh = b * kNumHeads + query_head;

    const int seqlen = cache_seqlens[b];
    const int valid_pages =
        (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;

    const int chunk_first_page =
        chunk * MID_CHUNK_PAGES;
    const int chunk_last_page =
        min(
            valid_pages,
            chunk_first_page + MID_CHUNK_PAGES);

    const int partial_index =
        bh * num_chunks + chunk;
    const int64_t partial_base =
        static_cast<int64_t>(partial_index) * HEAD_DIM;
    const int dimension_base = tx * 8;

    if (chunk_first_page >= chunk_last_page) {
        if (tz == 0) {
            if (tx == 0) {
                partial_lse[partial_index] = -CUDART_INF_F;
            }

            float8_t zero;
            zero_float8(zero);
            store_float8(
                partial_o + partial_base + dimension_base,
                zero);
        }
        return;
    }

    const int64_t q_base =
        (static_cast<int64_t>(b) * kNumHeads + query_head) *
        HEAD_DIM;
    const float8_t q_vector =
        load_bf16x8_as_float8(
            q + q_base + dimension_base);

    float8_t accumulator;
    zero_float8(accumulator);

    float running_max_log2 = -CUDART_INF_F;
    float running_sum = 0.0f;

    const int32_t* block_row =
        block_table + static_cast<int64_t>(b) * blocks_per_batch;

    int current_stage = 0;
    int current_first_page = chunk_first_page;
    int current_last_page =
        min(
            chunk_last_page,
            current_first_page + MID_TILE_PAGES);
    int current_tokens =
        min(
            seqlen - current_first_page * PAGE_SIZE,
            MID_TILE_TOKENS);

    async_stage_mid_tile<GROUP, BDZ>(
        k_cache_paged,
        k_stage[current_stage],
        block_row,
        current_first_page,
        current_last_page,
        seqlen,
        kv_head);

    async_stage_mid_tile<GROUP, BDZ>(
        v_cache_paged,
        v_stage,
        block_row,
        current_first_page,
        current_last_page,
        seqlen,
        kv_head);

    async_wait_all();
    __syncthreads();

    bool current_v_pending = false;

    while (current_first_page < chunk_last_page) {
        const int next_first_page =
            current_first_page + MID_TILE_PAGES;
        const bool has_next =
            next_first_page < chunk_last_page;
        const int next_stage = current_stage ^ 1;

        int next_last_page = next_first_page;
        int next_tokens = 0;

        if (has_next) {
            next_last_page =
                min(
                    chunk_last_page,
                    next_first_page + MID_TILE_PAGES);
            next_tokens =
                min(
                    seqlen - next_first_page * PAGE_SIZE,
                    MID_TILE_TOKENS);

            async_stage_mid_tile<GROUP, BDZ>(
                k_cache_paged,
                k_stage[next_stage],
                block_row,
                next_first_page,
                next_last_page,
                seqlen,
                kv_head);
        }

        float* score_row =
            score_or_merge[tz][ty];
        int worker_token_count = 0;

#pragma unroll 1
        for (int token = tz;
             token < current_tokens;
             token += BDZ) {
            const float8_t k_vector =
                load_bf16x8_as_float8(
                    k_stage[current_stage] +
                    token * HEAD_DIM +
                    dimension_base);

            float score =
                subgroup_sum_16(
                    dot_float8(q_vector, k_vector));

            if (tx == 0) {
                score_row[worker_token_count] =
                    score * kScaleLog2;
            }

            ++worker_token_count;
        }

        __syncwarp(kMask);

        const float lane_score =
            tx < worker_token_count
                ? score_row[tx]
                : -CUDART_INF_F;

        const float tile_max_log2 =
            subgroup_max_16(lane_score);

        float merged_max_log2 = 0.0f;
        float alpha = 0.0f;

        if (tx == 0) {
            merged_max_log2 =
                fmaxf(
                    running_max_log2,
                    tile_max_log2);
            alpha =
                running_sum > 0.0f
                    ? exp2f(
                        running_max_log2 -
                        merged_max_log2)
                    : 0.0f;
        }

        merged_max_log2 =
            __shfl_sync(
                kMask,
                merged_max_log2,
                0,
                16);
        alpha =
            __shfl_sync(kMask, alpha, 0, 16);

        const float probability =
            tx < worker_token_count
                ? exp2f(
                    lane_score -
                    merged_max_log2)
                : 0.0f;

        if (tx < worker_token_count) {
            score_row[tx] = probability;
        }

        const float tile_sum =
            subgroup_sum_16(probability);

        if (tx == 0) {
            running_sum =
                running_sum * alpha + tile_sum;
            running_max_log2 = merged_max_log2;
        }

        scale_float8(accumulator, alpha);
        __syncwarp(kMask);

        if (current_v_pending || has_next) {
            async_wait_all();
            __syncthreads();
        }

        int probability_index = 0;

#pragma unroll 1
        for (int token = tz;
             token < current_tokens;
             token += BDZ) {
            const float token_probability =
                score_row[probability_index++];

            const float8_t v_vector =
                load_bf16x8_as_float8(
                    v_stage +
                    token * HEAD_DIM +
                    dimension_base);

            fma_float8(
                accumulator,
                v_vector,
                token_probability);
        }

        __syncthreads();

        if (!has_next) {
            break;
        }

        async_stage_mid_tile<GROUP, BDZ>(
            v_cache_paged,
            v_stage,
            block_row,
            next_first_page,
            next_last_page,
            seqlen,
            kv_head);

        current_v_pending = true;
        current_stage = next_stage;
        current_first_page = next_first_page;
        current_last_page = next_last_page;
        current_tokens = next_tokens;
    }

    async_wait_all();
    __syncthreads();

    float* merge_o =
        reinterpret_cast<float*>(&k_stage[0][0]);
    float* merge_md =
        &score_or_merge[0][0][0];

    finish_partition_state<GROUP, BDZ>(
        accumulator,
        running_max_log2,
        running_sum,
        merge_o,
        merge_md,
        partial_o,
        partial_lse,
        partial_index);
}

template <int GROUP, int BDZ>
__device__ __forceinline__ void native_async_stage_page(
    const __nv_bfloat16* __restrict__ cache,
    __nv_bfloat16* __restrict__ shared_page,
    const int32_t* __restrict__ block_row,
    int logical_page,
    int seqlen,
    int kv_head) {
    constexpr int kNumHeads = 32;
    constexpr int kNumKVHeads = kNumHeads / GROUP;
    constexpr int kThreads = 16 * GROUP * BDZ;
    constexpr int kPacksPerToken = HEAD_DIM / 8;
    constexpr int kPacksPerPage =
        PAGE_SIZE * kPacksPerToken;
    constexpr int64_t kTokenStride =
        static_cast<int64_t>(kNumKVHeads) * HEAD_DIM;
    constexpr int64_t kPageStride =
        static_cast<int64_t>(PAGE_SIZE) * kTokenStride;

    static_assert(
        kThreads == kPacksPerPage,
        "One 128-bit page pack per thread is required");

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);
    const int linear_thread =
        tx + 16 * (ty + GROUP * tz);

    const int physical_block = block_row[logical_page];
    const int token = linear_thread >> 4;
    const int dim = (linear_thread & 15) * 8;

    const int remaining =
        seqlen - logical_page * PAGE_SIZE;
    const int page_tokens =
        remaining < PAGE_SIZE ? remaining : PAGE_SIZE;
    const bool valid = token < page_tokens;

    const int64_t page_base =
        static_cast<int64_t>(physical_block) * kPageStride +
        static_cast<int64_t>(kv_head) * HEAD_DIM;
    const int64_t global_offset =
        page_base +
        static_cast<int64_t>(token) * kTokenStride +
        dim;
    const int shared_offset =
        token * HEAD_DIM + dim;

    flashinfer::cp_async::load_128b_bsm_pred(
        shared_page + shared_offset,
        cache + global_offset,
        valid);
}

template <int GROUP, int BDZ>
__global__ __launch_bounds__(16 * GROUP * BDZ)
void paged_decode_native_maca_partition_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    float* __restrict__ partial_o,
    float* __restrict__ partial_lse,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int chunk_pages,
    int num_chunks) {
    static_assert(
        GROUP == 4 || GROUP == 8,
        "Unsupported GQA group");
    static_assert(
        16 * GROUP * BDZ == 256,
        "Native specialization requires 256 threads");
    static_assert(
        NATIVE_PIPELINE_STAGES == 2,
        "This specialization requires two stages");

    __shared__ __align__(16)
        __nv_bfloat16
        k_stage[NATIVE_PIPELINE_STAGES][PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(16)
        __nv_bfloat16
        v_stage[NATIVE_PIPELINE_STAGES][PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(64)
        float score_or_merge[BDZ][GROUP][PAGE_SIZE];

    constexpr unsigned kMask = 0xffffffffu;
    constexpr int kNumHeads = 32;
    constexpr int kNumKVHeads = kNumHeads / GROUP;
    constexpr float kScaleLog2 =
        0.08838834764831845f *
        1.4426950408889634074f;

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);

    const int chunk = static_cast<int>(blockIdx.y);
    const int bkv = static_cast<int>(blockIdx.x);
    const int b = bkv / kNumKVHeads;
    const int kv_head = bkv - b * kNumKVHeads;
    const int query_head = kv_head * GROUP + ty;
    const int bh = b * kNumHeads + query_head;

    const int seqlen = cache_seqlens[b];
    const int valid_pages =
        (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;

    const int chunk_first_page =
        chunk * chunk_pages;
    const int chunk_last_page =
        min(valid_pages, chunk_first_page + chunk_pages);

    const int partial_index =
        bh * num_chunks + chunk;
    const int64_t partial_base =
        static_cast<int64_t>(partial_index) * HEAD_DIM;
    const int dimension_base = tx * 8;

    if (chunk_first_page >= chunk_last_page) {
        if (tz == 0) {
            if (tx == 0) {
                partial_lse[partial_index] = -CUDART_INF_F;
            }

            float8_t zero;
            zero_float8(zero);
            store_float8(
                partial_o + partial_base + dimension_base,
                zero);
        }
        return;
    }

    const int64_t q_base =
        (static_cast<int64_t>(b) * kNumHeads + query_head) *
        HEAD_DIM;
    const float8_t q_vector =
        load_bf16x8_as_float8(
            q + q_base + dimension_base);

    float8_t accumulator;
    zero_float8(accumulator);

    float running_max_log2 = -CUDART_INF_F;
    float running_sum = 0.0f;

    const int32_t* block_row =
        block_table + static_cast<int64_t>(b) * blocks_per_batch;

    int current_stage = 0;
    int current_page = chunk_first_page;

    native_async_stage_page<GROUP, BDZ>(
        k_cache_paged,
        k_stage[current_stage],
        block_row,
        current_page,
        seqlen,
        kv_head);

    native_async_stage_page<GROUP, BDZ>(
        v_cache_paged,
        v_stage[current_stage],
        block_row,
        current_page,
        seqlen,
        kv_head);

    async_wait_all();
    __syncthreads();

    for (; current_page < chunk_last_page; ++current_page) {
        const int next_page = current_page + 1;
        const bool has_next = next_page < chunk_last_page;
        const int next_stage = current_stage ^ 1;

        if (has_next) {
            native_async_stage_page<GROUP, BDZ>(
                k_cache_paged,
                k_stage[next_stage],
                block_row,
                next_page,
                seqlen,
                kv_head);

            native_async_stage_page<GROUP, BDZ>(
                v_cache_paged,
                v_stage[next_stage],
                block_row,
                next_page,
                seqlen,
                kv_head);
        }

        const int remaining =
            seqlen - current_page * PAGE_SIZE;
        const int current_tokens =
            remaining < PAGE_SIZE ? remaining : PAGE_SIZE;

        float* score_row =
            score_or_merge[tz][ty];
        int worker_token_count = 0;

#pragma unroll
        for (int token = tz;
             token < PAGE_SIZE;
             token += BDZ) {
            if (token < current_tokens) {
                const float8_t k_vector =
                    load_bf16x8_as_float8(
                        k_stage[current_stage] +
                        token * HEAD_DIM +
                        dimension_base);

                float score =
                    subgroup_sum_16(
                        dot_float8(q_vector, k_vector));

                if (tx == 0) {
                    score_row[worker_token_count] =
                        score * kScaleLog2;
                }

                ++worker_token_count;
            }
        }

        __syncwarp(kMask);

        const float lane_score =
            tx < worker_token_count
                ? score_row[tx]
                : -CUDART_INF_F;

        const float page_max_log2 =
            subgroup_max_16(lane_score);

        float merged_max_log2 = 0.0f;
        float alpha = 0.0f;

        if (tx == 0) {
            merged_max_log2 =
                fmaxf(
                    running_max_log2,
                    page_max_log2);
            alpha =
                running_sum > 0.0f
                    ? exp2f(
                        running_max_log2 -
                        merged_max_log2)
                    : 0.0f;
        }

        merged_max_log2 =
            __shfl_sync(
                kMask,
                merged_max_log2,
                0,
                16);
        alpha =
            __shfl_sync(kMask, alpha, 0, 16);

        const float probability =
            tx < worker_token_count
                ? exp2f(
                    lane_score -
                    merged_max_log2)
                : 0.0f;

        if (tx < worker_token_count) {
            score_row[tx] = probability;
        }

        const float page_sum =
            subgroup_sum_16(probability);

        if (tx == 0) {
            running_sum =
                running_sum * alpha + page_sum;
            running_max_log2 = merged_max_log2;
        }

        scale_float8(accumulator, alpha);
        __syncwarp(kMask);

        int probability_index = 0;

#pragma unroll
        for (int token = tz;
             token < PAGE_SIZE;
             token += BDZ) {
            if (token < current_tokens) {
                const float token_probability =
                    score_row[probability_index++];

                const float8_t v_vector =
                    load_bf16x8_as_float8(
                        v_stage[current_stage] +
                        token * HEAD_DIM +
                        dimension_base);

                fma_float8(
                    accumulator,
                    v_vector,
                    token_probability);
            }
        }

        if (has_next) {
            async_wait_all();
            __syncthreads();
            current_stage = next_stage;
        }
    }

    __syncthreads();

    float* merge_o =
        reinterpret_cast<float*>(&k_stage[0][0]);
    float* merge_md =
        &score_or_merge[0][0][0];

    finish_partition_state<GROUP, BDZ>(
        accumulator,
        running_max_log2,
        running_sum,
        merge_o,
        merge_md,
        partial_o,
        partial_lse,
        partial_index);
}

__global__ __launch_bounds__(HEAD_DIM)
void combine_split_lse_kernel(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_lse,
    __nv_bfloat16* __restrict__ output,
    int splits) {
    __shared__ float weights[MAX_SPLITS];
    __shared__ float inverse_denominator;

    const int tid = threadIdx.x;
    const int bh = blockIdx.x;
    const int first_index = bh * splits;

    if (tid == 0) {
        float max_lse = -CUDART_INF_F;

        for (int split = 0; split < splits; ++split) {
            max_lse =
                fmaxf(
                    max_lse,
                    partial_lse[first_index + split]);
        }

        float denominator = 0.0f;

        for (int split = 0; split < splits; ++split) {
            const float lse =
                partial_lse[first_index + split];
            const float weight =
                lse == -CUDART_INF_F
                    ? 0.0f
                    : exp2f(lse - max_lse);

            weights[split] = weight;
            denominator += weight;
        }

        inverse_denominator =
            denominator > 0.0f
                ? 1.0f / denominator
                : 0.0f;
    }

    __syncthreads();

    float accumulator = 0.0f;

    for (int split = 0; split < splits; ++split) {
        const int index = first_index + split;

        accumulator =
            fmaf(
                weights[split],
                partial_o[
                    static_cast<int64_t>(index) * HEAD_DIM +
                    tid],
                accumulator);
    }

    output[
        static_cast<int64_t>(bh) * HEAD_DIM + tid] =
        __float2bfloat16(
            accumulator * inverse_denominator);
}

static int choose_baseline_splits(
    int64_t batch_size,
    int64_t seqlen_k) {
    if (batch_size <= 1) {
        if (seqlen_k >= 8192) {
            return 64;
        }
        if (seqlen_k >= 4096) {
            return 32;
        }
        return 16;
    }

    if (batch_size <= 8) {
        if (seqlen_k >= 32768) {
            return 32;
        }
        if (seqlen_k >= 8192) {
            return 16;
        }
        if (seqlen_k >= 4096) {
            return 8;
        }
        return 4;
    }

    if (batch_size <= 16) {
        return seqlen_k >= 8192 ? 8 : 4;
    }

    if (batch_size <= 32) {
        return 4;
    }

    return 2;
}

static void choose_native_partition(
    int blocks_per_batch,
    int64_t seqlen_k,
    int* chunk_pages,
    int* num_chunks) {
    int minimum_chunk_pages =
        seqlen_k <= 8192
            ? NATIVE_MIN_CHUNK_PAGES
            : NATIVE_LONG_MIN_CHUNK_PAGES;

    int occupancy_chunk_pages =
        (blocks_per_batch + MAX_SPLITS - 1) / MAX_SPLITS;

    if (occupancy_chunk_pages < minimum_chunk_pages) {
        occupancy_chunk_pages = minimum_chunk_pages;
    }

    *chunk_pages = occupancy_chunk_pages;
    *num_chunks =
        (blocks_per_batch + occupancy_chunk_pages - 1) /
        occupancy_chunk_pages;

    if (*num_chunks > MAX_SPLITS) {
        *num_chunks = MAX_SPLITS;
        *chunk_pages =
            (blocks_per_batch + MAX_SPLITS - 1) / MAX_SPLITS;
    }

    if (*num_chunks < 1) {
        *num_chunks = 1;
    }
    if (*chunk_pages < 1) {
        *chunk_pages = 1;
    }
}

static bool reserve_split_workspace(
    size_t partial_o_elems,
    size_t partial_lse_elems) {
    if (partial_o_elems > g_partial_o_elems) {
        float* new_partial_o = nullptr;

        const cudaError_t status =
            cudaMalloc(
                reinterpret_cast<void**>(&new_partial_o),
                partial_o_elems * sizeof(float));

        if (status != cudaSuccess ||
            new_partial_o == nullptr) {
            return false;
        }

        if (g_partial_o != nullptr) {
            cudaFree(g_partial_o);
        }

        g_partial_o = new_partial_o;
        g_partial_o_elems = partial_o_elems;
    }

    if (partial_lse_elems > g_partial_lse_elems) {
        float* new_partial_lse = nullptr;

        const cudaError_t status =
            cudaMalloc(
                reinterpret_cast<void**>(&new_partial_lse),
                partial_lse_elems * sizeof(float));

        if (status != cudaSuccess ||
            new_partial_lse == nullptr) {
            return false;
        }

        if (g_partial_lse != nullptr) {
            cudaFree(g_partial_lse);
        }

        g_partial_lse = new_partial_lse;
        g_partial_lse_elems = partial_lse_elems;
    }

    return g_partial_o != nullptr &&
           g_partial_lse != nullptr;
}

static void launch_scalar_fallback(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache_paged,
    const __nv_bfloat16* v_cache_paged,
    __nv_bfloat16* output,
    const int32_t* cache_seqlens,
    const int32_t* block_table,
    int blocks_per_batch,
    int total_heads,
    int num_heads,
    int num_heads_k) {
    paged_decode_kernel
        <<<total_heads, HEAD_DIM>>>(
            q,
            k_cache_paged,
            v_cache_paged,
            output,
            cache_seqlens,
            block_table,
            blocks_per_batch,
            num_heads,
            num_heads_k);
}

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
    if (headdim != HEAD_DIM ||
        page_block_size != PAGE_SIZE ||
        batch_size <= 0 ||
        num_heads <= 0 ||
        num_heads_k <= 0 ||
        num_blocks <= 0 ||
        num_heads < num_heads_k ||
        num_heads % num_heads_k != 0) {
        return;
    }

    const int blocks_per_batch =
        static_cast<int>(num_blocks / batch_size);

    if (blocks_per_batch <= 0) {
        return;
    }

    const int h = static_cast<int>(num_heads);
    const int hk = static_cast<int>(num_heads_k);
    const int total_heads =
        static_cast<int>(batch_size * num_heads);
    const int64_t gqa_ratio =
        num_heads / num_heads_k;

    const bool contest_gqa =
        seqlen_q == 1 &&
        causal == 0 &&
        num_heads == 32 &&
        ((num_heads_k == 4 && gqa_ratio == 8) ||
         (num_heads_k == 8 && gqa_ratio == 4));

    const uintptr_t cache_alignment =
        reinterpret_cast<uintptr_t>(k_cache_paged) |
        reinterpret_cast<uintptr_t>(v_cache_paged);
    const uintptr_t q_alignment =
        reinterpret_cast<uintptr_t>(q);

    const bool aligned_packed_path =
        (q_alignment & 3u) == 0u &&
        (cache_alignment & 15u) == 0u;

    const bool aligned_async_path =
        (q_alignment & 15u) == 0u &&
        (cache_alignment & 15u) == 0u;

    const bool use_page_softmax_direct_gqa =
        contest_gqa &&
        aligned_packed_path &&
        batch_size >= 8 &&
        seqlen_k > 32 &&
        seqlen_k <= 2048;

    if (use_page_softmax_direct_gqa) {
        const int total_kv_heads =
            static_cast<int>(batch_size * num_heads_k);

        if (num_heads_k == 4) {
            paged_decode_gqa_mma_kernel<4>
                <<<total_kv_heads, MMA_THREADS>>>(
                    q,
                    k_cache_paged,
                    v_cache_paged,
                    output,
                    cache_seqlens,
                    block_table,
                    blocks_per_batch);
        } else {
            paged_decode_gqa_mma_kernel<8>
                <<<total_kv_heads, MMA_THREADS>>>(
                    q,
                    k_cache_paged,
                    v_cache_paged,
                    output,
                    cache_seqlens,
                    block_table,
                    blocks_per_batch);
        }
        return;
    }

    if (seqlen_k <= 2048 || !contest_gqa) {
        launch_scalar_fallback(
            q,
            k_cache_paged,
            v_cache_paged,
            output,
            cache_seqlens,
            block_table,
            blocks_per_batch,
            total_heads,
            h,
            hk);
        return;
    }

    if (!aligned_packed_path) {
        launch_scalar_fallback(
            q,
            k_cache_paged,
            v_cache_paged,
            output,
            cache_seqlens,
            block_table,
            blocks_per_batch,
            total_heads,
            h,
            hk);
        return;
    }

    /*
     * Isolated round-1 donor route. All accepted round-0 routes remain
     * unchanged outside the two measured winning bands:
     *   batch 16: 4096 <= KV < 16384
     *   batch  8: KV >= 32768
     */
    const bool use_native_partition =
        contest_gqa &&
        aligned_async_path &&
        ((batch_size == 16 &&
          seqlen_k >= 4096 &&
          seqlen_k < 16384) ||
         (batch_size == 8 &&
          seqlen_k >= 32768));

    bool use_mid_async_specialization =
        !use_native_partition &&
        contest_gqa &&
        aligned_async_path &&
        batch_size >= 16 &&
        seqlen_k > 2048 &&
        seqlen_k <= 16384;

    int splits = 1;
    int native_chunk_pages = 0;

    if (use_native_partition) {
        choose_native_partition(
            blocks_per_batch,
            seqlen_k,
            &native_chunk_pages,
            &splits);
    } else if (use_mid_async_specialization) {
        splits =
            (blocks_per_batch +
             MID_CHUNK_PAGES - 1) /
            MID_CHUNK_PAGES;

        if (splits < 1 || splits > MAX_SPLITS) {
            use_mid_async_specialization = false;
        }
    }

    if (!use_native_partition &&
        !use_mid_async_specialization) {
        splits =
            choose_baseline_splits(
                batch_size,
                seqlen_k);

        if (splits > MAX_SPLITS) {
            splits = MAX_SPLITS;
        }
        if (splits > blocks_per_batch) {
            splits = blocks_per_batch;
        }
        if (splits < 1) {
            splits = 1;
        }

        if (splits < 2) {
            launch_scalar_fallback(
                q,
                k_cache_paged,
                v_cache_paged,
                output,
                cache_seqlens,
                block_table,
                blocks_per_batch,
                total_heads,
                h,
                hk);
            return;
        }
    }

    const size_t partial_lse_elems =
        static_cast<size_t>(total_heads) *
        static_cast<size_t>(splits);
    const size_t partial_o_elems =
        partial_lse_elems * HEAD_DIM;

    if (!reserve_split_workspace(
            partial_o_elems,
            partial_lse_elems)) {
        launch_scalar_fallback(
            q,
            k_cache_paged,
            v_cache_paged,
            output,
            cache_seqlens,
            block_table,
            blocks_per_batch,
            total_heads,
            h,
            hk);
        return;
    }

    const int total_kv_heads =
        static_cast<int>(batch_size * num_heads_k);
    const dim3 split_grid(
        total_kv_heads,
        splits,
        1);

    if (use_native_partition) {
        if (gqa_ratio == 4) {
            const dim3 block(16, 4, 4);

            paged_decode_native_maca_partition_kernel<4, 4>
                <<<split_grid, block>>>(
                    q,
                    k_cache_paged,
                    v_cache_paged,
                    g_partial_o,
                    g_partial_lse,
                    cache_seqlens,
                    block_table,
                    blocks_per_batch,
                    native_chunk_pages,
                    splits);
        } else {
            const dim3 block(16, 8, 2);

            paged_decode_native_maca_partition_kernel<8, 2>
                <<<split_grid, block>>>(
                    q,
                    k_cache_paged,
                    v_cache_paged,
                    g_partial_o,
                    g_partial_lse,
                    cache_seqlens,
                    block_table,
                    blocks_per_batch,
                    native_chunk_pages,
                    splits);
        }
    } else if (use_mid_async_specialization) {
        if (gqa_ratio == 4) {
            const dim3 block(16, 4, 4);

            paged_decode_mid_async_chunk_kernel<4, 4>
                <<<split_grid, block>>>(
                    q,
                    k_cache_paged,
                    v_cache_paged,
                    g_partial_o,
                    g_partial_lse,
                    cache_seqlens,
                    block_table,
                    blocks_per_batch,
                    splits);
        } else {
            const dim3 block(16, 8, 2);

            paged_decode_mid_async_chunk_kernel<8, 2>
                <<<split_grid, block>>>(
                    q,
                    k_cache_paged,
                    v_cache_paged,
                    g_partial_o,
                    g_partial_lse,
                    cache_seqlens,
                    block_table,
                    blocks_per_batch,
                    splits);
        }
    } else {
        if (gqa_ratio == 4) {
            const dim3 block(16, 4, 4);

            paged_decode_partition_baseline_kernel<4, 4>
                <<<split_grid, block>>>(
                    q,
                    k_cache_paged,
                    v_cache_paged,
                    g_partial_o,
                    g_partial_lse,
                    cache_seqlens,
                    block_table,
                    blocks_per_batch,
                    splits);
        } else {
            const dim3 block(16, 8, 2);

            paged_decode_partition_baseline_kernel<8, 2>
                <<<split_grid, block>>>(
                    q,
                    k_cache_paged,
                    v_cache_paged,
                    g_partial_o,
                    g_partial_lse,
                    cache_seqlens,
                    block_table,
                    blocks_per_batch,
                    splits);
        }
    }

    combine_split_lse_kernel
        <<<total_heads, HEAD_DIM>>>(
            g_partial_o,
            g_partial_lse,
            output,
            splits);
}

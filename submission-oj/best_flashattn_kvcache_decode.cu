#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mcflashinfer/cp_async.cuh>
#include <mcflashinfer/utils.cuh>
#include <stddef.h>
#include <stdint.h>

typedef _Float16 maca_half4 __attribute__((__vector_size__(8)));
typedef float maca_float4 __attribute__((__vector_size__(16)));

#define HEAD_DIM 128
#define PAGE_SIZE 16
#define WARP_SIZE_ 32
#define MAX_SPLITS 64

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

union alignas(8) mma_bf16_fragment_t {
    maca_half4 vector;
    __nv_bfloat16 element[4];
};

union alignas(16) mma_float_fragment_t {
    maca_float4 vector;
    float element[4];
};

__device__ __forceinline__ maca_float4 issue_bf16_mma(
    maca_half4 a,
    maca_half4 b,
    maca_float4 c) {
    return __builtin_mxc_mma_16x16x16bf16(a, b, c);
}

__device__ __forceinline__ float fast_exp2(float x) {
    return __builtin_exp2f(x);
}

__device__ __forceinline__ float fast_log2(float x) {
    return __builtin_log2f(x);
}

__device__ __forceinline__ float subgroup_sum_16(float value) {
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(
            0xffffffffu, value, offset, 16);
    }
    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = WARP_SIZE_ / 2;
         offset > 0;
         offset >>= 1) {
        value += __shfl_xor_sync(
            0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum_128(float value) {
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
        result = fmaf(
            lhs.value[i],
            rhs.value[i],
            result);
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
        value.value[i] =
            fmaf(scale, value.value[i], 0.0f);
    }
}

__device__ __forceinline__ void fma_float8(
    float8_t& accumulator,
    const float8_t& value,
    float weight) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        accumulator.value[i] =
            fmaf(
                weight,
                value.value[i],
                accumulator.value[i]);
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

__device__ __forceinline__ float8_t load_float8(
    const float* ptr) {
    const float4 lo =
        *reinterpret_cast<const float4*>(ptr);
    const float4 hi =
        *reinterpret_cast<const float4*>(ptr + 4);

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

__device__ __forceinline__ void store_bf16x8(
    __nv_bfloat16* ptr,
    const float8_t& value,
    float scale) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        ptr[i] = __float2bfloat16(
            value.value[i] * scale);
    }
}

__global__ __launch_bounds__(HEAD_DIM)
void paged_decode_fallback_kernel(
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

    const int group_size = num_heads / num_heads_k;
    const int kv_head = h / group_size;
    const int seqlen = cache_seqlens[b];

    const int64_t output_base =
        static_cast<int64_t>(bh) * HEAD_DIM;

    if (seqlen <= 0) {
        output[output_base + tid] =
            __float2bfloat16(0.0f);
        return;
    }

    const int32_t* block_row =
        block_table +
        static_cast<int64_t>(b) * blocks_per_batch;

    const int64_t token_stride =
        static_cast<int64_t>(num_heads_k) * HEAD_DIM;
    const int64_t page_stride =
        static_cast<int64_t>(PAGE_SIZE) * token_stride;
    const int64_t head_offset =
        static_cast<int64_t>(kv_head) * HEAD_DIM;

    if (seqlen == 1) {
        const int physical_page = __ldg(block_row);
        const int64_t offset =
            static_cast<int64_t>(physical_page) *
                page_stride +
            head_offset +
            tid;

        output[output_base + tid] =
            v_cache_paged[offset];
        return;
    }

    constexpr float kScale =
        0.08838834764831845f;

    const int64_t q_base =
        static_cast<int64_t>(bh) * HEAD_DIM;
    const float q_value =
        __bfloat162float(__ldg(q + q_base + tid));

    float running_max = -CUDART_INF_F;
    float running_sum = 0.0f;
    float accumulator = 0.0f;

    const int valid_pages =
        (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;

    for (int logical_page = 0;
         logical_page < valid_pages;
         ++logical_page) {
        const int token_begin =
            logical_page * PAGE_SIZE;
        const int remaining =
            seqlen - token_begin;
        const int page_tokens =
            remaining < PAGE_SIZE
                ? remaining
                : PAGE_SIZE;

        const int physical_page =
            __ldg(block_row + logical_page);

        const int64_t page_base =
            static_cast<int64_t>(physical_page) *
                page_stride +
            head_offset;

        for (int token = 0;
             token < page_tokens;
             ++token) {
            const int64_t token_offset =
                page_base +
                static_cast<int64_t>(token) *
                    token_stride +
                tid;

            const float k_value =
                __bfloat162float(
                    __ldg(k_cache_paged + token_offset));

            const float local_dot =
                q_value * k_value;
            const float score =
                block_sum_128(local_dot) * kScale;

            const float v_value =
                __bfloat162float(
                    __ldg(v_cache_paged + token_offset));

            const float new_max =
                fmaxf(running_max, score);
            const float alpha =
                running_sum > 0.0f
                    ? expf(running_max - new_max)
                    : 0.0f;
            const float probability =
                expf(score - new_max);

            accumulator =
                fmaf(
                    alpha,
                    accumulator,
                    probability * v_value);
            running_sum =
                fmaf(
                    alpha,
                    running_sum,
                    probability);
            running_max = new_max;
        }
    }

    output[output_base + tid] =
        __float2bfloat16(
            running_sum > 0.0f
                ? accumulator / running_sum
                : 0.0f);
}

template <int GROUP>
__global__ __launch_bounds__(16 * GROUP)
void paged_decode_gqa_direct_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int num_heads,
    int num_heads_k) {
    static_assert(
        GROUP == 4 || GROUP == 8,
        "Unsupported GQA group");

    __shared__ __align__(16)
        __nv_bfloat16 shared_k[PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(16)
        __nv_bfloat16 shared_v[PAGE_SIZE * HEAD_DIM];

    constexpr int kThreads = 16 * GROUP;
    constexpr int kPacksPerToken = HEAD_DIM / 8;
    constexpr int kPacksPerPage =
        PAGE_SIZE * kPacksPerToken;
    constexpr float kScaleLog2 =
        0.08838834764831845f *
        1.4426950408889634074f;

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int linear_thread = tx + 16 * ty;

    const int bkv = static_cast<int>(blockIdx.x);
    const int b = bkv / num_heads_k;
    const int kv_head = bkv - b * num_heads_k;

    const int group_size = num_heads / num_heads_k;
    const int query_head =
        kv_head * group_size + ty;
    const int bh =
        b * num_heads + query_head;

    const int dimension_base = tx * 8;
    const int64_t output_base =
        static_cast<int64_t>(bh) * HEAD_DIM;

    const int seqlen = cache_seqlens[b];

    if (seqlen <= 0) {
        float8_t zero;
        zero_float8(zero);
        store_bf16x8(
            output + output_base + dimension_base,
            zero,
            0.0f);
        return;
    }

    const int32_t* block_row =
        block_table +
        static_cast<int64_t>(b) * blocks_per_batch;

    const int64_t token_stride =
        static_cast<int64_t>(num_heads_k) * HEAD_DIM;
    const int64_t page_stride =
        static_cast<int64_t>(PAGE_SIZE) * token_stride;
    const int64_t head_offset =
        static_cast<int64_t>(kv_head) * HEAD_DIM;

    if (seqlen == 1) {
        const int physical_page = __ldg(block_row);
        const int64_t value_offset =
            static_cast<int64_t>(physical_page) *
                page_stride +
            head_offset +
            dimension_base;

        const float8_t value =
            load_bf16x8_as_float8(
                v_cache_paged + value_offset);

        store_bf16x8(
            output + output_base + dimension_base,
            value,
            1.0f);
        return;
    }

    const int64_t q_base =
        static_cast<int64_t>(bh) * HEAD_DIM;
    const float8_t q_vector =
        load_bf16x8_as_float8(
            q + q_base + dimension_base);

    float8_t accumulator;
    zero_float8(accumulator);

    float running_max_log2 = -CUDART_INF_F;
    float running_sum = 0.0f;

    const int valid_pages =
        (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;

    for (int logical_page = 0;
         logical_page < valid_pages;
         ++logical_page) {
        const int token_begin =
            logical_page * PAGE_SIZE;
        const int remaining =
            seqlen - token_begin;
        const int page_tokens =
            remaining < PAGE_SIZE
                ? remaining
                : PAGE_SIZE;

        const int physical_page =
            __ldg(block_row + logical_page);

        const int64_t page_base =
            static_cast<int64_t>(physical_page) *
                page_stride +
            head_offset;

        const int pack_count =
            page_tokens * kPacksPerToken;

        for (int pack = linear_thread;
             pack < pack_count;
             pack += kThreads) {
            const int token =
                pack / kPacksPerToken;
            const int pack_in_token =
                pack - token * kPacksPerToken;
            const int dim = pack_in_token * 8;

            const int64_t global_offset =
                page_base +
                static_cast<int64_t>(token) *
                    token_stride +
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

        __syncthreads();

#pragma unroll 1
        for (int token = 0;
             token < page_tokens;
             ++token) {
            const int shared_offset =
                token * HEAD_DIM + dimension_base;

            const float8_t k_vector =
                load_bf16x8_as_float8(
                    shared_k + shared_offset);

            float score =
                subgroup_sum_16(
                    dot_float8(q_vector, k_vector));

            score *= kScaleLog2;

            const float new_max_log2 =
                fmaxf(running_max_log2, score);

            const float alpha =
                running_sum > 0.0f
                    ? fast_exp2(
                        running_max_log2 -
                        new_max_log2)
                    : 0.0f;

            const float probability =
                fast_exp2(
                    score - new_max_log2);

            scale_float8(accumulator, alpha);

            const float8_t v_vector =
                load_bf16x8_as_float8(
                    shared_v + shared_offset);

            fma_float8(
                accumulator,
                v_vector,
                probability);

            running_sum =
                fmaf(
                    alpha,
                    running_sum,
                    probability);
            running_max_log2 =
                new_max_log2;
        }

        __syncthreads();
    }

    const float inverse =
        running_sum > 0.0f
            ? 1.0f / running_sum
            : 0.0f;

    store_bf16x8(
        output + output_base + dimension_base,
        accumulator,
        inverse);
}

template <int GROUP, int BDZ>
__global__ __launch_bounds__(16 * GROUP * BDZ)
void paged_decode_gqa_partition_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    float* __restrict__ partial_o,
    float* __restrict__ partial_lse,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int splits,
    int num_heads,
    int num_heads_k) {
    static_assert(
        GROUP == 4 || GROUP == 8,
        "Unsupported GQA group");
    static_assert(
        16 * GROUP * BDZ == 256,
        "Partition kernel requires 256 threads");

    __shared__ __align__(16)
        __nv_bfloat16 shared_k[PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(16)
        __nv_bfloat16 shared_v[PAGE_SIZE * HEAD_DIM];
    __shared__ __align__(16)
        float merge_o[BDZ * GROUP * HEAD_DIM];
    __shared__ __align__(16)
        float merge_lse[BDZ * GROUP];
    __shared__ __align__(16)
        float merge_weight[BDZ * GROUP];
    __shared__ float merge_denominator[GROUP];
    __shared__ float merge_max_lse[GROUP];

    constexpr int kThreads = 16 * GROUP * BDZ;
    constexpr int kPacksPerToken = HEAD_DIM / 8;
    constexpr int kPacksPerPage =
        PAGE_SIZE * kPacksPerToken;
    constexpr float kScaleLog2 =
        0.08838834764831845f *
        1.4426950408889634074f;

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);

    const int linear_thread =
        tx + 16 * (ty + GROUP * tz);

    const int split =
        static_cast<int>(blockIdx.y);
    const int bkv =
        static_cast<int>(blockIdx.x);

    const int b = bkv / num_heads_k;
    const int kv_head = bkv - b * num_heads_k;

    const int group_size = num_heads / num_heads_k;
    const int query_head =
        kv_head * group_size + ty;
    const int bh =
        b * num_heads + query_head;

    const int seqlen = cache_seqlens[b];
    const int valid_pages =
        seqlen > 0
            ? (seqlen + PAGE_SIZE - 1) / PAGE_SIZE
            : 0;

    const int first_page =
        static_cast<int>(
            static_cast<int64_t>(valid_pages) *
            split / splits);
    const int last_page =
        static_cast<int>(
            static_cast<int64_t>(valid_pages) *
            (split + 1) / splits);

    const int partial_index =
        bh * splits + split;
    const int64_t partial_base =
        static_cast<int64_t>(partial_index) *
        HEAD_DIM;
    const int dimension_base = tx * 8;

    if (first_page >= last_page) {
        if (tz == 0) {
            float8_t zero;
            zero_float8(zero);

            store_float8(
                partial_o +
                    partial_base +
                    dimension_base,
                zero);

            if (tx == 0) {
                partial_lse[partial_index] =
                    -CUDART_INF_F;
            }
        }
        return;
    }

    const int64_t q_base =
        static_cast<int64_t>(bh) *
            HEAD_DIM +
        dimension_base;

    const float8_t q_vector =
        load_bf16x8_as_float8(q + q_base);

    float8_t accumulator;
    zero_float8(accumulator);

    float running_max_log2 =
        -CUDART_INF_F;
    float running_sum = 0.0f;

    const int32_t* block_row =
        block_table +
        static_cast<int64_t>(b) *
            blocks_per_batch;

    const int64_t token_stride =
        static_cast<int64_t>(num_heads_k) *
        HEAD_DIM;
    const int64_t page_stride =
        static_cast<int64_t>(PAGE_SIZE) *
        token_stride;
    const int64_t head_offset =
        static_cast<int64_t>(kv_head) *
        HEAD_DIM;

    for (int logical_page = first_page;
         logical_page < last_page;
         ++logical_page) {
        const int token_begin =
            logical_page * PAGE_SIZE;
        const int remaining =
            seqlen - token_begin;
        const int page_tokens =
            remaining < PAGE_SIZE
                ? remaining
                : PAGE_SIZE;

        const int physical_page =
            __ldg(block_row + logical_page);

        const int64_t page_base =
            static_cast<int64_t>(physical_page) *
                page_stride +
            head_offset;

        const int pack_count =
            page_tokens * kPacksPerToken;

        for (int pack = linear_thread;
             pack < pack_count;
             pack += kThreads) {
            const int token =
                pack / kPacksPerToken;
            const int pack_in_token =
                pack - token * kPacksPerToken;
            const int dim = pack_in_token * 8;

            const int64_t global_offset =
                page_base +
                static_cast<int64_t>(token) *
                    token_stride +
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

        __syncthreads();

#pragma unroll 1
        for (int token = tz;
             token < page_tokens;
             token += BDZ) {
            const int shared_offset =
                token * HEAD_DIM +
                dimension_base;

            const float8_t k_vector =
                load_bf16x8_as_float8(
                    shared_k + shared_offset);

            float score =
                subgroup_sum_16(
                    dot_float8(q_vector, k_vector));

            score *= kScaleLog2;

            const float new_max_log2 =
                fmaxf(running_max_log2, score);

            const float alpha =
                running_sum > 0.0f
                    ? fast_exp2(
                        running_max_log2 -
                        new_max_log2)
                    : 0.0f;

            const float probability =
                fast_exp2(
                    score -
                    new_max_log2);

            scale_float8(accumulator, alpha);

            const float8_t v_vector =
                load_bf16x8_as_float8(
                    shared_v + shared_offset);

            fma_float8(
                accumulator,
                v_vector,
                probability);

            running_sum =
                fmaf(
                    alpha,
                    running_sum,
                    probability);
            running_max_log2 =
                new_max_log2;
        }

        __syncthreads();
    }

    const int state_index =
        tz * GROUP + ty;
    const int64_t merge_base =
        static_cast<int64_t>(state_index) *
            HEAD_DIM +
        dimension_base;

    const float inverse_local =
        running_sum > 0.0f
            ? 1.0f / running_sum
            : 0.0f;

    scale_float8(
        accumulator,
        inverse_local);

    store_float8(
        merge_o + merge_base,
        accumulator);

    if (tx == 0) {
        merge_lse[state_index] =
            running_sum > 0.0f
                ? running_max_log2 +
                    fast_log2(running_sum)
                : -CUDART_INF_F;
    }

    __syncthreads();

    if (tz == 0 && tx == 0) {
        float max_lse = -CUDART_INF_F;

#pragma unroll
        for (int z = 0; z < BDZ; ++z) {
            const int index =
                z * GROUP + ty;
            max_lse =
                fmaxf(
                    max_lse,
                    merge_lse[index]);
        }

        float denominator = 0.0f;

#pragma unroll
        for (int z = 0; z < BDZ; ++z) {
            const int index =
                z * GROUP + ty;
            const float lse =
                merge_lse[index];

            const float weight =
                lse == -CUDART_INF_F
                    ? 0.0f
                    : fast_exp2(
                        lse - max_lse);

            merge_weight[index] = weight;
            denominator += weight;
        }

        merge_max_lse[ty] = max_lse;
        merge_denominator[ty] = denominator;
    }

    __syncthreads();

    if (tz == 0) {
        float8_t merged_output;
        zero_float8(merged_output);

#pragma unroll
        for (int z = 0; z < BDZ; ++z) {
            const int index =
                z * GROUP + ty;
            const int64_t source_base =
                static_cast<int64_t>(index) *
                    HEAD_DIM +
                dimension_base;

            const float8_t source =
                load_float8(
                    merge_o + source_base);

            fma_float8(
                merged_output,
                source,
                merge_weight[index]);
        }

        const float denominator =
            merge_denominator[ty];
        const float inverse =
            denominator > 0.0f
                ? 1.0f / denominator
                : 0.0f;

        scale_float8(
            merged_output,
            inverse);

        store_float8(
            partial_o +
                partial_base +
                dimension_base,
            merged_output);

        if (tx == 0) {
            partial_lse[partial_index] =
                denominator > 0.0f
                    ? merge_max_lse[ty] +
                        fast_log2(denominator)
                    : -CUDART_INF_F;
        }
    }
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
    const int first_index =
        bh * splits;

    if (tid == 0) {
        float max_lse =
            -CUDART_INF_F;

        for (int split = 0;
             split < splits;
             ++split) {
            max_lse =
                fmaxf(
                    max_lse,
                    partial_lse[
                        first_index + split]);
        }

        float denominator = 0.0f;

        for (int split = 0;
             split < splits;
             ++split) {
            const float lse =
                partial_lse[
                    first_index + split];

            const float weight =
                lse == -CUDART_INF_F
                    ? 0.0f
                    : fast_exp2(
                        lse - max_lse);

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

    for (int split = 0;
         split < splits;
         ++split) {
        const int partial_index =
            first_index + split;

        accumulator =
            fmaf(
                weights[split],
                partial_o[
                    static_cast<int64_t>(
                        partial_index) *
                        HEAD_DIM +
                    tid],
                accumulator);
    }

    output[
        static_cast<int64_t>(bh) *
            HEAD_DIM +
        tid] =
        __float2bfloat16(
            accumulator *
            inverse_denominator);
}

static int choose_splits(
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
        return seqlen_k >= 8192
            ? 8
            : 4;
    }

    if (batch_size <= 32) {
        return 4;
    }

    return 2;
}

static bool reserve_split_workspace(
    size_t partial_o_elems,
    size_t partial_lse_elems) {
    if (partial_o_elems > g_partial_o_elems) {
        float* new_partial_o = nullptr;

        const cudaError_t status =
            cudaMalloc(
                reinterpret_cast<void**>(
                    &new_partial_o),
                partial_o_elems *
                    sizeof(float));

        if (status != cudaSuccess ||
            new_partial_o == nullptr) {
            return false;
        }

        if (g_partial_o != nullptr) {
            cudaFree(g_partial_o);
        }

        g_partial_o = new_partial_o;
        g_partial_o_elems =
            partial_o_elems;
    }

    if (partial_lse_elems >
        g_partial_lse_elems) {
        float* new_partial_lse = nullptr;

        const cudaError_t status =
            cudaMalloc(
                reinterpret_cast<void**>(
                    &new_partial_lse),
                partial_lse_elems *
                    sizeof(float));

        if (status != cudaSuccess ||
            new_partial_lse == nullptr) {
            return false;
        }

        if (g_partial_lse != nullptr) {
            cudaFree(g_partial_lse);
        }

        g_partial_lse = new_partial_lse;
        g_partial_lse_elems =
            partial_lse_elems;
    }

    return
        g_partial_o != nullptr &&
        g_partial_lse != nullptr;
}

static void launch_fallback(
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
    paged_decode_fallback_kernel
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

template <int GROUP>
static void launch_direct_gqa(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache_paged,
    const __nv_bfloat16* v_cache_paged,
    __nv_bfloat16* output,
    const int32_t* cache_seqlens,
    const int32_t* block_table,
    int blocks_per_batch,
    int total_kv_heads,
    int num_heads,
    int num_heads_k) {
    const dim3 block(16, GROUP, 1);

    paged_decode_gqa_direct_kernel<GROUP>
        <<<total_kv_heads, block>>>(
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
        static_cast<int>(
            num_blocks / batch_size);

    if (blocks_per_batch <= 0) {
        return;
    }

    const int h =
        static_cast<int>(num_heads);
    const int hk =
        static_cast<int>(num_heads_k);
    const int total_heads =
        static_cast<int>(
            batch_size * num_heads);
    const int total_kv_heads =
        static_cast<int>(
            batch_size * num_heads_k);
    const int group_size =
        static_cast<int>(
            num_heads / num_heads_k);

    const bool contest_gqa =
        seqlen_q == 1 &&
        causal == 0 &&
        num_heads == 32 &&
        ((num_heads_k == 4 &&
          group_size == 8) ||
         (num_heads_k == 8 &&
          group_size == 4));

    const uintptr_t packed_alignment =
        reinterpret_cast<uintptr_t>(q) |
        reinterpret_cast<uintptr_t>(
            k_cache_paged) |
        reinterpret_cast<uintptr_t>(
            v_cache_paged);

    const bool packed_aligned =
        (packed_alignment & 15u) == 0u;

    if (!contest_gqa ||
        !packed_aligned) {
        launch_fallback(
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

    if (seqlen_k <= 2048) {
        if (group_size == 4) {
            launch_direct_gqa<4>(
                q,
                k_cache_paged,
                v_cache_paged,
                output,
                cache_seqlens,
                block_table,
                blocks_per_batch,
                total_kv_heads,
                h,
                hk);
        } else {
            launch_direct_gqa<8>(
                q,
                k_cache_paged,
                v_cache_paged,
                output,
                cache_seqlens,
                block_table,
                blocks_per_batch,
                total_kv_heads,
                h,
                hk);
        }
        return;
    }

    int splits =
        choose_splits(
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
        if (group_size == 4) {
            launch_direct_gqa<4>(
                q,
                k_cache_paged,
                v_cache_paged,
                output,
                cache_seqlens,
                block_table,
                blocks_per_batch,
                total_kv_heads,
                h,
                hk);
        } else {
            launch_direct_gqa<8>(
                q,
                k_cache_paged,
                v_cache_paged,
                output,
                cache_seqlens,
                block_table,
                blocks_per_batch,
                total_kv_heads,
                h,
                hk);
        }
        return;
    }

    const size_t partial_lse_elems =
        static_cast<size_t>(total_heads) *
        static_cast<size_t>(splits);

    const size_t partial_o_elems =
        partial_lse_elems * HEAD_DIM;

    if (!reserve_split_workspace(
            partial_o_elems,
            partial_lse_elems)) {
        if (group_size == 4) {
            launch_direct_gqa<4>(
                q,
                k_cache_paged,
                v_cache_paged,
                output,
                cache_seqlens,
                block_table,
                blocks_per_batch,
                total_kv_heads,
                h,
                hk);
        } else {
            launch_direct_gqa<8>(
                q,
                k_cache_paged,
                v_cache_paged,
                output,
                cache_seqlens,
                block_table,
                blocks_per_batch,
                total_kv_heads,
                h,
                hk);
        }
        return;
    }

    const dim3 grid(
        total_kv_heads,
        splits,
        1);

    if (group_size == 4) {
        const dim3 block(16, 4, 4);

        paged_decode_gqa_partition_kernel<4, 4>
            <<<grid, block>>>(
                q,
                k_cache_paged,
                v_cache_paged,
                g_partial_o,
                g_partial_lse,
                cache_seqlens,
                block_table,
                blocks_per_batch,
                splits,
                h,
                hk);
    } else {
        const dim3 block(16, 8, 2);

        paged_decode_gqa_partition_kernel<8, 2>
            <<<grid, block>>>(
                q,
                k_cache_paged,
                v_cache_paged,
                g_partial_o,
                g_partial_lse,
                cache_seqlens,
                block_table,
                blocks_per_batch,
                splits,
                h,
                hk);
    }

    combine_split_lse_kernel
        <<<total_heads, HEAD_DIM>>>(
            g_partial_o,
            g_partial_lse,
            output,
            splits);
}

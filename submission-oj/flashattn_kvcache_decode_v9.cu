/*
 * FlashAttention Paged GQA Decode — v9 (CP-async BSM + Split-KV)
 *
 * Key improvements over v8:
 *   1. CP-async BSM loads: async 64-bit BSM copy for K/V, overlapping memory & compute
 *   2. Split-KV: BDZ=2 for GROUP_SIZE=4, partitioning KV tokens across BDZ groups
 *   3. Warp merge: sync_state() merge of partial softmax states after the page loop
 *
 * Block layout:
 *   gridDim.x = batch_size
 *   gridDim.y = num_kv_heads
 *   blockDim.x = WARP_SIZE = 32
 *   blockDim.y = group_size = num_heads / num_kv_heads
 *   blockDim.z = BDZ (= 1 for GROUP_SIZE=8, = 2 for GROUP_SIZE=4)
 *
 * Each warp (blockDim.x threads) computes QK for one Q head.
 * All Q heads in a block share K/V in shared memory.
 * BDZ > 1 means KV sequence is split into BDZ parallel partitions per block.
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
#define WARP_SIZE 32
#define VEC_SIZE (HEAD_DIM / WARP_SIZE)  // 4
#define NUM_STAGES 2

// Tokens per stage = BDZ * GROUP_SIZE. Each tz warps compute on GROUP_SIZE tokens.
// Template params: GROUP_SIZE, BDZ determine smem and compute granule.

// ─── BSM (Barrier Shared Memory) async load helpers ──────────────────────

__device__ __forceinline__ void bsm_load_kv(
    __nv_bfloat16* k_smem, const __nv_bfloat16* k_gmem,
    __nv_bfloat16* v_smem, const __nv_bfloat16* v_gmem,
    bool pred)
{
    // 64-bit BSM = 4 × bf16 = exactly VEC_SIZE elements per thread
    typedef __NATIVE_VECTOR__(2, int) VecType;
    auto* k_dst = (VecType*)k_smem;
    auto* k_src = (VecType*)k_gmem;
    auto* v_dst = (VecType*)v_smem;
    auto* v_src = (VecType*)v_gmem;
    __builtin_mxc_ldg_b64_bsm_predicator(k_dst, k_src, 0, true, true, false, true, pred, 1, MACA_ICMP_EQ);
    __builtin_mxc_ldg_b64_bsm_predicator(v_dst, v_src, 0, true, true, false, true, pred, 1, MACA_ICMP_EQ);
}

// Wait until at most N BSM ops are still in-flight per thread.
// N=0: all pending BSM done (used after preload).
// N=2: tolerate up to 2 in-flight ops = next chunk's K+V loads.
template <int N>
__device__ __forceinline__ void bsm_wait_n() {
    __builtin_mxc_arrive_gvmcnt(N);
    __builtin_mxc_barrier_ex(4);
}

// ─── Online Softmax State ────────────────────────────────────────────────

struct SoftmaxState {
    float o[VEC_SIZE];
    float m;
    float d;
};

__device__ __forceinline__ void state_init(SoftmaxState& st) {
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) st.o[i] = 0.f;
    st.m = -CUDART_INF_F;
    st.d = 0.f;
}

__device__ __forceinline__ void state_normalize(SoftmaxState& st) {
    float inv_d = 1.f / st.d;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) st.o[i] *= inv_d;
}

// Merge src into dst using online softmax merge:
//   dst = dst * exp2(dst.m - new_m) + src * exp2(src.m - new_m)
__device__ __forceinline__ void state_merge(SoftmaxState& dst, const float* src_o,
                                             float src_m, float src_d) {
    float m_new = fmaxf(dst.m, src_m);
    float dst_scale = exp2f(dst.m - m_new);
    float src_scale = exp2f(src_m - m_new);
    dst.d = dst.d * dst_scale + src_d * src_scale;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        dst.o[i] = dst.o[i] * dst_scale + src_o[i] * src_scale;
    }
    dst.m = m_new;
}

// ─── Main Kernel ─────────────────────────────────────────────────────────

template <int GROUP_SIZE, int BDZ>
__global__ void paged_decode_gqa_kernel_v9(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int blocks_per_batch,
    int num_kv_heads)
{
    const int b = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int qo_head = kv_head * GROUP_SIZE + threadIdx.y;

    const int seqlen = cache_seqlens[b];
    if (seqlen <= 0) return;
    const int valid_pages = (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;
    const int total_tokens = valid_pages * PAGE_SIZE;

    const int lane = threadIdx.x;  // 0..31
    const int ty   = threadIdx.y;  // 0..GROUP_SIZE-1
    const int tz   = threadIdx.z;  // 0..BDZ-1

    // ── Load Q vector ─────────────────────────────────────────────────
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

    // ── Shared memory ──────────────────────────────────────────────────
    // Layout: k_smem[NUM_STAGES][BDZ][GROUP_SIZE][HEAD_DIM] + same for V
    // Tokens per stage = BDZ * GROUP_SIZE
    // Thread (tx, ty, tz) loads token tz*GROUP_SIZE+ty of the current chunk
    constexpr int TOK_PER_STAGE = BDZ * GROUP_SIZE;
    constexpr int STAGE_SIZE_KV = BDZ * GROUP_SIZE * HEAD_DIM;  // bf16 elements per stage

    extern __shared__ uint8_t smem_raw[];
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* v_smem = k_smem + NUM_STAGES * STAGE_SIZE_KV;

    SoftmaxState st;
    state_init(st);

    // ── Stride helpers ─────────────────────────────────────────────────
    const int64_t page_stride = static_cast<int64_t>(PAGE_SIZE) * num_kv_heads * HEAD_DIM;
    const int64_t token_stride = static_cast<int64_t>(num_kv_heads) * HEAD_DIM;
    const int64_t block_row_base = static_cast<int64_t>(b) * blocks_per_batch;

    const float sm_scale = 1.f / sqrtf(static_cast<float>(HEAD_DIM));
    const float sm_scale_log2 = sm_scale * 1.4426950408889634f; // log2(e)

    // Total token chunks across all pages
    const int total_chunks = (total_tokens + TOK_PER_STAGE - 1) / TOK_PER_STAGE;

    // ── Preload first NUM_STAGES chunks via BSM ───────────────────────
    int stage_idx = 0;
    #pragma unroll
    for (int s = 0; s < NUM_STAGES && s < total_chunks; ++s) {
        int tok_base = s * TOK_PER_STAGE;
        // Thread (tx, ty, tz) loads token: tok_base + tz*GROUP_SIZE + ty
        int tk = tok_base + tz * GROUP_SIZE + ty;
        bool valid_tok = (tk < seqlen);

        if (valid_tok) {
            int page = tk / PAGE_SIZE;
            int tok_in_page = tk % PAGE_SIZE;
            int physical_block = block_table[block_row_base + page];
            int64_t gmem_base = static_cast<int64_t>(physical_block) * page_stride
                               + kv_head * HEAD_DIM
                               + static_cast<int64_t>(tok_in_page) * token_stride;

            int smem_k_idx = (stage_idx * BDZ + tz) * GROUP_SIZE * HEAD_DIM
                           + ty * HEAD_DIM + lane * VEC_SIZE;
            int smem_v_idx = smem_k_idx;  // same offset structure for V

            bsm_load_kv(k_smem + smem_k_idx, k_cache_paged + gmem_base + lane * VEC_SIZE,
                        v_smem + smem_v_idx, v_cache_paged + gmem_base + lane * VEC_SIZE,
                        true);
        }
        stage_idx = (stage_idx + 1) % NUM_STAGES;
    }
    bsm_wait_n<0>();  // Wait for all preload BSM ops
    __syncthreads();

    // ── Main token-chunk loop (compute + async pipeline) ──────────────
    for (int chunk = 0; chunk < total_chunks; ++chunk) {
        // Wait until ≤2 BSM ops/thread remain (the current compute stage is ready,
        // while the other stage's loads from the previous iteration may still be in-flight)
        bsm_wait_n<2>();

        const int compute_stage = chunk % NUM_STAGES;
        const int tok_base = chunk * TOK_PER_STAGE;

        // ── Compute QK + softmax + P×V for this chunk ──────────────
        // Each tz processes GROUP_SIZE tokens: tz*GROUP_SIZE .. tz*GROUP_SIZE+GROUP_SIZE-1
        #pragma unroll
        for (int j = 0; j < GROUP_SIZE; ++j) {
            int tk = tok_base + tz * GROUP_SIZE + j;
            if (tk >= seqlen) continue;

            int smem_kv_base = (compute_stage * BDZ + tz) * GROUP_SIZE * HEAD_DIM
                             + j * HEAD_DIM;

            // QK dot product with warp-shuffle reduction
            float dot = 0.f;
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                dot += q_vec[i] * __bfloat162float(k_smem[smem_kv_base + lane * VEC_SIZE + i]);
            }
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, offset);
            }

            // Online softmax (base-2)
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

            // P×V accumulation
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                st.o[i] += p * __bfloat162float(v_smem[smem_kv_base + lane * VEC_SIZE + i]);
            }
        }
        __syncthreads();

        // ── Issue BSM loads for chunk + NUM_STAGES (next to consume this stage) ──
        int next_chunk = chunk + NUM_STAGES;
        if (next_chunk < total_chunks) {
            int next_stage = next_chunk % NUM_STAGES;
            int next_tok_base = next_chunk * TOK_PER_STAGE;

            int tk = next_tok_base + tz * GROUP_SIZE + ty;
            bool valid_tok = (tk < seqlen);

            if (valid_tok) {
                int page = tk / PAGE_SIZE;
                int tok_in_page = tk % PAGE_SIZE;
                int physical_block = block_table[block_row_base + page];
                int64_t gmem_base = static_cast<int64_t>(physical_block) * page_stride
                                   + kv_head * HEAD_DIM
                                   + static_cast<int64_t>(tok_in_page) * token_stride;

                int smem_k_idx = (next_stage * BDZ + tz) * GROUP_SIZE * HEAD_DIM
                               + ty * HEAD_DIM + lane * VEC_SIZE;
                int smem_v_idx = smem_k_idx;

                bsm_load_kv(k_smem + smem_k_idx, k_cache_paged + gmem_base + lane * VEC_SIZE,
                            v_smem + smem_v_idx, v_cache_paged + gmem_base + lane * VEC_SIZE,
                            true);
            }
        }
        __syncthreads();
    }

    // ── Split-KV merge: if BDZ > 1, merge partial states across tz ──
    if (BDZ > 1) {
        // Reuse K smem for partial o storage (as float), V smem for m/d
        float* merge_o  = reinterpret_cast<float*>(k_smem);
        float* merge_md = reinterpret_cast<float*>(v_smem);

        // Store partial state: each (tz, ty) warp stores its o[VEC_SIZE], m, d
        int warp_idx = tz * GROUP_SIZE + ty;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            merge_o[warp_idx * HEAD_DIM + lane * VEC_SIZE + i] = st.o[i];
        }
        merge_md[warp_idx * 2]     = st.m;
        merge_md[warp_idx * 2 + 1] = st.d;
        __syncthreads();

        // Each ty merges the BDZ partial results from different tz
        // Start with tz=0 as base, merge in tz=1, tz=2, ...
        state_init(st);
        {
            int widx0 = 0 * GROUP_SIZE + ty;
            st.m = merge_md[widx0 * 2];
            st.d = merge_md[widx0 * 2 + 1];
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                st.o[i] = merge_o[widx0 * HEAD_DIM + lane * VEC_SIZE + i];
            }
        }

        #pragma unroll
        for (int z = 1; z < BDZ; ++z) {
            int widx_z = z * GROUP_SIZE + ty;
            float mz = merge_md[widx_z * 2];
            float dz = merge_md[widx_z * 2 + 1];
            float* oz = merge_o + widx_z * HEAD_DIM + lane * VEC_SIZE;
            state_merge(st, oz, mz, dz);
        }
        __syncthreads();
    }

    // ── Normalize and write output ─────────────────────────────────────
    state_normalize(st);
    if (qo_head < 32) {
        const int64_t out_base = (static_cast<int64_t>(b) * 32 + qo_head) * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            output[out_base + lane * VEC_SIZE + i] = __float2bfloat16(st.o[i]);
        }
    }
}

// ─── Host dispatch ────────────────────────────────────────────────────────

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

    // BDZ: split-KV factor — partition KV tokens across BDZ groups
    // For GROUP_SIZE=8 (NHK=4): BDZ=1, no split (256 threads/block)
    // For GROUP_SIZE=4 (NHK=8): BDZ=2, split-KV (256 threads/block)
    const int bdz = (group_size == 4) ? 2 : 1;

    dim3 grid(static_cast<unsigned int>(batch_size),
              static_cast<unsigned int>(num_heads_k));
    dim3 block(WARP_SIZE, static_cast<unsigned int>(group_size),
               static_cast<unsigned int>(bdz));

    // smem: NUM_STAGES × (K + V) × BDZ × GROUP × HEAD_DIM × sizeof(bf16)
    // K total: 2 * BDZ * GROUP * 128 = 2*BDZ*GROUP*HEAD_DIM bf16 elements
    // V total: same
    // Total: 4 * BDZ * GROUP * HEAD_DIM * 2 bytes
    // BDZ=2,GROUP=4: 4*2*4*128*2 = 8192 bytes
    // BDZ=1,GROUP=8: 4*1*8*128*2 = 8192 bytes
    const int bdz_v = (group_size == 4) ? 2 : 1;
    size_t smem_bytes = 4 * bdz_v * group_size * HEAD_DIM * sizeof(__nv_bfloat16);

    if (group_size == 8) {
        paged_decode_gqa_kernel_v9<8, 1><<<grid, block, smem_bytes>>>(
            q, k_cache_paged, v_cache_paged, output,
            cache_seqlens, block_table, blocks_per_batch, nkh);
    } else {
        paged_decode_gqa_kernel_v9<4, 2><<<grid, block, smem_bytes>>>(
            q, k_cache_paged, v_cache_paged, output,
            cache_seqlens, block_table, blocks_per_batch, nkh);
    }
}

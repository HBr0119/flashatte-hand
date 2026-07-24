/*
 * Minimal test to verify MACA MMA fragment layout.
 * Tests: A(Q) col-major, B(K) row-major, C row-major with 64 threads.
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>

typedef _Float16 maca_half4 __attribute__((__vector_size__(8)));
typedef float maca_float4 __attribute__((__vector_size__(16)));

#define HEAD_DIM 128
#define MMA_TILE 16
#define MMA_K_TILES (HEAD_DIM / MMA_TILE)
#define MMA_THREADS 64

union alignas(8) mma_bf16_fragment_t {
    maca_half4 vector;
    __nv_bfloat16 element[4];
};

union alignas(16) mma_float_fragment_t {
    maca_float4 vector;
    float element[4];
};

__global__ __launch_bounds__(MMA_THREADS)
void test_mma_kernel(__nv_bfloat16* output, int group) {
    const int tid = threadIdx.x;
    const int fragment_linear_base = tid * 4;

    // Create known Q and K values in fragments
    mma_bf16_fragment_t q_fragments[MMA_K_TILES];
    mma_float_fragment_t score_fragment;

    // Initialize: set each (query_row, k_dim) to a known pattern
    // Q[query_row][kt*16 + k_in_tile] = query_row * 1000 + kt * 100 + k_in_tile
    // K[token][kt*16 + k_in_tile] = token * 1000 + kt * 100 + k_in_tile
    // So dot = sum over all dims of Q*K

#pragma unroll
    for (int kt = 0; kt < MMA_K_TILES; ++kt) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int linear = fragment_linear_base + i;
            const int query_row = linear & 15;   // row (col-major Q: row + k*16)
            const int k_in_tile = linear >> 4;

            // Q: set known value
            float q_val = query_row * 1000.0f + kt * 100.0f + k_in_tile;
            q_fragments[kt].element[i] = __float2bfloat16(q_val);
        }
    }

    // Initialize accumulator
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        score_fragment.element[i] = 0.0f;
    }

    // MMA computation
#pragma unroll
    for (int kt = 0; kt < MMA_K_TILES; ++kt) {
        mma_bf16_fragment_t k_fragment;

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int linear = fragment_linear_base + i;
            const int k_in_tile = linear >> 4;   // row-major K: k*16 + token
            const int token = linear & 15;

            // K: set known value
            float k_val = token * 1000.0f + kt * 100.0f + k_in_tile;
            k_fragment.element[i] = __float2bfloat16(k_val);
        }

        score_fragment.vector = __builtin_mxc_mma_16x16x16bf16(
            q_fragments[kt].vector,
            k_fragment.vector,
            score_fragment.vector);
    }

    // Write C fragment to global memory
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int linear = fragment_linear_base + i;
        const int query_row = linear >> 4;   // row-major C: row*16 + col
        const int token = linear & 15;

        if (query_row < group && linear < 256) {
            output[linear] = __float2bfloat16(score_fragment.element[i]);
        }
    }
}

int main() {
    constexpr int group = 4; // test with GROUP=4
    __nv_bfloat16* d_output;
    float h_output[256];  // full 16x16 output

    cudaMalloc(&d_output, 256 * sizeof(__nv_bfloat16));
    test_mma_kernel<<<1, MMA_THREADS>>>(d_output, group);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, 256 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Compute expected: dot[row][col] = sum_{kt=0..7, k=0..15} Q[row][kt*16+k] * K[col][kt*16+k]
    // Q[row][kt*16+k] = row*1000 + kt*100 + k
    // K[col][kt*16+k] = col*1000 + kt*100 + k
    // dot[row][col] = sum_{kt,k} (row*1000 + kt*100 + k) * (col*1000 + kt*100 + k)

    printf("MMA output matrix (16x16, row-major C layout):\n");
    printf("Row Token ->\n");
    bool all_ok = true;
    for (int row = 0; row < 4; ++row) {  // only first GROUP rows
        printf("Q%d: ", row);
        for (int token = 0; token < 16; ++token) {
            int idx = row * 16 + token;
            float mma_val = h_output[idx];

            // Expected: compute scalar dot product
            float expected = 0.0f;
            for (int kt = 0; kt < 8; ++kt) {
                for (int k = 0; k < 16; ++k) {
                    float qv = row * 1000.0f + kt * 100.0f + k;
                    float kv = token * 1000.0f + kt * 100.0f + k;
                    expected += qv * kv;
                }
            }

            bool ok = (fabsf(mma_val - expected) / fmaxf(fabsf(expected), 1.0f)) < 0.01f;
            if (!ok) all_ok = false;
            printf("%s%.0f%s ", ok ? "" : "[", mma_val, ok ? "" : "]");
        }
        printf("\n");
    }
    printf("\n%s\n", all_ok ? "ALL CORRECT!" : "ERRORS FOUND!");

    cudaFree(d_output);
    return all_ok ? 0 : 1;
}

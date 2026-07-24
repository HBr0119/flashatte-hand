"""
Minimal MMA fragment layout test.
Tests __builtin_mxc_mma_16x16x16bf16 with 64 threads.
"""
import torch
import torch.utils.cpp_extension

MMA_TEST_SRC = """
#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef _Float16 maca_half4 __attribute__((__vector_size__(8)));
typedef float maca_float4 __attribute__((__vector_size__(16)));

#define MMA_TILE 16
#define MMA_K_TILES 1
#define MMA_THREADS 64

union alignas(8) mma_bf16_fragment_t {
    maca_half4 vector;
    __nv_bfloat16 element[4];
};

union alignas(16) mma_float_fragment_t {
    maca_float4 vector;
    float element[4];
};

extern "C" __global__ __launch_bounds__(MMA_THREADS)
void test_mma(__nv_bfloat16* output) {
    const int tid = threadIdx.x;
    const int linear_base = tid * 4;

    mma_bf16_fragment_t q_frag;
    mma_bf16_fragment_t k_frag;
    mma_float_fragment_t score;

    for (int i = 0; i < 4; ++i) score.element[i] = 0.0f;

    // Q: col-major, row + k*16.  Q[row][k] = row*100 + k
    for (int i = 0; i < 4; ++i) {
        int linear = linear_base + i;
        int row = linear & 15;
        int k = linear >> 4;
        q_frag.element[i] = __float2bfloat16((float)(row * 100 + k));
    }

    // K: row-major, k*16 + token.  K[token][k] = token*100 + k
    for (int i = 0; i < 4; ++i) {
        int linear = linear_base + i;
        int k = linear >> 4;
        int token = linear & 15;
        k_frag.element[i] = __float2bfloat16((float)(token * 100 + k));
    }

    score.vector = __builtin_mxc_mma_16x16x16bf16(
        q_frag.vector, k_frag.vector, score.vector);

    // Write output (row-major: row*16 + col)
    for (int i = 0; i < 4; ++i) {
        int linear = linear_base + i;
        int row = linear >> 4;
        int col = linear & 15;
        if (linear < 256) {
            output[linear] = __float2bfloat16(score.element[i]);
        }
    }
}
"""

def main():
    module = torch.utils.cpp_extension.load_inline(
        name='test_mma_fragment',
        cpp_sources=[],
        cuda_sources=[MMA_TEST_SRC],
        functions=['test_mma'],
        verbose=False,
    )

    output = torch.zeros(256, dtype=torch.bfloat16, device='cuda')
    module.test_mma(output)
    h_out = output.cpu().float().numpy()

    # Check row-major interpretation: C[row][col] = h_out[row*16+col]
    # Expected: sum_{k=0}^{15} (row*100+k) * (col*100+k)
    print("=== Row-major C (row*16+col) ===")
    row_major_ok = True
    for row in range(4):  # only first 4 rows
        for col in range(4):  # only first 4 cols
            got = h_out[row * 16 + col]
            exp_val = sum((row * 100.0 + k) * (col * 100.0 + k) for k in range(16))
            ok = abs(got - exp_val) / max(abs(exp_val), 1.0) < 0.01
            if not ok:
                row_major_ok = False
            print(f"  C[{row}][{col}]: got={got:.1f} exp={exp_val:.1f} {'OK' if ok else 'FAIL'}")

    # Check col-major interpretation: C[row][col] = h_out[col*16+row]
    print("\n=== Col-major C (col*16+row) ===")
    col_major_ok = True
    for row in range(4):
        for col in range(4):
            got = h_out[col * 16 + row]
            exp_val = sum((row * 100.0 + k) * (col * 100.0 + k) for k in range(16))
            ok = abs(got - exp_val) / max(abs(exp_val), 1.0) < 0.01
            if not ok:
                col_major_ok = False
            print(f"  C[{row}][{col}]: got={got:.1f} exp={exp_val:.1f} {'OK' if ok else 'FAIL'}")

    # Check 8-col version (warp-local: 16 rows x 8 cols for 32 threads)
    print("\n=== 16x8 warp-local C (row*8+col) ===")
    warp8_ok = True
    for row in range(4):
        for col in range(4):
            got = h_out[row * 8 + col]
            exp_val = sum((row * 100.0 + k) * (col * 100.0 + k) for k in range(16))
            ok = abs(got - exp_val) / max(abs(exp_val), 1.0) < 0.01
            if not ok:
                warp8_ok = False
            print(f"  C[{row}][{col}]: got={got:.1f} exp={exp_val:.1f} {'OK' if ok else 'FAIL'}")

    print(f"\n=== Summary ===")
    print(f"  Row-major 16x16: {'PASS' if row_major_ok else 'FAIL'}")
    print(f"  Col-major 16x16: {'PASS' if col_major_ok else 'FAIL'}")
    print(f"  Warp-local 16x8: {'PASS' if warp8_ok else 'FAIL'}")

if __name__ == '__main__':
    main()

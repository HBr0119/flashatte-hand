#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

typedef _Float16 maca_half4 __attribute__((__vector_size__(8)));
typedef float maca_float4 __attribute__((__vector_size__(16)));

#define MMA_TILE 16
#define MMA_THREADS 64

union alignas(8) mma_bf16_fragment_t {
    maca_half4 vector;
    __nv_bfloat16 element[4];
};

union alignas(16) mma_float_fragment_t {
    maca_float4 vector;
    float element[4];
};

// Test: Q all ones (like initialize_q_fragments would do for a specific row set to 1)
// K: only one non-zero at thread t, elem e
// Use the ORIGINAL fragment layout from our kernel
__global__ void test_simple(__nv_bfloat16* output, int k_tid, int k_elem) {
    const int tid = threadIdx.x;
    const int linear_base = tid * 4;

    mma_bf16_fragment_t q_frag;
    mma_bf16_fragment_t k_frag;
    mma_float_fragment_t score;

    for (int i = 0; i < 4; ++i) score.element[i] = 0.0f;

    // Original Q loading: col-major (row + k*16). Set ALL Q to 1.0
    for (int i = 0; i < 4; ++i) {
        q_frag.element[i] = __float2bfloat16(1.0f);
    }

    // K: set only at (k_tid, k_elem) = 1. Use row-major (k*16+token)
    for (int i = 0; i < 4; ++i) {
        int linear = linear_base + i;
        int k = linear >> 4;
        int token = linear & 15;
        // Only set K at specific thread/elem
        if (tid == k_tid && i == k_elem) {
            k_frag.element[i] = __float2bfloat16(1.0f);
        } else {
            k_frag.element[i] = __float2bfloat16(0.0f);
        }
    }

    score.vector = __builtin_mxc_mma_16x16x16bf16(
        q_frag.vector, k_frag.vector, score.vector);

    for (int i = 0; i < 4; ++i) {
        int linear = linear_base + i;
        if (linear < 256) {
            output[linear] = __float2bfloat16(score.element[i]);
        }
    }
}

torch::Tensor t0_e0() {
    auto o = torch::empty({256}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    test_simple<<<1, MMA_THREADS>>>(reinterpret_cast<__nv_bfloat16*>(o.data_ptr()), 0, 0);
    return o;
}
torch::Tensor t1_e0() {
    auto o = torch::empty({256}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    test_simple<<<1, MMA_THREADS>>>(reinterpret_cast<__nv_bfloat16*>(o.data_ptr()), 1, 0);
    return o;
}
torch::Tensor t4_e0() {
    auto o = torch::empty({256}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    test_simple<<<1, MMA_THREADS>>>(reinterpret_cast<__nv_bfloat16*>(o.data_ptr()), 4, 0);
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("t0_e0", &t0_e0, "Q=all1, K at tid=0,elem=0 = 1");
    m.def("t1_e0", &t1_e0, "Q=all1, K at tid=1,elem=0 = 1");
    m.def("t4_e0", &t4_e0, "Q=all1, K at tid=4,elem=0 = 1");
}

# v2 — Working Baseline (MMA QK + Scalar P×V)

- Source: `flashattn_kvcache_decode_mma_v2.cu`
- Status: ✅ 14/14 正确性通过，score 0.63
- 结构:
  - 64-thread MMA QK (`__builtin_mxc_mma_16x16x16bf16`)：2 warp，每个 warp 覆盖 8 个 k-dim
  - Scalar online softmax（tid < kGroup 串行处理）
  - Scalar P×V（逐 token 累加 `output_acc[d] = fmaf(prob, v, acc)`）
  - CP-async page loading (flashinfer::cp_async::load_128b_bsm)
  - 多 page 循环，page interleave 合并
  - per-thread register accumulator `output_acc[kDimsPerThread]`

- Profiling:
  | 阶段 | Cycles | % |
  |------|--------|---|
  | Softmax | 83,039 | 38.5% |
  | P×V scalar | 65,991 | 30.6% |
  | MMA dot(Q,K) | 44,369 | 20.5% |
  | CP-async load | 22,534 | 10.4% |

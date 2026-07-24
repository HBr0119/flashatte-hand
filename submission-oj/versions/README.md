# FlashAttention GQA Kernel 版本演进

## 版本一览

| 版本 | 目录 | QK MMA | P×V | Softmax | 14/14 | Score | 备注 |
|------|------|--------|-----|---------|-------|-------|------|
| v2 | `v2_baseline/` | ✅ MMA | 标量 | 串行(1thd) | ✅ | 0.63 | 工作基线 |
| v3 | `v3_mma_pv_cp_async/` | ✅ MMA | ✅ MMA | 串行 | ✅ | 0.62 | CP-async |
| v4 | `v4_mma_pv_fixed/` | ✅ MMA | ✅ MMA | 串行 | ✅ | 0.63 | Sync copy |
| v5 | `v5_warp_softmax/` | ✅ MMA | 标量 | **并行(warp)** | ✅ | **0.65** | 当前最优 |

## Profiling 数据 (v2 baseline)
| 阶段 | Cycles | % |
|------|--------|---|
| Softmax | 83,039 | **38.5%** ← v5 优化目标 |
| P×V scalar | 65,991 | 30.6% |
| MMA dot(Q,K) | 44,369 | 20.5% |
| CP-async load | 22,534 | 10.4% |

## 优化结论
- **P×V MMA**: 对 16 token 页面无优势（MMA 开销 > 标量）
- **Softmax warp 并行化**: +3.2% score 改进，达到 0.65
- **下一步**: CP-async double buffering + fast math

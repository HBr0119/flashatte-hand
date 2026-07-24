# v4 — MMA P×V with Sync Copies (v8)

- Source: `flashattn_kvcache_decode_mma_v4.cu`
- Forked from: v2（直接 copy）
- Status: ✅ 14/14 正确性通过，score 0.63

## 变更 vs v2
1. `shared_output_acc[kGroup][HEAD_DIM]` 替代 per-thread `output_acc`
2. MMA P×V 替换标量 P×V（单 `pv_acc` accumulator + 每次清零）
3. CP-async → sync `int4` copy（诊断用）
4. V-zero 清零每个 page 的无效 token 位置
5. 移除 `output_acc[kDimsPerThread]` 寄存器数组

## NaN 根因与修复
同 v3。

## 性能
| 指标 | v2 (baseline) | v4 |
|------|-------------|-----|
| Score | 0.63 | 0.63 |
| 14/14 | ✅ | ✅ |
| kv=141 (GQA) | ~1x | 0.33x |

**结论**: 与 v3 结论相同。MMA P×V 在短序列 GQA 上无优势。
Softmax 并行化（38.5%瓶颈）才是正确的优化方向。

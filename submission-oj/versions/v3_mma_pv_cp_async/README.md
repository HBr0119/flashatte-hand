# v3 — MMA P×V with CP-async (v7)

- Source: `flashattn_kvcache_decode_mma_v3.cu`
- Forked from: v2
- Status: ✅ 14/14 正确性通过，score 0.62

## 变更 vs v2
1. `shared_output_acc[kGroup][HEAD_DIM]` 替代 per-thread `output_acc`
2. MMA P×V 替换标量 P×V（单 `pv_acc` accumulator + 每次清零）
3. 保留 CP-async page loading
4. 输出直接从 `shared_output_acc` 写入 bf16 output
5. V-zero 清零每个 page 的无效 token 位置

## NaN 根因与修复
- **根因**: 两个独立问题叠加：
  1. `shared_v` 无效 token 残留垃圾 → MMA `0 × NaN = NaN`（IEEE 754）
  2. `pv_acc_all[8]` 8 accumulator 寄存器压力 → spill
- **修复 1**: 每页 K/V 加载后清零 `shared_v[t >= page_tokens]`
- **修复 2**: 单 `pv_acc` accumulator

## 性能
| 指标 | v2 (baseline) | v3 |
|------|-------------|-----|
| Score | 0.63 | 0.62 |
| 14/14 | ✅ | ✅ |
| kv=141 (GQA) | ~1x | 0.34x |
| kv=1 | 2.2x | 2.0x |

**结论**: MMA P×V 在 16 token 小页面上开销大于标量 P×V。
下一步应优化 Softmax（38.5% 瓶颈）而非 P×V。

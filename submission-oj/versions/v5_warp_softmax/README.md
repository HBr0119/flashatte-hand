# v5 — Warp-Parallel Softmax (v9)

- Source: `flashattn_kvcache_decode_mma_v5.cu`
- Forked from: v2（工作基线，标量 P×V）
- Status: ✅ 14/14 正确性通过，score 0.65

## 变更 vs v2
1. **Softmax warp 并行化**: 所有 kThreadsPerHead 线程参与（原来是单个 tid < kGroup）
   - 8-thread case（NUM_KV_HEADS=4）：每线程 2 个 token，3 步 warp shuffle reduction
   - 16-thread case（NUM_KV_HEADS=8）：每线程 1 个 token，4 步 warp shuffle reduction
2. 其余与 v2 不变（CP-async、标量 P×V、output_acc）

## 性能对比

| 版本 | Score | Short KV | Mid KV | Long KV |
|------|-------|----------|--------|---------|
| v2 (baseline) | 0.63 | — | — | — |
| v5 (warp softmax) | **0.65** | 0.98 | 0.31 | 0.32 |

**+3.2%** 总体改进。Softmax 从 38.5% → 估计 ~30%，但 expf() 指令延迟是主要瓶颈。

## 结论
Warp shuffle reduction 的 overhead 部分抵消了并行化收益。
要进一步加速 softmax 需要：
1. 使用 fast math intrinsics (`__expf()` 替代 `expf()`)
2. CP-async double buffering 隐藏内存延迟

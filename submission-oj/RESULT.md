# OJ Contest Submission

OJ 赛道最佳候选 kernel，从原始 baseline 启动，经 39 轮 agent 迭代优化。

## 当前最佳

```text
来源: iter_039 Round 5 (14/14 正确, gate-blocked by mid_kv<1.0)
分数: 0.7553
策略: non-MMA dispatch + launch tune + page-softmax specialization
```

## iter_039 逐轮结果

| Round | 正确性 | 总分 | short_kv | mid_kv | long_kv | 说明 |
|-------|--------|------|----------|--------|---------|------|
| baseline | 14/14 | 0.623 | 0.93 | 0.31 | 0.32 | iter_038 best 起点 |
| r0 (MMA) | 6/14 | 1.196 | 1.69 | null | 0.21 | agent 开 MMA, 改写 page iterator 导致长 KV 崩溃 |
| r1 (修复) | 14/14 | 0.653 | 0.99 | 0.31 | 0.32 | 修复正确性但关掉 MMA |
| r2 (MMA) | 4/14 | 1.493 | 1.49 | null | null | 重新开 MMA, 短 KV 暴涨但中长 KV 全崩 |
| r3 (MMA) | 4/14 | 1.494 | 1.49 | null | null | 同上 |
| r4 (MMA) | 4/14 | 1.510 | 1.51 | null | null | 短 KV 继续微涨 |
| r5 (修复) | 14/14 | 0.755 | 1.18 | 0.29 | 0.34 | 14/14 全过! 超 iter_024, 被 mid_kv gate 拦下 |

## MMA 实验总结 (iter_028-039)

MACA C500 xcore MMA (`__builtin_mxc_mma_16x16x16bf16`) 已验证：
- **iter_039 r5: 14/14 正确, 0.755 (当前最佳!)**
- 最高原始分数: **1.547** (iter_032 R0, 仅 4/14 正确)
- 短 KV MMA speedup 最高 2.48x (seqlen=1)
- 诊断: agent 不会做精确手术——开 MMA 就重写 page iterator
- 手工植入: 编译通过但 3 个 case 正确性失败 (warp fragment 映射)

## 分数历程

```
iter_001: 0.478   iter_011: 0.596   iter_020: 0.647   iter_030: 0.762
iter_004: 0.508   iter_016: 0.588   iter_023: 0.728   iter_031: 0.931
                   iter_017: 0.635   iter_024: 0.747   iter_032: 1.547
                                                       iter_033: 0.841
                                                       iter_034: 0.820
                                                       iter_035: 0.585
                                                       iter_036: 0.590
                                                       iter_037: 0.626
                                                       iter_038: 0.614
                                                       iter_039: 0.755 ★
```

## 文件

```text
best_flashattn_kvcache_decode.cu  — 当前最佳 (iter_039 r5, 0.7553)
flashattn_kvcache_decode_mma.cu   — 手工 MMA 植入 (编译通过, 3 case 正确性失败)
```

## 手工 MMA 集成 (2025-07-21)

尝试手工将 MACA `__builtin_mxc_mma_16x16x16bf16` 替换 scalar dot(q,k)。

### 方法

- 从 iter_024 baseline 出发，新建 `paged_decode_gqa_mma_kernel`
- 保留 **100% 原有 page iterator、CP-async K/V 加载、online softmax、scalar P×V**
- 只替换 dot(Q,K) 为 MMA

### 结果

- **11/14 正确，3 个 GQA dispatch 用例全错**（kv=141, 362, 2048）
- max_abs_err: 1.56 ~ 4.03

### 根因

通过 5 轮 GPU mini test 确定：MACA MMA 的 fragment-to-matrix-element 映射
与代码假设不同。Q 和 K 在相同 thread 上只产生 64/256 个 C 位置，
正确的 C 输出需要跨 thread A/B 配对。需要 MACA MMA fragment 布局文档
或参考实现来确定正确映射。

详见: `submission-oj/MMA_MANUAL_INTEGRATION_REPORT.md`

## 手工 MMA 集成 (2025-07-21)

手动基于 MACA SDK LLVM header (`__clang_maca_mma_functions.h`) 的 WMMA fragment 映射
信息，直接在 baseline kernel 中替换 scalar dot(q,k) 为 `__builtin_mxc_mma_16x16x16bf16`。

### 关键发现

1. **64-thread MMA fragment 映射**: Warp0 覆盖 k=0..7, Warp1 覆盖 k=8..15。
   一个 MMA call (64 threads) 覆盖完整 16 k-dimensions。
   通过 GPU 诊断测试 (`/tmp/test_mma64_diag3.py`) 精确验证。

2. **Thread 管理 bug (根因)**: `running_max`/`running_sum` 作为 per-thread 变量，
   在 softmax (tid→head) 和 P×V (tid/kThreadsPerHead→head) 两个阶段被不一致索引，
   导致不同 head 的状态互相污染。

3. **修复**: 改为 per-head shared memory 数组 `shared_running_max[kGroup]`。

### v6 Kernel 结果

```text
文件: submission-oj/flashattn_kvcache_decode_mma_v2.cu
正确性: 14/14 全部通过
  kv=141 (GQA, GROUP=8): max_abs_err=0.008 ✓ (之前 v1: 2.09)
  kv=362 (GQA, GROUP=4): max_abs_err=0.004 ✓ (之前 v1: 4.03)
  kv=2048 (GQA, GROUP=8): max_abs_err=0.008 ✓ (之前 v1: 1.56)
Score: 0.63
```

MMA 计算本身正确但 overhead 较大 (64 threads 做 MMA + 后续 shared memory 操作)，
短序列 GQA 路径比 scalar baseline 慢。

## Contest Profile

```text
num_heads = 32
num_heads_k = 4 or 8
headdim = 128
seqlen_q = 1
page_block_size = 16
num_blocks = batch_size * ceil(seqlen_k / page_block_size)
cache_seqlens varies per batch row
reference uses num_splits=0
```

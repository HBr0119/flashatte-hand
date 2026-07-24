# FlashAttention KV Cache Decode v11 优化与性能报告

## 1. 版本定位

v11 以 `flashattn_kvcache_decode_v10.cu` 为计算主干，保留 McFlashInfer 风格的 C500 BSM 搬运与双缓冲流水，同时吸收 `best_flashattn_kvcache_decode.cu` 中已经被实测证明有效的调度、Split-KV 和归并策略。

它不是简单拼接两个版本，而是按瓶颈拆分职责：

- 短、中等 KV：继续利用 v10 的 128-bit BSM、page staging 和 page 级 softmax 更新。
- 长 KV：使用 best 风格的激进 Split-KV 提高设备并行度。
- 块内与 split 间归并：由单个 leader 计算公共 LSE 权重，避免每个输出维度重复执行 `exp2`。
- 极短 KV：绕过 256-thread 双缓冲主核，使用轻量 direct GQA kernel。

源码：`flashattn_kvcache_decode_v11_best_hybrid.cu`

## 2. v11 实现的优化

### 2.1 极短 KV 专用路径

当 `seqlen_k <= 32` 时，v11 使用单独的 direct GQA kernel：

- `GROUP=4` 使用 64 threads。
- `GROUP=8` 使用 128 threads。
- 一个 block 对应一个 batch/KV head。
- 同组 query heads 共享同一页 K/V。
- 只使用单级 K/V shared memory，不建立 BDZ 层，也不执行块内 state merge。
- `seqlen=1` 直接复制 V，完全跳过 QK、softmax 和 PV。

该路径主要消除极短序列中 256 threads、双缓冲和块内归并的固定成本。

### 2.2 保留 v10 的 C500 BSM 主循环

对于非极短序列，v11 保留 v10 的核心结构：

- 每线程处理连续 8 个 BF16，即 128-bit 数据。
- 使用 `__builtin_mxc_ldg_b128_bsm*` 将 paged K/V 搬入 shared memory。
- K/V 使用两级 shared-memory buffer。
- 使用 base-2 online softmax 状态。
- QK 使用 16-lane shuffle reduction。
- 同一 KV head 对应的 GQA query heads 共享 K/V page。

这部分是 v10 在 KV=141、362、2048 上领先 best 的主要原因，因此 v11 没有替换其短序列主计算循环。

### 2.3 best 风格 Split-KV 策略

v10 在部分中长序列上切分不足。例如 batch=1、KV=8192 时只有 4 splits，总 partition block 数过少。v11 改用 best 的经验调度：

| Batch 范围 | KV 范围 | splits |
|---|---:|---:|
| `B<=1` | `KV>=8192` | 64 |
| `B<=1` | `4096<=KV<8192` | 32 |
| `B<=8` | `KV>=32768` | 32 |
| `B<=8` | `KV>=8192` | 16 |
| `B<=8` | `KV>=4096` | 8 |
| `B<=16` | `KV>=8192` | 8 |
| `B<=16` | 其他长序列 | 4 |
| `B<=32` | 长序列 | 4 |
| 更大 batch | 长序列 | 2 |

最终 splits 还会被 `MAX_SPLITS`、有效 page 数以及 `blocks_per_batch` 限制，避免创建无意义的 partition。

### 2.4 均衡 page 分区

v10 原先使用向上取整的 `pages_per_split`，尾部 split 可能更短甚至为空。v11 改为比例边界：

```cpp
first_page = valid_pages * split / num_splits;
last_page  = valid_pages * (split + 1) / num_splits;
```

这样任意两个 split 的 page 数最多相差一页。

### 2.5 块内 leader LSE 归并

v10 的 `state_merge` 由全部 16 个 x-lanes 重复执行。虽然各 lane 处理不同输出维度，但它们读取的 `m/d` 和计算的 LSE 权重完全相同，因此相同的 `exp2` 被重复执行 16 次。

v11 改为：

1. 每个 BDZ state 先生成局部归一化输出和局部 LSE。
2. `tz=0, tx=0` 为每个 query head 计算一次最大 LSE、权重和分母。
3. 权重写入 shared memory。
4. 16 个 x-lanes 只执行各自输出向量的加权累加。

该优化减少公共标量工作的重复执行，并缩短 `SoftmaxState` 合并阶段的寄存器生命周期。

### 2.6 shared-weight combine kernel

v10 combine 中的 128 个输出线程分别扫描全部 splits，并重复计算最大 LSE、权重和分母。v11 改为：

- `tid=0` 计算一次全部 split 权重和总分母。
- 权重及倒数存入 shared memory。
- 128 个线程只读取公共权重并累加各自的输出维度。
- 权重为 0 时不读取对应 `partial_o`，避免 `0 * NaN` 污染结果。

### 2.7 空 split 与 workspace 安全性

- 空 split 同时写入 `partial_lse=-inf` 和全零 `partial_o`。
- workspace 扩容时先申请新空间，成功后才释放旧空间。
- 分配失败时退回 direct kernel。

### 2.8 奇数 split 起始页流水修复

首版 v11 使用绝对页号选择双缓冲 stage：

```cpp
stage = page % 2;
```

但每个 split 都把自己的首个 page 预加载到 stage 0。当 `first_page` 为奇数时，计算会错误读取尚未初始化的 stage 1，最终产生 NaN。

修复后使用相对页号：

```cpp
stage = (page - first_page) % 2;
```

修复后官方 14 个 case 全部通过。

## 3. C500 实测结果

测试结果：

- 正确性：14/14 通过。
- 总分：`0.8180178744`。
- 平均 candidate 时间：`0.5418148602 ms`。
- 平均 reference 时间：`0.1501220573 ms`。

| Batch | KV | KV heads | candidate ms | speedup | max abs error |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 8 | 0.020173 | 2.303 | 0 |
| 4 | 2 | 8 | 0.018918 | 2.223 | 0.0078125 |
| 16 | 17 | 4 | 0.025293 | 2.162 | 0.015625 |
| 64 | 8 | 4 | 0.024064 | 1.737 | 0.015625 |
| 16 | 141 | 4 | 0.076646 | 0.482 | 0.0078125 |
| 32 | 362 | 8 | 0.139853 | 0.380 | 0.00390625 |
| 64 | 2048 | 4 | 1.107456 | 0.150 | 0.0078125 |
| 16 | 4096 | 4 | 0.678656 | 0.146 | 0.001953125 |
| 32 | 4096 | 8 | 0.879386 | 0.351 | 0.001953125 |
| 1 | 8192 | 4 | 0.149683 | 0.458 | 0.000244140625 |
| 16 | 12251 | 4 | 1.472947 | 0.168 | 0.0009765625 |
| 8 | 32768 | 8 | 1.517798 | 0.368 | 0.00048828125 |
| 1 | 58966 | 8 | 0.665190 | 0.315 | 0.0001220703125 |
| 1 | 61519 | 4 | 0.809344 | 0.208 | 0.0001220703125 |

区域得分：

| 区域 | v11 speedup |
|---|---:|
| short KV `<=2048` | 1.3482 |
| mid KV `=4096` | 0.2487 |
| long KV `>=8192` | 0.3035 |
| small batch `<=4` | 1.1016 |
| large batch `>=8` | 0.6605 |
| target long+small | 0.3271 |

## 4. 相比历史版本

| 版本 | 总分 | 说明 |
|---|---:|---|
| v10 | 0.651 | 完整 McFlashInfer 风格 BSM 主核，但 split/combine 偏重 |
| best | 0.755 | 调度与长序列 partition 更强 |
| v11 | **0.818** | v10 短序列主循环与 best 调度/归并结合 |

按总分计算，v11 相比 v10 提升约 25.7%，相比 best 提升约 8.3%。不同轮次的 reference 时间存在波动，因此跨版本单 case 比较应主要观察 candidate 时间和稳定趋势。

## 5. 当前瓶颈与 v12 方向

v11 的 split 数已经与 best 基本一致，但 KV=4096 及更长序列仍普遍慢于 best。这说明剩余差距主要来自 partition 主计算核，而不是 split 数：

- v10 partition 固定使用 256 threads 和双级 BSM。
- 每个 split page 数较少时，双缓冲启动、等待和块内状态维护成本难以摊薄。
- `scores[GROUP_SIZE]` 与完整 `SoftmaxState` 增加寄存器压力。
- best 的单级 page staging、逐 token online softmax 在这些 partition shape 上更轻。

因此 v12 的设计是双主核分发：

- `KV<=2048`：沿用 v11/v10 BSM kernel。
- `KV>2048`：使用 best 风格 partition kernel。
- split 策略、均衡分区、空 split 处理和 shared-weight combine 继续沿用 v11。


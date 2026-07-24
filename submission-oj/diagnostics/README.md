# 诊断脚本

用于验证 MMA fragment 加载、QK MMA、P×V MMA 正确性的独立诊断 kernel。

## diag_pv_mma
- `diag_pv_mma.cu` + `diag_pv_mma.py`
- 隔离测试 MMA P×V：给定已知 prob 和 V，对比 MMA 输出 vs 标量参考
- 结果：max error 0.0009（bf16 精度范围内，无 NaN）

## diag_integrated
- `diag_integrated.cu` + `diag_integrated.py`
- 集成测试：QK MMA → softmax → rescale → MMA P×V（完整 pipeline）
- 对比 MMA P×V vs 标量 P×V vs CPU 参考
- 结果：max error 0.0001 vs 标量，0.0002 vs CPU，无 NaN

## 关键发现
- MMA fragment 加载模式在独立/集成 diagnostic 中均正确
- NaN 仅在完整 kernel 上下文（v3/v4）中出现
- 根因：`pv_acc_all[8]`（8 个 MMA accumulator × 4 float）寄存器压力导致 spill

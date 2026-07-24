# Smoke Tests

快速验证 kernel 正确性的单 case 测试。

## smoke_v3.py / smoke_v4.py
- 编译运行 v3/v4 kernel，测试 kv=141, batch_size=8
- 检查输出是否含 NaN/Inf

## smoke_v4_bs16.py
- batch_size=16, kv=141（匹配官方 eval 条件）
- 待运行确认

## smoke_v4_O1.py
- 用 `-O1` 编译 v4，验证 NaN 是否与优化级别相关
- 结果：`-O1` 仍有 NaN（`-O3` 修复后也通过）

## smoke_v4_singleblock.py
- 单 block 测试（通过修改 run_kernel dispatch 逻辑）
- 待运行

## test_v4_singlepage.py / test_multi_page.py
- 独立编译的小 kernel 测试
- 不依赖 v4.cu 的 run_kernel dispatch

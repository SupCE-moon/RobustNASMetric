# 暂缓导入的成对文件

这两个文件来自主工作树 `784fc74333` 的未跟踪改动，依赖尚未提交的 `metrics.py` 补丁
（`_e36_cross_layer_stats` / `_e36_within_layer_stats`）。单独启用会让 `pytest` 变红，
因此这里只保存**原始字节**作为证据，不参与测试收集（文件名不以 `test_` 开头路径导入，
且不在 `tests/` 目录下）。

| 文件 | 大小 | SHA-256 |
| --- | --- | --- |
| `e36_batched.py` | 3,748 | `594d5693f0b50967ccccfa6e9e7e3039d85caf3d017b755ff815d3010da1df98` |
| `test_e36_batched.py` | 2,449 | `19a57ac8f45ee16056be002b8e35ad4f0fd792711152361209a79939d910a938` |

合并顺序应为：先评审并应用
`patches/main-uncommitted-784fc74-20261112.patch`，再把这两个文件移回
`src/robust_metric_lab/` 与 `tests/`，最后运行 `pytest tests -q`。

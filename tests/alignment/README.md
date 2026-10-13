# 对齐测试（tests/alignment）

存放与原始仓库参考输出的数值对齐测试（dev-plan M3 起填充）。合并门槛（AGENTS.md 规则 5）：

- fixture 来自在 `align/` 中 Docker 固化的原始仓库环境跑出的参考输出（分数即数组，体积小，进仓库）。
- 每个测试断言容差，失败信息给出「原始值 vs vsumbench 值」。
- 任何与原始实现的已知偏差，必须先记录在 `docs/protocols/` 再改代码。

M0 阶段此目录为空；随 M3 的 uniform / bi-LSTM / VASNet / PGLSUM 基线逐个填充。

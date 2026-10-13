# 对齐测试（tests/alignment）

存放与原始仓库参考输出的数值对齐测试。合并门槛（AGENTS.md 规则 5）：

- `test_eccv16_protocol_alignment.py`（M2）：参考流水线为**逐字转录**的源仓库评测代码（PGL-SUM `generate_summary`/`knapSack` + DR-DSN 的 `rank` 分支），在 75 个真实视频上比对选择掩码与逐人 F1；无真实数据时 skip。协议结论见 `docs/protocols/eccv16-summary-protocol.md`。
- M3 起的模型级 fixture：来自在 `align/` 中 Docker 固化的原始仓库环境跑出的参考输出（分数即数组，体积小，进仓库）。
- 每个测试断言容差，失败信息给出「原始值 vs vsumbench 值」。
- 任何与原始实现的已知偏差，必须先记录在 `docs/protocols/` 再改代码。

# AGENTS.md

面向 AI 编程代理与人类开发者的项目规范。本文件只放铁律和指针，详细设计在 `docs/` 中按需阅读。

## 项目定位

vsumbench 是视频摘要（video summarization）领域的统一实验框架，对标 lm-evaluation-harness：让大家**用统一的数据集和统一的评测协议跑自己的方法**，简化搭建实验环境的流程。核心价值：数据与协议开箱即用、评测协议显式化、协议层数值与源仓库实现对齐。**不内置、不复现他人的学习方法**——论文中与其他方法的对比数字按社区惯例直接引自原论文；学习模型由使用者/社区经 registry 接入。背景见 [README.md](README.md)；总体开发计划、里程碑与**当前进度**见 [docs/design/dev-plan.md](docs/design/dev-plan.md)。

## 铁律（任何改动不得违反）

1. **依赖最小**：核心代码只依赖 `numpy` + `h5py` + 标准库。torch 等训练框架一律放 extras（如 `vsumbench[torch]`）。
2. **分层单向依赖**：`cli / reporting / bench → selection / metrics / models → data → types`。低层禁止 import 高层；models 与 metrics 互不依赖。
3. **开闭原则**：新增数据集 / 模型 / 指标 = 实现对应 Protocol 接口 + 在 registry 注册 + 配测试。不允许为接入新组件而修改核心代码路径。
4. **协议改动必须显式**：任何影响数值的评测选择（标注聚合方式、归一化、budget 定义、selection 策略、split）只能是 `EvalConfig` 的显式参数，禁止埋进代码逻辑里静默改变。
5. **对齐测试是合并门槛**：改动可能影响评测数值（加载、展开、selection、指标）时，`tests/alignment/`（协议层：参考实现逐字转录比对）必须全绿。若确需偏离原始实现，先在 `docs/protocols/` 记录原因与偏差，再改代码。
6. **数据永不入库**：`data/`、`runs/`、`*.h5` 已 gitignore。仓库只分发 loader 与 checksum，不分发数据；不得硬编码本机数据路径。

## 质量门槛

- 提交前 `ruff check`、`mypy`（strict）、`pytest` 全绿（脚手架落地后以 CI 为准；命令见 dev-plan.md M0）。
- 公共 API 必须带完整类型注解（包带 `py.typed`）和单元测试；指标类实现必须含手算用例。
- 一切随机性通过显式 seed 注入；评测结果必须能落入 run manifest（git SHA + 完整 config + 数据 checksum）。

## 目标目录结构（M0 起逐步落地）

```
src/vsumbench/
  types.py  config.py  registry.py  bench.py
  data/  models/  selection/  metrics/  cli/  reporting/
tests/{unit,alignment,integration}
docs/{design,protocols,api}
```

## 上下文文档（按需阅读，不必一次读完）

- `docs/design/dev-plan.md` — 开发计划、里程碑与验收标准；**当前进度以此文件为准**
- `docs/design/core-contracts.md` — 核心接口契约（M0 产出；改接口前必读）
- `docs/protocols/` — 协议考古：各源仓库隐式评测选择的逐条记录
- `README.md` — 项目定位与 v0.1 范围

## 工作约定

- 用户工作语言为中文；标识符、docstring 遵循仓库既有惯例（英文标识符）。
- 开始或完成一个里程碑时，更新 `docs/design/dev-plan.md` 末尾的进度记录。
- 代码风格与注释密度向周围代码看齐；不为评审者写解释性注释。

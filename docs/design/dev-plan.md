# vsumbench 开发计划（从 0 到 v0.1）

- **状态**：定稿于 2026-10-13，作为项目执行的基准计划
- **维护规则**：里程碑状态变化时更新本文档（见末尾进度记录）；接口契约变动需同步 `core-contracts.md`
- **总体思路**：契约先行、分层解耦、对齐测试守护数值。分 6 个里程碑，单人全职约 9–10 周

本项目的价值全在「协议考古 + 数值可信」上。架构上最要紧的不是功能多，而是：**把接口契约定准，把每个数字变成可追溯的**。以下所有安排围绕这两点。

## 0. 「成熟、不是 demo」的验收口径

每个阶段都拿这个对照：

1. **5 分钟可复现**：新人 `pip install vsumbench` 后 5 分钟内复现出某篇论文表格里的一个数字。
2. **开闭原则**：新增一个数据集 / 模型 / 指标 = 实现一个接口 + 一组测试 + 注册，**不改动核心代码**。
3. **数字可追溯**：任何结果能回答「哪份代码（git SHA）、哪个协议参数、哪份数据（checksum）」。
4. **CI 常绿 + 发布承诺**：semver、CHANGELOG、明确的弃用政策。

## 1. 总体架构

五层结构，依赖严格单向向下：

```
┌─────────────────────────────────────────────────┐
│  CLI (vsumm) / Bench API / reporting            │  用户入口
├─────────────────────────────────────────────────┤
│  协议层: EvalConfig, Selector, Budget, Splits   │  把隐式评测选择显式化
├────────────────────────┬────────────────────────┤
│  Models (registry)     │  Metrics               │
├────────────────────────┴────────────────────────┤
│  Data: datasets / splits / download+checksum    │
├─────────────────────────────────────────────────┤
│  Core types: Video, Scores, ShotTable, Mask     │
└─────────────────────────────────────────────────┘
```

目录采用 src layout：

```
src/vsumbench/
  types.py        # Video / Scores / ShotTable 等核心数据结构
  config.py       # EvalConfig（协议参数的唯一真相源）+ JSON 序列化
  registry.py     # 基于 entry points 的注册机制
  bench.py        # Bench 编排器
  data/           # download.py, tvsum.py, summe.py, splits.py
  models/         # uniform.py, bilstm.py, vasnet.py, pglsum.py
  selection/      # shot_based.py, knapsack.py
  metrics/        # f1.py, rank.py
  cli/  reporting/
tests/
  unit/  alignment/   # 对齐 fixture + 容差断言
  integration/        # 真实数据端到端（pytest mark，可跳过）
docs/
  protocols/          # 协议考古文档（每篇一个源仓库）
  api/  design/       # mkdocs 站点 + 设计文档
```

两个关键的扩展性机制：

- **Registry 用 entry points**（`vsumbench.models` / `.metrics` / `.datasets` 三个组），第三方包不改我们源码就能注册自己的模型——这是 lm-eval-harness 生态能长起来的原因。
- **依赖分层**：core 只有 numpy + h5py；torch 等学习类依赖全部进 extras（`pip install vsumbench[torch]`），守住 README 的 Minimal 承诺。

## 2. 里程碑计划

### M0 — 工程地基（约 1 周）

pyproject（src layout、extras）、ruff + mypy(strict) + pytest、pre-commit、GitHub Actions（Python 3.10–3.13 矩阵）。

**本里程碑最核心的产出是设计文档** `docs/design/core-contracts.md`：定稿 `Model.predict(video) -> Scores`、`Selector.select(scores, shots, budget) -> Mask`、`Metric.score(...)`、`DatasetAdapter.load() -> list[Video]` 四个接口和 `EvalConfig` 的完整字段（多标注取 max 还是 mean、分数归一化方式、budget 定义……）。

**验收**：`pip install -e ".[dev]"` 可用，CI 全绿，契约文档评审通过。

⚠️ 整个项目 80% 的长期维护成本取决于这几个接口是否一次想清楚，宁可在这多花三天。

### M1 — 数据层（约 2 周）

`vsumm download`（原始来源 + checksum 校验 + `VSUM_CACHE_DIR` 缓存 + 本地路径覆盖的离线模式）；TVSum / SumMe loader 统一到 `Video` 结构（帧特征、多份人工摘要、镜头边界）；TVSum canonical 5-fold + 种子化随机切分；**小型合成 fixture** 供单测使用，不依赖真实数据和网络。

**验收**：真实数据集成测试可跑，checksum 不匹配时明确报错。

### M2 — 协议与指标（约 2 周，后半与 M3 并行）

shot-based 与 knapsack 两种 selection；F1@{10,15,30,50}%；Kendall's τ。指标全部配手算用例单测。

同时做**协议考古**：每个源仓库（VASNet、PGLSUM、SUM-GAN 系、DR-DSN 及两篇原始数据集论文）写一篇 `docs/protocols/*.md`，逐条列出隐式评测选择，全部转化为 `EvalConfig` 的显式参数。

**验收**：同一份模型分数，切换协议参数能分别复现不同论文报告的数字——这是本项目的核心卖点，也是和所有现有 repo 的本质区别。

### M3 — Baselines（约 3 周）

顺序：**uniform → bi-LSTM → VASNet → PGLSUM**（从无学习到有学习，逐步搭起训练 / 推理设施）。

对齐测试的方法论：用 pinned 环境（Docker 固化，放 `align/`，不进 CI）把原始仓库跑一次，参考输出存为 fixture 进仓库（分数就是数组，很小）；CI 断言 vsumbench 输出在容差内；种子固定，任何已知偏差写进文档而非隐藏。

**验收**：每个 baseline 一张对齐表（原始数字 vs 我们的数字 vs 容差），全绿才允许合并——这是 README 里 Faithful 承诺的技术支撑。

### M4 — 编排与体验（约 1.5 周）

Bench API + `vsumm eval / download / report`；**run manifest**：每次评测落一份 JSON，记录 git SHA、完整 config、数据 checksum，结果可 diff、可汇总成论文用的对比表。

**验收**：README 里那两段示例（CLI 和 Python API）逐字可跑。

### M5 — v0.1 发布（约 1 周）

mkdocs 文档站（教程 + 协议文档 + API 参考）、CONTRIBUTING.md（注册指南 + 对齐测试准入要求）、PyPI trusted publishing、CHANGELOG。

**验收**：干净环境一条命令完成安装到出结果。

v0.1 之后的事（OVP / VTW / QFVS、Spearman ρ、SUM-GAN / DR-DSN / CSNet、社区 splits）在这个架构下都退化为「再实现一个接口」，不需要结构性改动。

## 3. 测试与质量策略（防 demo 化的核心）

金字塔四层：

1. **单元测试**（快、零依赖）：指标手算用例、selection 边界条件、config 序列化
2. **对齐测试**（金色 fixture + 容差）
3. **集成测试**（真实数据端到端，打 mark，CI 缓存数据，本地可跳）
4. **CLI 冒烟**：全流程一遍

外加：核心层覆盖率 ≥ 90%、mypy strict、发布前 checklist。

## 4. 已定的技术决策（现在定死，避免返工）

| 决策点 | 选择 | 理由 |
|---|---|---|
| 注册机制 | entry points，非装饰器单例 | 第三方可扩展 |
| 配置系统 | dataclass + JSON，不引 pydantic | 守住最小依赖承诺 |
| 随机性 | 一切走显式 seed，进 manifest | 可复现 |
| 数据策略 | 只分发 loader；多镜像 + checksum + 离线模式 | 链接失效是本项目的立项起因 |
| 对齐资产 | fixture 进仓库，原始环境 Docker 化放 `align/` | CI 不背 Keras 时代的历史债 |

## 5. 风险与对策

- **Google Drive 数据链接失效**：checksum + 多镜像（可选 HuggingFace 镜像）+ 本地路径覆盖 + 定期巡检下载的 scheduled CI。
- **TF1 / Keras 时代原始代码跑不起来**：Docker 一次性固化生成 fixture，CI 只比对 fixture。
- **各原仓库协议互相矛盾**：这正是我们的机会——考古文档显式化 + 参数化 + 推荐默认值，不静默站队。
- **原代码许可证（部分 GPL）**：重实现以论文为规范，逐仓库记录 license 调查结论；必要时 clean-room。
- **单人开发节奏风险**：里程碑小步走，每个 M 结束时仓库都处于可用状态，不存在长期半成品。

## 6. 进度记录

- 2026-10-13：**M0 完成**。落地：pyproject（src layout、`torch`/`dev` extras、ruff/mypy strict/pytest 配置）、`types`/`config`/`registry`/`bench`/`cli` 及子包骨架（`py.typed` + `vsumm` 入口）、CI（Python 3.10–3.13 矩阵 + macos）、pre-commit、83 个单元测试（含分层依赖守护测试），三门槛本地全绿（3.10 / 3.12 双版本验证）。契约定稿 `docs/design/core-contracts.md` v1，含三处已批准偏差：Selector 显式 `aggregation` 参数、metrics 拆 summary/rank 两组、新增 selectors 注册组；GT 摘要构造等待考古项在 §8 列出。
- 2026-10-13：计划定稿；M0 尚未启动。

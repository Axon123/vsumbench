# vsumbench 开发计划（从 0 到 v0.1）

- **状态**：定稿于 2026-10-13，作为项目执行的基准计划；同日经用户决策调整定位（见进度记录）：**不做内置学习基线**，对比数字直接引自论文
- **维护规则**：里程碑状态变化时更新本文档（见末尾进度记录）；接口契约变动需同步 `core-contracts.md`
- **总体思路**：契约先行、分层解耦、协议对齐守护数值。分 5 个里程碑（原 6 个，砍掉基线复现），单人全职约 6–7 周

本项目是**实验框架**：让大家用统一的数据集和统一的评测协议跑自己的方法，简化搭建实验环境的流程。论文里与其他方法的对比数字按社区惯例直接引自原论文，本框架不负责复现它们。价值全在「协议考古 + 数值可信」上：**把接口契约定准，把每个数字变成可追溯的**。以下所有安排围绕这两点。

## 0. 「成熟、不是 demo」的验收口径

每个阶段都拿这个对照：

1. **5 分钟出数**：新人 `pip install vsumbench` 后 5 分钟内下载好 TVSum/SumMe，在自己的模型上按 canonical 协议跑出 F1（模型只需实现 `predict(video) -> Scores`）。
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
  models/         # score_space.py（步→帧展开）；学习模型一律第三方经 registry 接入
  selection/      # shot_based.py, knapsack.py
  metrics/        # f1.py, rank.py
  cli/  reporting/
tests/
  unit/  alignment/   # 协议层逐字转录对齐 + 容差断言
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

### M2 — 协议与指标（约 2 周）

shot-based 与 knapsack 两种 selection；F1@{10,15,30,50}%；Kendall's τ。指标全部配手算用例单测。

同时做**协议考古**：每个源仓库（VASNet、PGLSUM、SUM-GAN 系、DR-DSN 及两篇原始数据集论文）写一篇 `docs/protocols/*.md`，逐条列出隐式评测选择，全部转化为 `EvalConfig` 的显式参数。

**验收**：同一份模型分数，切换协议参数能分别得到不同论文口径的数字——这是本项目的核心卖点，也是和所有现有 repo 的本质区别。（协议层对齐测试以逐字转录的参考实现守护，见 `tests/alignment/`。）

> 原计划的 M3（实现 uniform/bi-LSTM/VASNet/PGLSUM 四个基线并对齐原始仓库）已于 2026-10-13 经用户决策砍掉：论文对比数字按社区惯例直接引自原论文，框架不负责复现。学习模型一律由使用者/社区经 registry 接入；唯一保留的内置模型是无学习的 `uniform` 参照（CLI 冒烟与示例用，无对齐承诺）。Docker 固化原始环境的 `align/` 设施随之取消；若第三方要移植某个已发表模型并做数值对齐，方法仍记录在 CONTRIBUTING（可选准入）。

### M3 — 编排与体验（约 1.5 周，原 M4）

Bench API + `vsumm eval / download / report`；**run manifest**：每次评测落一份 JSON，记录 git SHA、完整 config、数据 checksum，结果可 diff、可汇总成论文用的对比表（对比表中的他人方法数字由用户从论文抄录，框架不生成）。内置 `uniform` 参照模型随本里程碑落地（models registry 首个内置条目）。

**验收**：README 里那两段示例（CLI 和 Python API）逐字可跑。

### M4 — v0.1 发布（约 1 周，原 M5）

mkdocs 文档站（教程 + 协议文档 + API 参考）、CONTRIBUTING.md（注册指南；移植已发表模型者建议附对齐测试，作为可选准入）、PyPI trusted publishing、CHANGELOG。

**验收**：干净环境一条命令完成安装到出结果。

v0.1 之后的事（OVP / VTW / QFVS、Spearman ρ、社区 splits、他人模型的社区移植）在这个架构下都退化为「再实现一个接口」，不需要结构性改动。

## 3. 测试与质量策略（防 demo 化的核心）

金字塔四层：

1. **单元测试**（快、零依赖）：指标手算用例、selection 边界条件、config 序列化
2. **对齐测试**（协议层：参考实现**逐字转录**进测试，真实数据上比对掩码与指标值 + 容差断言；M2 已落地）
3. **集成测试**（真实数据端到端，打 mark，CI 缓存数据，本地可跳）
4. **CLI 冒烟**：全流程一遍（用内置 `uniform` 参照）

外加：核心层覆盖率 ≥ 90%、mypy strict、发布前 checklist。

## 4. 已定的技术决策（现在定死，避免返工）

| 决策点 | 选择 | 理由 |
|---|---|---|
| 注册机制 | entry points，非装饰器单例 | 第三方可扩展 |
| 配置系统 | dataclass + JSON，不引 pydantic | 守住最小依赖承诺 |
| 随机性 | 一切走显式 seed，进 manifest | 可复现 |
| 数据策略 | 只分发 loader；多镜像 + checksum + 离线模式 | 链接失效是本项目的立项起因 |
| 对齐资产 | 参考评测代码逐字转录进 `tests/alignment/`，真实数据上比对 | 不内置基线，无 Docker/环境固化负担 |

## 5. 风险与对策

- **Google Drive 数据链接失效**：checksum + 多镜像（可选 HuggingFace 镜像）+ 本地路径覆盖 + 定期巡检下载的 scheduled CI。
- **各原仓库协议互相矛盾**：这正是我们的机会——考古文档显式化 + 参数化 + 推荐默认值，不静默站队。
- **框架没有"官方数字"可背书**：不做基线复现意味着框架本身不产出与论文对得上的方法数字；对策是协议层对齐测试（我们的尺子准）+ run manifest（用户数字可追溯）+ 协议文档明示各论文口径差异。
- **单人开发节奏风险**：里程碑小步走，每个 M 结束时仓库都处于可用状态，不存在长期半成品。

## 6. 进度记录

- 2026-10-13：**M3 完成**（编排与体验）。落地：`bench.py`——`Bench` 双形态构造（显式 `EvalConfig`，或便捷 kwargs 落到推荐协议默认 uniform/knapsack/f1/F1@{10,15,30,50}%，其余回落 EvalConfig 字段默认）、`register(name, model)` 注册即选用、run 循环（加载 → registry 全解析 → 切分 → 每折 deepcopy+fit → predict（校验 id/长度）→ 逐预算 select → 逐人指标 → `gt_aggregation` 收缩；逐视频记录挂折内——canonical 折 test 集互相重叠（tvsum 50 槽 34 视频、summe 25 槽 18），学习模型跨折同视频可不同值，扁平映射会覆盖；跨折 mean/std(ddof=1)/max/n_folds）；`reporting/manifest.py`——schema `vsumbench.run-manifest/1`（git SHA+dirty、python/platform/numpy、完整 config、逐文件 sha256 且与 SOURCES 不匹配即失败、SOURCES 未收录显式记 null 不装作验证过、切分明细、结果块；非有限浮点写 null 保证严格 JSON；时间戳 run 目录永不覆盖）；`reporting/table.py` + CLI `vsumm eval`（协议 flags 或 `--config` 全量 JSON，互斥）/`vsumm report`（文件或目录扫描 manifest，markdown 对比表，他人方法数字仍由用户从论文抄录）；`models/uniform.py` 经 entry points 注册（models 组首个内置条目）；`from vsumbench import Bench` 惰性导出（PEP 562，子模块导入保持轻量、分层守护测试继续有效）。README 两段示例的 API 形状逐字可用；CLI 真实 tvsum canonical-5 冒烟通过（uniform F1@15% = 0.8217±0.0241，max 口径）。测试：单测 255 个全绿（手算端到端用例、greedy 严格小于预算语义、checksum 记录/不匹配、NaN→null、registry 单例零污染+fit 只见 train、混用构造拒绝）；集成 8 个全绿（tvsum/summe canonical-5 端到端含 manifest 严格 JSON 往返）；三门槛（ruff / mypy strict / pytest）全绿。契约升 v1.3（§10 编排层）。剩余到 v0.1：M4（mkdocs 文档站、CONTRIBUTING、PyPI trusted publishing、CHANGELOG）。
- 2026-10-13：**定位调整：砍掉基线复现里程碑**（用户决策）。vsumbench 定位为「统一数据集 + 统一评测协议的实验框架」：方便大家采用统一数据集、简化实验搭建；论文中与其他方法的对比数字按社区惯例直接引自原论文，框架不内置、不复现他人方法。变更：原 M3（uniform→bi-LSTM→VASNet→PGLSUM 四基线 + Docker `align/` 对齐设施）整体取消；原 M4/M5 顺位改为 M3/M4；唯一保留的内置模型是无学习的 `uniform` 参照（随新 M3 编排里程碑落地，供 CLI 冒烟与示例，无对齐承诺）。README/AGENTS/core-contracts 同步改口径。已完成且不受影响的资产：M0–M2 全部（数据层、协议考古、selectors/metrics、协议层对齐测试）。
- 2026-10-13：**M2 完成**（协议与指标）。考古：六个源仓库评测代码逐文件核对（PGL-SUM/CA-SUM/SUM-GAN-AAE/sl/SUM-GAN-TF + DR-DSN=pytorch-vsumm-reinforce + ok1zjf/VASNet fork；原 e-apostolidis/VASNet 已删库），结论进 `docs/protocols/eccv16-summary-protocol.md`（主文档 + 各仓库差异表）与 `tvsum-summe-originals.md`（原始论文 vs eccv16 打包）。关键事实：展开=前向填充且**尾部延续**（M1 文档"尾部补 0"系误记，已修正 core-contracts §2.2）；预算全部 floor；knapsack 并列**不取**、greedy(`rank`) 并列偏后 + 严格小于 limit；跨标注 `avg(tvsum)/max(summe)` 硬编码（eccv16 原论文则均 max）；VASNet 对训练目标再 min-max（h5 gtscore 实测并非 min-max，非幂等）；τ 不出自任何源仓库（源自 eccv16 原论文、步空间对 gtscore）。落地：`EvalConfig` + `budget_rounding`/`step_score_expansion`（契约 v1.2，Budget 带 `rounding`+`frame_limit()`）、`models/score_space.py`（步→帧展开唯一实现，float32 忠实）、`selection/{_common,shot_based,knapsack}.py`（selectors registry：`greedy`/`knapsack`，并列语义逐字复刻，DP 向量化但浮点行为与参考一致）、`metrics/{f1,rank}.py`（`f1` 比值口径 + 空摘要记 0 为已记录偏差；`kendall_tau` 为 τ-b 逐人版，tvsum/summe 待逐人分数接入）。测试：单测 210 个全绿（指标手算用例、selector 边界/并列/取整、config/registry/分层扩展），对齐测试 `test_eccv16_protocol_alignment.py` 在 75 个真实视频上以逐字转录参考实现验证选择掩码与逐人 F1（rtol 1e-12，knapsack 另与参考 DP 300 组随机并列实例全等）；三门槛（ruff/mypy strict/pytest）+ 集成测试 6 个全绿。
- 2026-10-13：**M1 完成**。落地：`data/sources.py`（eccv16 TVSum/SumMe 镜像表 + sha256，出处：ECCV16 特征 → KaiyangZhou h5 → e-apostolidis 仓库内嵌 rar）、`data/download.py`（`vsumm download`，多镜像 + sha256 校验 + bsdtar/unrar/unar/7z 解包 + `VSUM_CACHE_DIR`/`--root` 离线覆盖，stdlib-only）、`data/_eccv16.py` + `tvsum.py`/`summe.py`（adapter 经 entry points 注册）、`data/splits.py`（`tvsum-canonical-5`/`summe-canonical-5` 逐字转录 + 种子化随机 k-fold）、`Video` 契约升 v1.1（双帧空间 + `picks` 字段，eccv16 h5 的特征在 15 帧子采样步空间、GT 在原始帧空间，详见 core-contracts §2.2 决策记录）。测试：合成 h5 fixture 全离线单测 141 个（download 用本地 HTTP + tar 镜像，含坏镜像回退/校验失败不落盘）；真实数据集成测试 6 个（checksum + 全量加载 + canonical 切分匹配）本地通过；真实 rar 解包→校验→落盘路径已验证。考古事实（逐视频验证 75/75）：`picks = range(0, n_frames, 15)`、`change_points` 闭区间端点完美铺贴、`user_summary` 严格二值、两 h5 均无逐人重要性分。
- 2026-10-13：**M0 完成**。落地：pyproject（src layout、`torch`/`dev` extras、ruff/mypy strict/pytest 配置）、`types`/`config`/`registry`/`bench`/`cli` 及子包骨架（`py.typed` + `vsumm` 入口）、CI（Python 3.10–3.13 矩阵 + macos）、pre-commit、83 个单元测试（含分层依赖守护测试），三门槛本地全绿（3.10 / 3.12 双版本验证）。契约定稿 `docs/design/core-contracts.md` v1，含三处已批准偏差：Selector 显式 `aggregation` 参数、metrics 拆 summary/rank 两组、新增 selectors 注册组；GT 摘要构造等待考古项在 §8 列出。
- 2026-10-13：计划定稿；M0 尚未启动。

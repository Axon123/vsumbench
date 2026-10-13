# 核心接口契约（core contracts）

- **状态**：v1.3（M3，2026-10-13）。v1.1：`Video` 增加双帧空间语义与 `picks` 字段（§2.2）；v1.2：`Budget` 增加 `rounding` 字段（§2.4）、`EvalConfig` 增加 `budget_rounding` / `step_score_expansion`（§4），§8 的 M2 留白项全部定稿并修正一处 M1 错误记述（展开尾部语义）；v1.3：新增 §10 编排层（Bench / RunResult / run manifest 结构，M3 落地）
- **受众**：任何要给 vsumbench 添加数据集 / 模型 / 指标 / selector 的人（包括未来的我们自己）
- **维护规则**：改任何接口或 `EvalConfig` 字段语义前必读并更新本文档。契约由 `src/vsumbench/types.py`、`config.py`、`registry.py` 承载：本文档写动机与决策记录，代码是可执行的事实，两者不一致时立即修文档。
- **兼容性政策**：新增字段 / 新组件 = minor；改字段默认值或语义、删字段 = major，需迁移说明 + `tests/alignment/` 全绿。

## 1. 分层与依赖方向

```
cli / reporting / bench          # 用户入口与编排
        │
selection / metrics / models     # 实现 §3 的协议，只依赖 types + config
        │
registry                         # 名字 → 组件（只依赖 types）
        │
data (loaders / splits / download)
        │
types ──▶ config                 # types 的协议签名引用 EvalConfig
config                            # 叶子：不 import 任何 vsumbench 模块
```

- 依赖严格单向向下（AGENTS.md 规则 2），`tests/unit/test_layering.py` 用子进程逐模块断言实际加载集，防止隐性上溯。
- models 与 metrics 互不依赖；两者都不知道 registry 的存在——它们被 registry 发现，而不是反过来。
- 扩展新组件 = 实现 Protocol + entry point（或 `register()`）+ 配测试，**不改核心代码**（规则 3）。

## 2. 核心数据结构（`vsumbench.types`）

通用约定：

- 帧下标 `0 .. n_frames-1`；镜头 `i` 覆盖半开区间 `[boundaries[i], boundaries[i+1])`。
- 构造时接受 numpy 数组或嵌套序列，**一律拷贝并置为只读**——共享 fixture 不会被某个视图意外污染，评测数值可信任。
- 所有 dataclass 均 frozen。

### 2.1 ShotTable —— 帧分割

| 字段 | 类型 | 不变量 |
|---|---|---|
| `boundaries` | `int64[ n_shots+1 ]` | 严格递增；`[0]==0`；`[-1]==n_frames`；无空镜头 |
| `n_frames` | `int` | ≥ 1，且与 boundaries 终点一致 |

- 数据集自带镜头边界（SumMe gtt）由 loader 原样转入；数据集没有（TVSum）时，分段方式是**协议选择**，由编排层按 `EvalConfig` 生成（M2 定稿），loader 不得私自决定。
- `shot_span(i)` 返回第 `i` 个镜头的 `[start, end)`。

### 2.2 Video —— 评测的最小完整单元

| 字段 | 类型 | 说明 |
|---|---|---|
| `dataset` / `video_id` | `str` | 注册名 + 数据集内唯一 id |
| `n_frames` | `int` | **原始帧空间**长度；`shots` 与所有逐人标注对齐到它 |
| `shots` | `ShotTable` | 原始帧空间的分割，与 `n_frames` 对齐 |
| `features` | `float32/64 [n_steps, dim]` 或 `None` | 帧特征（GoogLeNet 等）；行与 `picks` 一一对应 |
| `picks` | `int64[n_steps]` 或 `None` | 每个 feature 行对应的原始帧下标；严格递增且落在 `[0, n_frames)`；`None` 表示两个空间重合 |
| `fps` | `float` 或 `None` | 仅秒类协议需要；源数据没有就留空，不猜 |
| `user_scores` | `tuple[float[n_frames], ...]` | 逐人帧重要性（原始帧空间） |
| `user_summaries` | `tuple[bool[n_frames], ...]` | 逐人帧级二值摘要（原始帧空间） |

**决策记录：为什么是双帧空间（v1.1）。** eccv16 TVSum/SumMe 的 h5 把特征按每 15 原始帧 1 步子采样（`features`/`gtscore` 在步空间，`user_summary`/`change_points`/`n_frames` 在原始帧空间，`picks` 是映射——M1 已逐视频验证，两数据集 75 个视频全部严格成立）。单一 `n_frames` 不变量在真实数据上不成立；让 loader 私自把某个空间重采样到另一个空间是影响数值的协议选择（规则 4），因此 `Video` 忠实携带两个空间 + `picks`，步 ↔ 帧的展开方式（展开值、尾部填充）推迟到 M2 考古后进 `EvalConfig`（§8）。`n_steps` 属性 = `len(picks)`（无 `picks` 时回落 `n_frames`）。另：两个 canonical h5 **均无**逐人重要性分（`gtscore` 是跨人聚合的派生量，被 loader 忽略），故 TVSum/SumMe 的 `user_scores` 为空；逐人分数需另接原始 ydata 标注（v0.1 后）。

**M2 定稿（展开协议，v1.2）**：`Scores` 的规范空间是原始帧空间；步分 → 帧分由模型侧唯一的共享实现 `vsumbench.models.score_space.expand_step_scores` 完成，协议参数是 `EvalConfig.step_score_expansion`。eccv16 谱系实测为**前向填充、尾部延续**（尾区间 `[picks[-1], n_frames)` 取最后一个分数；M1 文档中"尾部补 0"的记述有误，参考代码的补零分支是长度错配防御的死代码——见 docs/protocols/eccv16-summary-protocol.md §1）。

**GT 原始性原则**：loader 只做格式搬运，不做任何协议加工——不归一化、不聚合、不把 gtscore 转成摘要。`Video` 里存的是原始逐人标注；每个「多标注怎么合成一个数」的选择都归 `EvalConfig`（规则 4）。

### 2.3 Scores —— 模型输出（**帧级**）

```python
Scores(video_id: str, values: float[n_frames])
```

**决策记录：为什么钉死帧级。** VASNet 系在段表示上打分，PGLSUM/SUM-GAN 系在帧级打分；帧级是唯一无损的公共分母（段级模型把段分数铺回帧即可，`reduceat` 反向无损）。帧 → 段的聚合（mean/max）是**影响数值的协议参数**，因此放进 `EvalConfig.segment_score_aggregation`，由编排层显式传给 selector，而不是藏在某个模型或 selector 内部。

### 2.4 Mask / Budget / Split

- `Mask(video_id, frames: bool[n_frames])` —— 机器摘要的帧级表示，selector 的输出、summary metric 的输入。
- `Budget(fraction: float, rounding: "floor"|"round"|"ceil" = "floor")`，`0 < fraction ≤ 1` —— 预算占总帧数比例（0.15 = F1@15%）；`frame_limit(n_frames)` 按 `n_frames * fraction` 先乘后取整（M2 定稿：eccv16 谱系全部用 `int()` 即 floor，见 docs/protocols/eccv16-summary-protocol.md §1；`round` 为 Python 的 half-to-even）。
- `Split(train: tuple[str,...], test: tuple[str,...])` —— video id 级切分；两侧非空、无重复、无交集。

## 3. 四个扩展协议（Protocol，结构化类型）

所有协议都要求 `name: str`（注册名）。组件对同一 config 必须是确定性的：一切随机性只能来自 `EvalConfig.seed`（§6）。

### 3.1 Model

```python
class Model(Protocol):
    name: str

    def fit(self, videos: Sequence[Video], config: EvalConfig) -> None: ...
    def predict(self, video: Video) -> Scores: ...
```

- `fit` 收到训练 split 与完整协议配置（超参在 `config.training`，种子在 `config.seed`）；无学习基线可以是 no-op。
- `predict` 返回**帧级** `Scores`，长度必须等于 `video.n_frames`（消费者可复验）。段内打分的模型自己负责把段分数铺回帧。
- 生命周期：registry 里放的是**实例**；有状态模型在 `fit` 后持有参数。编排层（M3 Bench）在**每折** fit 前 deepcopy 实例，跨折与跨 run 均无状态泄漏。

### 3.2 Selector

```python
class Selector(Protocol):
    name: str

    def select(
        self, scores: Scores, shots: ShotTable, budget: Budget, *, aggregation: SegmentAggregation
    ) -> Mask: ...
```

- 职责：按 `aggregation` 把帧分聚成段分，再在预算内选段、铺回帧得到 `Mask`（如 shot-based 贪心、knapsack）。
- `aggregation` 由编排层从 config 显式传入——selector 自己**无权**选择聚合方式（规则 4）。
- 必须确定性、无随机性。

### 3.3 SummaryMetric / RankMetric（两组）

```python
class SummaryMetric(Protocol):
    name: str

    def score(self, summary: Mask, video: Video) -> np.ndarray: ...  # shape [n_user_summaries]


class RankMetric(Protocol):
    name: str

    def score(self, scores: Scores, video: Video) -> np.ndarray: ...  # shape [n_user_scores]
```

- **每标注一个值**：F1 对 20 份用户摘要各算一个，Kendall τ 对 20 份 gtscore 各算一个；跨标注如何收缩成报告值（max/mean）是编排层按 `gt_aggregation` 做的事，指标本身永不聚合。
- 指标实现必须带手算用例单测（AGENTS.md 质量门槛）。

### 3.4 DatasetAdapter

```python
class DatasetAdapter(Protocol):
    name: str

    def load(self, root: Path) -> list[Video]: ...
```

- **只加载、不下载**：下载 / 校验 / 缓存归 `vsumbench.data.download`（M1）；`root` 必须已是本地 canonical 布局。
- **只搬运、不加工**：见 §2.2 GT 原始性原则。多份标注原样保留。

## 4. EvalConfig（`vsumbench.config`）

协议参数的唯一真相源；纯 JSON-safe dataclass（无 numpy），可直接落进 run manifest。默认值即**推荐协议**。

| 字段 | 类型 / 默认 | 语义与理由 |
|---|---|---|
| `dataset` | `str` | datasets registry 名 |
| `model` | `str` | models registry 名 |
| `selector` | `str` | selectors registry 名 |
| `metrics` | `tuple[str, ...]` | 指标 registry 名列表 |
| `budgets` | `tuple[float, ...]` | 预算比例列表，如 `(0.1,0.15,0.3,0.5)` 对应 F1@{10,15,30,50}% |
| `splits` | `CanonicalSplits \| RandomSplits` | 切分协议（见下） |
| `seed` | `int = 0` | 全局唯一随机性来源 |
| `gt_aggregation` | `"max" \| "mean"` = `"max"` | 跨标注聚合。推荐默认 = eccv16 原论文口径（两数据集均 max）；注意 KaiyangZhou/e-apostolidis 谱系代码对 TVSum 用 avg（`'avg' if tvsum else 'max'` 硬编码），复现那些论文表值时需切 `mean` |
| `segment_score_aggregation` | `"mean" \| "max"` = `"mean"` | 帧分 → 段分聚合（§2.3 决策）。谱系全部用 mean（max 为本工具包的显式泛化） |
| `budget_rounding` | `"floor" \| "round" \| "ceil"` = `"floor"` | 预算 → 帧数取整（M2 考古：全部谱系仓库 floor，`int(n_frames * proportion)`） |
| `step_score_expansion` | `"forward_fill" \| "zero_fill"` = `"forward_fill"` | 步分 → 帧分展开（M2 考古：谱系为前向填充、尾部延续，见 §2.2 M2 定稿与 protocols/eccv16-summary-protocol.md §1） |
| `score_normalization` | `{method: none/min_max/z_score}` = `none` | 打分归一化。M2 考古定稿作用位置：谱系在**选择前不归一化**预测分；min_max 只作为**训练目标**归一化存在（VASNet 口径），由接入的模型（第三方/自研）按 config 消费，不作用于评测路径 |
| `training` | `TrainingConfig` | epochs / lr / batch / device + `extra`（模型专属 JSON 参数，同样进 manifest，无处可藏） |

切分配置是带 `kind` 标签的 tagged union：

- `CanonicalSplits(name)` —— 使用已发表的具名切分；M1 内置 `tvsum-canonical-5` 与 `summe-canonical-5`（VASNet 谱系，逐字转录自 PGL-SUM/CA-SUM 的 `data/splits/*.json`）；
- `RandomSplits(n_folds=5, seed=0)` —— 种子化随机 k-fold；`seed` 独立于顶层 `seed`，便于固定切分复用。

**JSON 契约**：`to_dict()` 输出 JSON-safe dict（tuple→list，tagged union 带 `"kind"`）；`from_dict()` 拒绝未知键（防手写配置拼错字段名）、缺省键回落默认值；`to_json/from_json/save/load` 走该规范形式。

## 5. Registry（`vsumbench.registry`）

五个 entry-point 组，一组一个 `Registry` 实例：

| 组名 | 协议 | 用途 |
|---|---|---|
| `vsumbench.datasets` | `DatasetAdapter` | 数据集 |
| `vsumbench.models` | `Model` | 模型 |
| `vsumbench.selectors` | `Selector` | 选择策略 |
| `vsumbench.metrics` | `SummaryMetric` | 摘要重叠类指标（F1@budget） |
| `vsumbench.rank_metrics` | `RankMetric` | 排序相关类指标（Kendall τ） |

- 第三方声明示例（零 vsumbench 源码改动）：

  ```toml
  [project.entry-points."vsumbench.models"]
  my-model = "my_pkg.model:MY_MODEL"
  ```

- 值必须是**实例**（不是类）。运行时只做属性/方法存在性检查（`runtime_checkable` protocol 的 `isinstance`），签名兼容由 mypy 把关。
- 发现时机：首次 `get()/names()` 惰性扫描 entry points；加载失败抛 `RuntimeError` 并指名出错条目。
- 优先级：显式 `register()` > 同名 entry point（先到先得，`setdefault` 语义）。
- **内置组件（M1 的 tvsum/summe、M2 的 selector/指标、M3 的 uniform 参照）走同一机制**，在我们自己的 `pyproject.toml` 里声明，与第三方完全同构——这是「开闭原则」的机制保证。学习模型一律第三方接入（2026-10-13 定位调整，不内置基线）。

## 6. 随机性与可复现

- 唯一种子来源：`EvalConfig.seed`（切分如需独立复用，用 `RandomSplits.seed` 覆盖）。模型训练、任何抽样都从这里派生；组件**不得**自带熵源（`random.random()`、未播种的 numpy/torch 全局态都算违规）。
- 评测结果必须能落到 run manifest：git SHA + 完整 `EvalConfig` + 数据 checksum。manifest 已随 M3 落地（`vsumbench.reporting.manifest`，schema `vsumbench.run-manifest/1`），结构见 §10。

## 7. 与 dev-plan 签名的已批准偏差

1. **`Selector.select` 增加 keyword-only `aggregation` 参数**（dev-plan 原签名 `select(scores, shots, budget) -> Mask`）。理由：帧→段聚合改变数值，必须是显式协议参数（规则 4），放签名里由编排层传入，而非各 selector 自行其是。
2. **metrics 拆成两组** `SummaryMetric` / `RankMetric`（entry-point 组 `vsumbench.metrics` / `vsumbench.rank_metrics`，dev-plan 只列了单一 `.metrics` 组）。理由：F1 与 Kendall τ 输入类型不同（Mask vs Scores），分组让注册与分发都类型安全。
3. **新增 `vsumbench.selectors` 注册组**。理由：selector 与数据集/模型/指标同等可扩展，机制统一。

## 8. M2 协议考古定稿记录（原留白清单）

M2 对六个源仓库（VASNet fork、PGL-SUM、CA-SUM、SUM-GAN-AAE/sl、DR-DSN/pytorch-vsumm-reinforce、原版 SUM-GAN TF）完成逐文件考古，逐条结论与代码引文见 **docs/protocols/eccv16-summary-protocol.md**（主文档）与 **docs/protocols/tvsum-summe-originals.md**（原始论文 vs eccv16 打包）。定稿结果：

- **GT 摘要构造**：GT 是人工标注原样数据，评测只构造机器摘要；`gtscore → 摘要` 的转换（knapsack over shots + floor 预算）是**机器侧**选择路径，由 selector 承担，GT 侧无对应参数。
- **budget 取整**：`budget_rounding = "floor"`（§4），`Budget.frame_limit` 复刻 `int(n_frames * proportion)` 浮点行为。
- **归一化作用位置**：`score_normalization` 定位为模型侧训练目标归一化（VASNet per-video min_max），评测/选择路径不归一化（§4）。
- **步分 → 帧分展开**：`step_score_expansion = "forward_fill"`，尾部延续最后一步分数（修正 M1 的"尾部补 0"误记，§2.2）；共享实现 `models/score_space.py`。
- **SumMe fps**：维持不猜（h5 无 fps，秒类协议等上游提供）。
- **F1@K 语义**：预算只作用于机器摘要；GT 不变。谱系仓库全部只评 15%，`budgets` 元组为工具包泛化。
- 另记录两处谱系协议事实，供 M4 编排定稿：跨标注聚合 `'avg' if tvsum else 'max'`（与 eccv16 原论文的 max 口径不一致，是跨论文数字差异主因）；最优 epoch 按**测试集** F1 挑选（测试集窥视，编排层须显式暴露）。

## 9. 组件测试准入（合并门槛）

- **数据集 adapter**：合成 fixture 单测（零网络、零真实数据）+ checksum 集成测试（`-m integration`）。
- **模型**：vsumbench 不内置学习模型（定位调整，见 dev-plan）；内置的 `uniform` 参照只需单测。第三方移植已发表模型时**建议**附对齐测试（参考 `tests/alignment/` 的逐字转录法），已知偏差先记录 `docs/protocols/`。
- **指标**：手算用例（必须有，小数组人工算期望值）。
- **selector**：边界条件（budget 全选 / 极小、单段视频、并列分数、恰好整除）。

## 10. 编排层（M3：Bench / RunResult / run manifest）

### 10.1 Bench

两种构造形态：`Bench(config)` 显式全量协议；`Bench(dataset=..., splits=..., ...)` 便捷 kwargs——未给字段取**推荐协议默认**：`model="uniform"`（仅冒烟参照）、`selector="knapsack"`、`metrics=("f1",)`、`budgets=(0.1,0.15,0.3,0.5)`，其余协议字段回落 `EvalConfig` 字段默认（§4）；`splits` 字符串 = canonical 名，缺省 = `RandomSplits()`。`data_root` 是运行时参数（数据在哪），不是协议参数（怎么评），可与任一形态并用。`Bench.register(name, model)` 注册进 MODELS **并**将该模型选为运行对象（README 示例的形状）。

run 循环（对数值零自由裁量，只做组装）：adapter 加载 → 全部 registry 名解析（错误尽早暴露）→ 切分 → 每折：deepcopy 模型 → `fit(train)` → 逐 test 视频 `predict`（校验 video_id 与长度）→ 逐预算 `select` → 逐人指标值 → 按 `gt_aggregation` 收缩。

### 10.2 结果块与 run manifest

- **逐视频记录挂在折内**（`results.per_fold[i].videos[vid][metric][budget_key]`）：canonical 折的 test 集**互相重叠**（逐字转录事实：tvsum 50 槽 34 个不同视频、summe 25 槽 18 个），学习模型对同一视频在不同折可给出不同分数，扁平 per-video 映射会覆盖丢值。
- summary 指标以预算比例字符串为键（`"0.15"`）；rank 指标预算无关，键为 `"none"`（`NO_BUDGET`）。
- 跨折 summary：折均值的 mean / std（样本标准差 ddof=1，单折记 null）/ max / n_folds——max 与 std 一并记录，供不同论文口径（「最优折」vs「均值±方差」）直接引用，不改协议。
- NaN 逐人值（如退化标注下的 Kendall τ）**传播**到聚合并在 manifest 中写为 `null`（严格 JSON，拒绝 `NaN` 字面量）。
- manifest = `schema`（`vsumbench.run-manifest/1`）+ `created_utc` + `code`（版本、git SHA、git dirty、python/platform/numpy）+ `data`（root、逐文件 sha256 与 spec 比对；SOURCES 未收录的数据集显式记 `files: null`，不假装验证过）+ `config`（`EvalConfig.to_dict()` 原样）+ `splits`（逐折 train/test 明细）+ `results`。数据集在 SOURCES 收录而 checksum 不匹配时 run 直接失败，不产出数字。
- run 目录 `<UTC 时间戳>-<dataset>-<model>-<selector>`，同秒重跑加序号，永不覆盖（可 diff）。

# 核心接口契约（core contracts）

- **状态**：M0 定稿 v1（2026-10-13）
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
| `n_frames` | `int` | 下面所有数组的公共长度 |
| `shots` | `ShotTable` | 必须与 `n_frames` 对齐 |
| `features` | `float32/64 [n_frames, dim]` 或 `None` | 帧特征（GoogLeNet 等）；纯 selection/metric 实验可缺省 |
| `fps` | `float` 或 `None` | 仅秒类协议需要；源数据没有就留空，不猜 |
| `user_scores` | `tuple[float[n_frames], ...]` | 逐人帧重要性（TVSum 20 人） |
| `user_summaries` | `tuple[bool[n_frames], ...]` | 逐人帧级二值摘要（SumMe 15 人） |

**GT 原始性原则**：loader 只做格式搬运，不做任何协议加工——不归一化、不聚合、不把 gtscore 转成摘要。`Video` 里存的是原始逐人标注；每个「多标注怎么合成一个数」的选择都归 `EvalConfig`（规则 4）。

### 2.3 Scores —— 模型输出（**帧级**）

```python
Scores(video_id: str, values: float[n_frames])
```

**决策记录：为什么钉死帧级。** VASNet 系在段表示上打分，PGLSUM/SUM-GAN 系在帧级打分；帧级是唯一无损的公共分母（段级模型把段分数铺回帧即可，`reduceat` 反向无损）。帧 → 段的聚合（mean/max）是**影响数值的协议参数**，因此放进 `EvalConfig.segment_score_aggregation`，由编排层显式传给 selector，而不是藏在某个模型或 selector 内部。

### 2.4 Mask / Budget / Split

- `Mask(video_id, frames: bool[n_frames])` —— 机器摘要的帧级表示，selector 的输出、summary metric 的输入。
- `Budget(fraction: float)`，`0 < fraction ≤ 1` —— 预算占总帧数比例（0.15 = F1@15%）。**帧数取整规则（round/floor/ceil）刻意留到 M2 考古后定稿**，不提前猜。
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
- 生命周期：registry 里放的是**实例**；有状态模型在 `fit` 后持有参数。编排层（M4）在每次 run 前 deepcopy 实例，避免跨 run 状态泄漏。

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
| `gt_aggregation` | `"max" \| "mean"` = `"max"` | 跨标注聚合。TVSum 系论文惯用 max（对最好的标注对齐） |
| `segment_score_aggregation` | `"mean" \| "max"` = `"mean"` | 帧分 → 段分聚合（§2.3 决策） |
| `score_normalization` | `{method: none/min_max/z_score}` = `none` | 打分归一化；各源仓库差异大，M2 考古后给推荐默认 |
| `training` | `TrainingConfig` | epochs / lr / batch / device + `extra`（模型专属 JSON 参数，同样进 manifest，无处可藏） |

切分配置是带 `kind` 标签的 tagged union：

- `CanonicalSplits(name)` —— 使用已发表的具名切分（`tvsum-canonical-5` 等，M1 内置）；
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
- **内置组件（M1+ 的 tvsum/summe、M2 的 selector/指标、M3 基线）走同一机制**，在我们自己的 `pyproject.toml` 里声明，与第三方完全同构——这是「开闭原则」的机制保证。

## 6. 随机性与可复现

- 唯一种子来源：`EvalConfig.seed`（切分如需独立复用，用 `RandomSplits.seed` 覆盖）。模型训练、任何抽样都从这里派生；组件**不得**自带熵源（`random.random()`、未播种的 numpy/torch 全局态都算违规）。
- 评测结果必须能落到 run manifest：git SHA + 完整 `EvalConfig` + 数据 checksum（manifest 结构 M4 定稿）。

## 7. 与 dev-plan 签名的已批准偏差

1. **`Selector.select` 增加 keyword-only `aggregation` 参数**（dev-plan 原签名 `select(scores, shots, budget) -> Mask`）。理由：帧→段聚合改变数值，必须是显式协议参数（规则 4），放签名里由编排层传入，而非各 selector 自行其是。
2. **metrics 拆成两组** `SummaryMetric` / `RankMetric`（entry-point 组 `vsumbench.metrics` / `vsumbench.rank_metrics`，dev-plan 只列了单一 `.metrics` 组）。理由：F1 与 Kendall τ 输入类型不同（Mask vs Scores），分组让注册与分发都类型安全。
3. **新增 `vsumbench.selectors` 注册组**。理由：selector 与数据集/模型/指标同等可扩展，机制统一。

## 8. 待 M2 协议考古定稿（现在刻意留白）

以下是已知会影响数值、但**尚无依据钉死**的选择。每条定稿流程：进 `EvalConfig` 字段（或确认现有字段语义）→ `docs/protocols/*.md` 记录出处 → 补手算用例 / 对齐测试。

- TVSum `gtscore` → 用户摘要的构造方式（段聚合、预算、knapsack vs top-k）；
- budget 的帧数取整规则（round / floor / ceil）；
- gtscore 训练目标的归一化（VASNet 的 per-video min_max）与 `score_normalization` 的确切作用位置（训练目标 vs 选择前）；
- 特征帧率与 `n_frames` 的对齐（部分源仓库有 subsample）；
- SumMe fps 缺省的处理；
- F1@K 的多预算语义（GT 摘要固定 15% 还是随 K 同步）。

## 9. 组件测试准入（合并门槛）

- **数据集 adapter**：合成 fixture 单测（零网络、零真实数据）+ checksum 集成测试（`-m integration`）。
- **模型**：对齐测试（`tests/alignment/` 金色 fixture + 容差断言）；已知偏差先记录 `docs/protocols/` 再改代码。
- **指标**：手算用例（必须有，小数组人工算期望值）。
- **selector**：边界条件（budget 全选 / 极小、单段视频、并列分数、恰好整除）。

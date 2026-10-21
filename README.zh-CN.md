# vsumbench

[![English](https://img.shields.io/badge/README-English-blue)](README.md)
[![简体中文](https://img.shields.io/badge/README-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-orange)](README.zh-CN.md)

**视频摘要研究的统一实验框架。**

> 本文件与 [README.md](README.md) 内容同步维护。

视频摘要的评测碎片化已经十年了：每篇论文都自带一套数据集加载代码、一套指标实现、一套私有切分——搭建实验意味着复活一堆 Keras 时代、数据链接早已失效的仓库，而且论文之间的数字经常不可比。

`vsumbench` 给你的方法提供一个共享的、开箱即用的基底：带 checksum 校验下载的 canonical 数据集、一套显式的评测协议、以及让每个数字都可追溯的 run manifest。它是视频摘要社区的 `lm-evaluation-harness`——你带方法，我们提供实验环境：

```bash
pip install vsumbench

vsumm download tvsum
vsumm eval --model mine --dataset tvsum --splits tvsum-canonical-5
```

对于你自己的模型，只需要一个把视频映射为逐帧重要性分数的类：

```python
from vsumbench import Bench

bench = Bench(dataset="tvsum", splits="tvsum-canonical-5")
bench.register("mine", my_model)

bench.run()  # → 共享协议下的逐折 F1 + run manifest
```

其他方法的对比数字照社区惯例直接引自原论文；不同之处在于，你自己的数字现在是在一套有文档、可复现的统一协议下算出来的，而不是又一个私有变体。

## 一次评测是如何运行的

你只需提供**一样东西**：一个把视频映射为逐帧重要性分数的模型。其余全部由框架执行：

| 阶段 | 谁做 | 发生什么 |
|---|---|---|
| 获取视频 | 框架 | `vsumm download` 下载并校验 h5；loader 把 `Video` 交给你的模型 |
| 打分 | **你** | `Model.predict(video) -> Scores`，帧级 |
| 选段 | 框架 | 帧分 → 镜头分 → 预算内的 0/1 背包 → 布尔摘要掩码 |
| 测量 | 框架 | F1 对每份人工摘要、τ/ρ 对 `gtscore`；按 config 收缩、跨折聚合 |

### 你的模型拿到什么

一个 `Video` 携带评测所需的全部基底（以 TVSum `video_1` 为例给出真实形状）：

| 字段 | 形状（示例） | 含义 |
|---|---|---|
| `features` | `[707, 1024]` | GoogLeNet pool5 特征，每 15 帧一行 |
| `picks` | `[707]` | 每个 feature 行对应的原始帧下标 |
| `shots` | 71 个镜头 | 数据集自带的 KTS 边界 |
| `user_summaries` | 20 × `[10597]` | 逐人二值帧掩码（F1 的 ground truth） |
| `gt_score` | `[707]` | 数据集自带的聚合重要性分（训练目标；参考口径 rank 指标的 GT） |
| `n_frames` | 10597 | 原始帧空间长度 |

**没有原始帧**：eccv16 谱系惯例——因此也是全部已发表对比数字——建立在这些预提取特征之上。需要原始视频的方法目前不在此基底范围内。

### 模型契约

```python
import numpy as np
from vsumbench.models.score_space import expand_step_scores
from vsumbench.types import Scores


class MyModel:
    name = "mine"

    def fit(self, videos, config) -> None:
        # 收到 train split 的 Videos 和完整 EvalConfig。
        # 超参在 config.training；一切随机性必须从 config.seed
        # 派生（不得有隐藏熵源）。
        ...

    def predict(self, video) -> Scores:
        step_scores = my_scoring(video.features)  # [n_steps] —— 你的方法
        return Scores(
            video.video_id,
            expand_step_scores(step_scores, video.picks, video.n_frames),  # [n_frames]
        )
```

两条规则：`predict` 必须返回**帧级**分数——步空间打分的模型用共享的 `expand_step_scores`（谱系的前向填充）展开，让所有模型在同一个空间会合；模型对固定 config 必须确定性。

底层机制上，选段是对镜头的忠实 0/1 背包：镜头价值 = 其帧分数的均值（或最大值），重量 = 镜头帧数，容量 = `floor(n_frames * budget)`；被选中镜头的全部帧构成机器摘要。这些组件都是公开的，你也可以直接驱动它们：

```python
from vsumbench.metrics.f1 import F1_METRIC
from vsumbench.models.score_space import expand_step_scores
from vsumbench.registry import DATASETS, SELECTORS
from vsumbench.data.download import cache_dir
from vsumbench.types import Budget, Scores
import numpy as np

video = DATASETS.get("tvsum").load(cache_dir())[0]
frame = expand_step_scores(my_scoring(video.features), video.picks, video.n_frames)
mask = SELECTORS.get("knapsack").select(
    Scores(video.video_id, frame), video.shots, Budget(0.15), aggregation="mean"
)
f1 = float(np.max(F1_METRIC.score(mask, video)))  # 逐人 F1 → gt_aggregation
```

### 用 Python API 跑

```python
from vsumbench import Bench

bench = Bench(
    dataset="tvsum",
    splits="tvsum-canonical-5",  # 谱系标准
    metrics=("f1", "kendall_tau_gtscore", "spearman_rho_gtscore"),
    budgets=(0.15,),  # 谱系报告 15%
)
bench.register("mine", MyModel())  # 注册并选用，一步完成
result = bench.run()  # → runs/<时间戳>-tvsum-mine-knapsack/
print(result.summary["f1"]["0.15"])  # {'mean': ..., 'std': ..., 'max': ..., 'n_folds': 5}
```

每折 bench 会 deepcopy 你的模型（跨折无泄漏）、在 train id 上 fit、对 test id predict、逐预算选段、计算逐人指标并按 `gt_aggregation` 收缩。

### 用 CLI 跑

`bench.register` 只在本进程有效。要让 `vsumm eval` 认识你的模型，在你自己的包里声明——这一行就是全部接入工作，零 vsumbench 源码改动：

```toml
[project.entry-points."vsumbench.models"]
mine = "my_pkg.model:MY_MODEL"   # 是实例，不是类
```

```bash
vsumm download tvsum
vsumm eval --dataset tvsum --model mine --splits tvsum-canonical-5 \
           --metrics f1,kendall_tau_gtscore,spearman_rho_gtscore --budgets 0.15
vsumm report runs/               # 扫描 manifest → markdown 对比表
```

数据落在 `$VSUM_CACHE_DIR`（默认 `~/.cache/vsumbench`），经 checksum 校验；`--data-root` 可指向本地已有副本离线使用。

### 让数字可比

- `gt_aggregation` 默认 `max`（eccv16 原论文口径）。KaiyangZhou/VASNet 谱系代码对 TVSum 硬编码 `avg`——与那批论文的表对比时用 `--gt-aggregation mean` 切换。
- 优先 canonical 5 折切分与显式预算；两者都会记入 manifest。
- manifest 记录 git SHA（和 dirty 标志）、完整 config 与数据 checksum——在干净工作树上跑，每个数字才能回答「哪份代码、哪些协议参数、哪份数据」。

## 设计原则

- **忠实协议。** 评测流水线（分数展开、镜头选择、预算、F1）逐字转录自参考实现，并在真实数据上与它们比对验证（见 `tests/alignment/`）。一切偏差都有文档记录，不藏私。
- **显式评测。** 有争议的协议选择——帧选择策略、预算、GT 聚合、切分、归一化——都是带推荐默认值的显式参数。分歧变成可配置，而不是静默站队。
- **最小依赖。** 核心只依赖 `numpy` + `h5py`；训练框架全部放可选 extras。
- **数据永不打包。** `vsumm download` 从原始来源下载并校验 checksum。我们分发 loader，不分发数据。

## 范围

| 层 | 首发（v0.1） | 之后 |
|---|---|---|
| 数据集 | TVSum、SumMe、TVSum+SumMe | OVP、VTW、QFVS |
| 指标 | F1@{10,15,30,50}%（shot-based / knapsack 选择），Kendall τ-b 与 Spearman ρ 对数据集 `gtscore`（Zhang et al. 2016 参考口径） | 逐人 τ/ρ（需原始 ydata 标注）、扩展协议 |
| 模型 | 你自己的，经 registry 接入（另有无学习 `uniform` 参照供冒烟测试） | 已发表方法的社区移植 |
| 切分 | TVSum & SumMe canonical 5 折、种子化随机切分 | 社区切分 |

## 状态

🚧 **早期开发。** 数据层（`vsumm download`）、协议层（selectors、F1、对 `gtscore` 的参考口径 τ/ρ、对齐测试）与编排层（`vsumm eval` / `vsumm report`、run manifest、`Bench` API）均已实现——进度见 [docs/design/dev-plan.md](docs/design/dev-plan.md)（中文）。TVSum/SumMe 谱系的协议考古在 [docs/protocols](docs/protocols/)。v0.1 剩余：PyPI 发布。

## 参与贡献

新模型经 registry 接入——完整指南（开发环境、各组件扩展契约、测试要求、如何带对齐测试移植已发表方法）见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

[MIT](LICENSE)

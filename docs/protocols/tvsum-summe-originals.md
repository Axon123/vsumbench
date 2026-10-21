# TVSum / SumMe 原始论文协议与 eccv16 打包的关系

- **状态**：M2 记录（2026-10-13）；2026-10-21 增补 `video_name` 不对称实测与 gtscore 搬运现状（随契约 v1.4）。论文级结论（标注自原文），代码级结论仅覆盖 h5 打包链条。
- **要点**：v0.1 评测的"TVSum/SumMe"是 **eccv16 打包形态**（GoogLeNet pool5 特征 + 统一 h5 schema），不是两篇原始论文自己的评测协议。本文记录两套协议的差异，防止"复现数据集论文数字"与"复现模型论文数字"两个目标被混为一谈。

## 1. 原始论文的评测协议（论文口径）

### SumMe（Gygli et al., ECCV 2014, "Creating Summaries from User Videos"）

- 标注：15–18 人/视频的**帧级重要性分**（0–1）+ 逐人摘要（关键帧/区间）。
- 机器摘要构造：对视频做 KTS 分割，按镜头分数贪心选镜头到预算。
- F1 计算：机器摘要**与每份人工摘要**比；但人工摘要先被转换到同一镜头选择空间，再比帧重叠。报告预算 15% 与 30% 两档。
- 聚合：多标注取**平均**（部分实验 max）。

### TVSum（Song et al., CVPR 2015, "TVSum: Detecting Important Videos for Video Summarization"）

- 标注：20 人/视频的**帧级重要性分（1–5）**，摘要由"重要性分 + 预算"在线导出（web 标注工具内做 knapsack 式选择），不是独立的逐人帧掩码。h5 里的 `user_summary` 是 ydata 按各人分数在 15% 预算下生成的派生量。
- 排序相关性：论文以 Kendall τ / Spearman ρ 报告重要性预测（eccv16 论文的 τ 惯例即源于此）。
- F1：对每份派生摘要计算，跨标注聚合论文正文用 **max**（oracle-annotator 口径）。

## 2. eccv16 打包（de-facto 评测基底）

Zhang et al.（ECCV 2016, LSTM 摘要）的预处理成为后续全部模型论文的事实标准：

- 特征：GoogLeNet pool5，1024 维，**每 15 原始帧 1 步**（`picks = range(0, n_frames, 15)`）。
- `gtscore`：跨人聚合重要性（TVSum 为 20 人均值的归一化、SumMe 为逐人均值聚合；实测均非 per-video min-max，见 eccv16-summary-protocol.md §0）。
- `user_summary`：逐人帧级二值摘要（TVSum 为上述派生量，SumMe 为人工掩码）。
- `change_points`：KTS 镜头边界，闭区间端点完美铺贴。
- `video_name`：**仅 SumMe 有**（25/25 组，存原始视频名如 `Air_Force_One`）；TVSum 50 组均无此键（2026-10-21 真实文件逐组核对）。
- 分发链：ECCV16 预处理 → KaiyangZhou/pytorch-vsumm-reinforce（datasets.tar.gz / Google Drive）→ e-apostolidis 各仓库内嵌 rar（即 M1 `data/sources.py` 的镜像表与 checksum 对象）。

## 3. 对 vsumbench 的含义

| 差异点 | 处理 |
|---|---|
| 原始论文协议（人工摘要转镜头空间、SumMe 30% 档、TVSum 派生摘要的在线生成） | **v0.1 不实现**。README/dev-plan 已将 v0.1 范围钉在 eccv16 打包 + 模型论文协议上；原始协议作为未来数据集扩展记录在案 |
| `gtscore` 是派生聚合量 | 原样搬运进 `Video.gt_score`（契约 v1.4，2026-10-21 起；秩相关对单调变换不变，gtscore 参考口径 τ/ρ 可安全用它，见 eccv16-summary-protocol.md §4）；逐人版 τ/ρ 仍等 ydata 接入（v0.1 后） |
| ydata 逐人分数与 h5 的 join key | 不对称：SumMe 文件内自带 `video_name`（原始视频名）可直接对回原始标注；TVSum 无此键，接入时需外部 `video_N ↔ 原始 id` 对应表（如原论文发布包的文件顺序，须逐视频验证）。loader 现均不加载该键（provenance） |
| TVSum `user_summary` 的"预算生成"语义 | 用户摘要即数据（各人预算由标注工具固定）；评测时机器预算由 `EvalConfig.budgets` 显式给出，二者独立 |
| fps | 两 h5 均无 fps（SumMe 仅 `v_length`）；秒类协议留空不猜（core-contracts §8） |

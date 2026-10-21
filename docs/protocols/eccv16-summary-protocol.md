# eccv16 谱系摘要评测协议考古

- **状态**：M2 定稿（2026-10-13）。所有结论逐条给出源仓库证据；引用代码为逐字转录（py3 化）。
- **范围**：e-apostolidis 家族（PGL-SUM、CA-SUM、SUM-GAN-AAE/sl）与 KaiyangZhou 谱系（pytorch-vsumm-reinforce，即 DR-DSN 官方实现；ok1zjf/VASNet fork 的评测文件头部明确注明 courtesy of KaiyangZhou）。原版 SUM-GAN TF 仓库（j-min/Adversarial_Video_Summary）的 `evaluate` 只导出分数 JSON，F1 由本谱系线下脚本计算。
- **作用**：本文档是 `EvalConfig` 中 `budget_rounding` / `step_score_expansion` / `gt_aggregation` / `segment_score_aggregation` 推荐默认值的依据；对齐测试 `tests/alignment/test_eccv16_protocol_alignment.py` 以逐字转录的参考实现守护本协议。

## 0. 数据基底

所有仓库评测同一对 canonical h5：`eccv16_dataset_{tvsum,summe}_google_pool5.h5`（ECCV16 特征 → KaiyangZhou 分发 → e-apostolidis 仓库内嵌）。双帧空间事实（M1 逐视频验证 75/75）：`features`/`gtscore` 在步空间（`picks = range(0, n_frames, 15)`），`user_summary`/`change_points`/`n_frames` 在原始帧空间。`gtscore` 为跨人聚合量：实测 **并非** per-video min-max（TVSum 50 视频中 max==1 的为 0 个、min==0 的仅 4 个；SumMe 25 个中 max==1 仅 3 个），全局值域 [0, 1]。

## 1. 共享评测流水线（六仓库一致）

以模型输出的**步空间分数** `ypred`（长度 = len(picks)）为输入：

1. **展开到帧空间（forward fill）**：`frame_scores = zeros(n_frames, float32)`；`positions = picks + [n_frames]`；区间 `[positions[i], positions[i+1])` 填 `ypred[i]`。**尾区间 `[picks[-1], n_frames)` 拿到最后一个分数**（"前向填充、尾部延续"）。参考代码中 `if i == len(ypred): 填 0` 分支对合法输入是死代码，仅防御 `len(picks) > len(ypred)` 的长度错配。证据：PGL-SUM `evaluation/generate_summary.py` L20–31 与 DR-DSN `vsum_tools.py` `generate_summary` L24–35 逐字相同。
2. **镜头分数 = 帧分数均值**：`float(frame_scores[cps[i,0] : cps[i,1]+1].mean())`，float32 切片上的 `np.mean`（逐镜头循环）。无任何仓库使用 max。
3. **预算取整 = floor**：`int(n_frames * proportion)`（PGL-SUM）或 `int(math.floor(n_frames * proportion))`（DR-DSN），二者等价。**proportion 硬编码 0.15**，全部仓库只评 F1@15%。
4. **选择（二选一）**：
   - `knapsack`（默认）：经典 0/1 背包 DP，value = 镜头均值（**不乘长度**），weight = 镜头帧数，容量 = floor(15% · n_frames)。并列语义（两族实现一致）：**取与不取价值相等时不取**；回溯从后往前，效果上并列时保留**靠前**的镜头。PGL-SUM/CA-SUM/SUM-GAN 用 Bhavya Jain 的 float64 表实现；DR-DSN `knapsack_dp` 是 float32 表 + strict `>` 的 keep 矩阵（并列同样不取）。VASNet fork（Jiri Fajtl 修改）换成 OR-Tools 求解器，**并列解不可控**——这是 MiRA/SummaryNet 谱系的已知偏差源。
   - `rank`/greedy（仅 DR-DSN/VASNet 谱系）：`order = np.argsort(seg_score)[::-1]`（降序；稳定升序反转后，**并列时靠后的镜头先被访问**）；仅当 `total_len + len < limit`（**严格小于**）才收——预算永远填不满，甚至 limit ≤ 最短镜头时结果为空。
5. **机器摘要 = 选中镜头的帧并集**（原始帧空间，长度 n_frames）。
6. **F1（逐人）**：`overlap = S ∧ G`，precision = overlap/|S|，recall = overlap/|G|，F1 = 2PR/(P+R)。对机器摘要过短/过长有 pad/truncate（DR-DSN 谱系）；除零处理分两族：DR-DSN/VASNet 用 `+1e-8` 分母，PGL-SUM 族用精确整除（空摘要会 ZeroDivisionError）且**结果 ×100**。
7. **跨标注聚合**：`eval_metric = 'avg' if tvsum else 'max'`（VASNet/DR-DSN main.py 硬编码；PGL-SUM/CA-SUM 的 `evaluate_exp.sh` 同样 SumMe→max、TVSum→avg）。**注意**：eccv16 原论文（Zhang et al. 2016）对两数据集均用 max——这是跨论文复现差异的最大来源之一（约 1–3 个 F1 点）。
8. **数据集级 = 测试视频均值**（`np.mean(all_f_scores)`）。
9. **epoch/model 选择**：`choose_best_epoch.py` 按**测试集 F1 最高**的 epoch 报告（PGL-SUM/CA-SUM；其余谱系同样在测试集上挑最优 epoch）。这是谱系公认的协议瑕疵（测试集窥视），vsumbench 显式暴露为编排参数而非隐藏（M4 定稿）。

## 2. 各仓库差异表

| 选择点 | PGL-SUM / CA-SUM / SUM-GAN-AAE/sl | DR-DSN (pytorch-vsumm-reinforce) | VASNet fork (ok1zjf, MiRA 系修改) |
|---|---|---|---|
| knapsack 实现 | float64 表（Bhavya Jain） | `knapsack_dp`，float32 表 | OR-Tools（并列不可控） |
| greedy (`rank`) | 无 | 有 | 有 |
| F1 量纲 | ×100 | 比值 [0,1] | 比值 [0,1] |
| F1 除零 | ZeroDivisionError（空摘要） | 分母 +1e-8 | 分母 +1e-8 |
| gtscore 训练目标 | 原样使用 | 原样使用 | **再 per-video min-max**（`target -= min; target /= max`，main.py L236–237；因 h5 gtscore 并非 min-max，这不是幂等操作） |
| 代码语言 | py3 | py2（`xrange`） | py3 |

## 3. 落入 vsumbench 的方式

| 协议事实 | EvalConfig 字段 / 组件 | 默认值 |
|---|---|---|
| 展开规则（前向填充、尾部延续） | `step_score_expansion = "forward_fill"`（`zero_fill` 为变体选项）；实现：`vsumbench.models.score_space.expand_step_scores` | forward_fill |
| 预算 floor | `budget_rounding = "floor"`；`Budget.frame_limit()` 按 `n_frames * fraction` 先乘后取整复刻浮点行为 | floor |
| 镜头分数 = 帧均值 | `segment_score_aggregation = "mean"`（max 为本工具包的显式泛化，无仓库使用） | mean |
| knapsack / greedy | selectors registry：`knapsack` / `greedy`，并列语义逐字复刻（并列不取、回溯偏前；greedy 并列偏后 + 严格小于） | — |
| 逐人 F1 | metrics registry：`f1`，比值 [0,1]，空摘要记 0（对 PGL-SUM 崩溃与 DR-DSN epsilon 的**已记录偏差**） | — |
| 跨标注 max/avg | `gt_aggregation`（推荐默认 max = eccv16 原论文口径；复现 KaiyangZhou/e-apostolidis 论文表值时 TVSum 需切 mean） | max |
| F1@K | `budgets` 元组；机器摘要预算随 K 变化，GT 摘要原样不动（谱系一致：GT 是人工标注，不存在"GT 预算"） | — |
| score_normalization | 谱系在**选择前不归一化**预测分；VASNet 的 min-max 只作用于**训练目标**。字段保留给 M3 模型侧消费，作用位置以此为准 | none |

## 4. Kendall τ / Spearman ρ 的出处（非本谱系）

六个仓库**均未**计算排序相关指标。τ 进入领域来自 eccv16 原论文（Zhang et al. 2016，Table 报告 τ 与 Spearman ρ），在**步空间**上比较预测分与 h5 `gtscore`。vsumbench 落地两条口径（`metrics/rank.py`，均 tie-corrected、取 scipy/MATLAB 口径：τ-b 与 average-rank Pearson ρ，NaN 约定一致）：

- **gtscore 参考口径**（`kendall_tau_gtscore` / `spearman_rho_gtscore`，2026-10-21 落地）：直接对 `gtscore` 在步空间计算，即 Zhang 口径；预测帧分在 `picks` 处采样。秩相关对单调变换不变，故 gtscore 上游聚合口径未文档化**不影响数值**（对 F1 类重叠指标则不成立）。tvsum/summe 上今日可用。
- **逐人分数版**（`kendall_tau`）：对每份 `user_scores` 各算一个；canonical h5 无逐人重要性分，等原始 ydata 标注接入（v0.1 后，见 core-contracts §2.2）。

## 5. 考古文件快照

fetching 日期 2026-10-13，本地快照逐字比对：

- `e-apostolidis/PGL-SUM@master`：`evaluation/{generate_summary,knapsack_implementation,evaluation_metrics,compute_fscores}.py`、`model/data_loader.py`、`evaluation/evaluate_exp.sh`
- `e-apostolidis/CA-SUM@main`：`evaluation/{generate_summary,evaluation_metrics}.py`（与 PGL-SUM 仅注释格式差异）
- `e-apostolidis/SUM-GAN-AAE@master` / `SUM-GAN-sl@master`：`evaluation/`（同 PGL-SUM）
- `KaiyangZhou/pytorch-vsumm-reinforce@master`（AAAI'18 Zhou/Qiao/Xiang，DR-DSN）：`vsum_tools.py`、`knapsack.py`、`main.py`、`README.md`
- `ok1zjf/VASNet@master`（原 e-apostolidis/VASNet 已删库，此 fork 保留）：`vsum_tools.py`、`main.py`
- `j-min/Adversarial_Video_Summary@master`（原版 SUM-GAN TF）：`solver.py`

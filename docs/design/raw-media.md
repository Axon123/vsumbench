# 原始视频旁路：官方来源、`video_id` 映射与本地解析

- **状态**：2026-11-02 落地（非里程碑增量）
- **结论**：评测基底仍是 eccv16 h5 特征与既有 `Video` / `EvalConfig` 契约；原始视频是**可选旁路**，由 `vsumbench.data.media` 按 h5 的 `video_id` 解析，绝不按目录顺序对齐。

## 1. 范围与不变量

`vsumbench` 的数值协议（选择、预算、F1、τ/ρ）只依赖 h5 里的特征、`picks`、镜头边界与标注。需要像素的方法（端到端视频模型、可视化、抽帧）可以在同一套 `video_id` 上额外取原始视频，但这条路径：

- 不进入 `Video` 数据结构，不改变 `Model.predict(video) -> Scores` 契约；
- 不进入 `EvalConfig`，不改变任何评测数值；
- 不把视频文件打进 git，也不在仓库里分发；
- 不扫描、不排序目录来推断身份。

## 2. 官方来源

仓库只提交**映射表与 checksum**（`src/vsumbench/data/media_sources.py`），视频由使用者从官方发布包下载。PyPI 发布的 wheel/sdist 同样只含代码、映射与 checksum，不包含任何视频、标注或 eccv16 h5 文件。

| 数据集 | 官方页面 | 官方归档 | 大小 | 归档 sha256 |
|---|---|---|---|---|
| TVSum | <https://github.com/yalesong/tvsum> | <https://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz> | 671,779,858 B（约 0.67 GB） | `407d340bcd06fdc6d17374ebe6760b4a96816bcace228559c8283d9fb2520dea` |
| SumMe | <https://data.vision.ee.ethz.ch/cvl/SumMe/> | <https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip> | 2,396,688,224 B（约 2.4 GB） | `42aac7668888ae30885abdb61bb4a10a2e82543695354a54de82e48868b8fb05` |

归档内部结构：

- **SumMe**：`videos/<原始视频名>.mp4`（同目录也有 `.webm`；本仓库固定选 mp4）。
- **TVSum**：外层 tgz 中的 `ydata-tvsum50-v1_1/ydata-tvsum50-video.zip`，内层 zip 的成员为 `video/<YouTube id>.mp4`。

两个归档都是**整包分发**，没有稳定的单视频下载端点；因此安装任意一个视频也要先取得整个官方归档。

## 3. `video_id` 映射的建立与核验

h5 的 `video_id`（`video_1` … `video_50` / `video_1` … `video_25`）是唯一 join key。原始文件名只作为 provenance，不参与本地寻址。

### TVSum

- 官方元数据 `ydata-tvsum50-info.tsv`（在 `ydata-tvsum50-data.zip` 内）按 `video_id`（YouTube id）列出 50 个视频；其**行序**与 eccv16 h5 的 `video_1` … `video_50` 一致。
- 该行序不是猜的，逐视频做了两项核验：
  1. h5 的 `n_frames` 与 TSV 的 `length` 在 24/25/30 fps 下吻合（50/50，推算 fps 全部落在 23.98–30.18）；
  2. `ydata-tvsum50-anno.tsv` 中对应 YouTube id 的 20 份逐帧分数长度恰好等于 h5 的 `n_frames`（50/50），且按 15 帧步聚合后的均值与 h5 `gtscore` 的 Spearman ρ 最低 0.875、中位数 0.984（聚合细节不同导致不恒为 1，但错配不会呈现这种一致性）。
- 例：`video_1 -> AwmHb44_ouw`，`video_50 -> -esJrBWj2d8`。映射表同时保留 YouTube URL，仅用于追溯；下载器**不访问 YouTube**。

### SumMe

- h5 每个组自带 `video_name`（如 `Air_Force_One`）；与归档中的 `videos/<video_name>.mp4` 逐条精确匹配（25/25），无需推断顺序。
- 例：`video_1 -> videos/Air_Force_One.mp4`，`video_10 -> videos/Excavators river crossing.mp4`（原始文件名可含空格，本地名仍是 `video_10.mp4`）。

### 映射表内容

`MEDIA_ENTRIES[dataset][video_id]` 记录：

| 字段 | 含义 |
|---|---|
| `filename` | 本地 canonical basename，恒为 `<video_id>.mp4` |
| `archive_member` | 归档（或内层 zip）中的精确成员路径 |
| `original_name` | TVSum 的 YouTube id / SumMe 的 `video_name`，仅 provenance |
| `source_url` | TVSum 的 YouTube URL，仅 provenance（SumMe 为 `None`） |
| `size` / `sha256` | 该成员解压后字节数 / 内容哈希 |

全部 75 条 `size` 与 `sha256` 已对 2026-11-02 的官方归档逐条重算核验；归档本身也以 `archive_size` + `archive_sha256` 固定。官方包若更新，必须重新生成映射并同步本文档，不能沿用旧哈希。

## 4. 本地布局与解析

```
<media_root>/
  <dataset>/
    video_1.mp4
    video_2.mp4
    ...
  .archives/            # 仅 --keep-archive 或手工放置时存在
    SumMe.zip
    tvsum50_ver_1_1.tgz
```

`media_root` 的优先级：

1. API 的 `root=` / CLI 的 `--media-root`
2. 环境变量 `VSUM_MEDIA_DIR`
3. `$VSUM_CACHE_DIR/media`（`VSUM_CACHE_DIR` 缺省为 `~/.cache/vsumbench`）

解析只认映射表里的 `<video_id>.mp4`。目录里多余的文件会被 `media_index()` 忽略，绝不可能因为“多了/少了某个文件”而改变某个 id 指向的视频。

## 5. Python API 与 CLI

```python
from vsumbench.data.media import (
    download_media,
    expected_media_path,
    media_dir,
    media_entry,
    media_index,
    media_path,
    media_source,
)
```

| API | 作用 |
|---|---|
| `media_dir(override=None)` | 解析 media root |
| `media_source(dataset)` | 官方归档描述（URL、大小、sha256、格式、内层归档） |
| `media_entry(dataset, video_id)` | 单条映射记录 |
| `expected_media_path(dataset, video_id, root=None)` | canonical 路径（不检查是否存在） |
| `media_path(dataset, video_id, root=None, verify_sha256=False)` | 解析并校验本地文件；缺失/损坏抛 `MediaNotAvailableError` |
| `media_index(dataset, root=None, verify_sha256=False)` | 返回已安装视频的 `{video_id: path}` |
| `download_media(dataset, root=None, video_ids=None, report=None, verify_sha256=True, keep_archive=False)` | 下载官方归档并安装指定视频 |

```bash
# 安装单个视频（仍会下载完整归档）
vsumm download-media tvsum --video-id video_1

# 安装全部映射视频；保留校验过的归档供后续复用
vsumm download-media summe --keep-archive

# 解析；文件缺失或哈希不符时非零退出
vsumm media-path tvsum video_1 --verify-sha256
```

`--video-id` 可重复；省略表示该数据集全部映射视频。`--media-root` 对两个子命令都可用。

`download_media` 的行为：

1. 已存在且通过大小（默认还校验 sha256）的本地文件直接跳过；损坏文件会被重新安装覆盖。
2. 若 `<media_root>/.archives/<archive_filename>` 存在且校验通过则直接复用，否则下载官方归档；下载完整包后校验大小与 sha256。
3. 按 `archive_member` **精确成员名**查找（不做“第几个文件”式定位），校验成员大小/哈希后先写入临时目录。
4. 所有请求的视频都 staging 成功后，才逐个 `os.replace` 到 canonical 路径。staging 失败不会留下任何目标文件；替换阶段若中途失败，已落地的文件都是正确的，重跑会补齐其余。
5. 未指定 `keep_archive=True` 时删除临时归档；指定时归档落到 `.archives/` 并在后续调用中复用。

TVSum 需要先流式抽出 670 MB 的内层视频 zip；SumMe 直接按 zip 中央目录定位成员。

## 6. 缺失、损坏与离线使用

`media_path` 的错误文本会给出：预期的本地路径、应安装的精确归档成员、官方归档 URL 与归档 sha256、可直接执行的安装命令，并明确提示“按 `video_id` 寻址，绝不按排序后的目录对齐”。CLI 对“文件不可用”返回 1，对未知 dataset / `video_id` 等用法错误返回 2。

离线场景有三条路：

- 预先安装好 `<media_root>/<dataset>/<video_id>.mp4`（内容须与官方成员一致；`--verify-sha256` 可确认）；
- 把官方归档放到 `<media_root>/.archives/<archive_filename>`，`download_media` 会校验后复用，无需网络；
- 直接使用已有的 `media_index()` 结果，不触发任何下载。

## 7. 许可

仓库只分发映射、URL 与 checksum，不分发视频。使用者仍需自行遵守数据条款：

- **TVSum**：官方 README 称视频为 CC-BY 3.0；但归档内附带的 Yahoo! Webscope ReadMe 将数据限定为学术研究、要求 Data Sharing Agreement，并禁止整体或部分再分发/公开。两处表述不一致，保守做法是仅用于学术研究且不再分发，具体以使用者取得的官方条款为准。
- **SumMe**：归档内 `README.txt` 明确视频仅限非商业研究使用（视频版权不完全由数据集作者持有）；标注数据可用于研究。

## 8. 限制与后续

- **整包下载**：装一个视频也要 0.67 GB（TVSum）或 2.4 GB（SumMe）；没有逐视频官方端点，也不做 YouTube 抓取/`yt-dlp` 回退（ToS 与链接腐坏风险）。
- **无断点续传**：下载失败会重试并从头开始；可用 `--keep-archive` 或手工放置归档减少重复下载。
- **格式固定**：SumMe 只取 mp4，不转码、不抽帧；消费方自行用 ffmpeg/OpenCV 解码。
- **链接腐坏**：官方 URL 与哈希固定的是当前发布包；归档被替换时必须更新 `media_sources.py`、本文档和测试。仓库不承诺在官方源消失后继续可下载。
- **run manifest 不记录媒体哈希**：manifest 仍只记录 h5 评测数据的 checksum。原始视频是评测之外的旁路；依赖像素的方法若需严格追溯，应自行记录 `media_path(..., verify_sha256=True)` 的路径/哈希。等原始帧方法成为一等公民时再考虑纳入 manifest。
- **不做身份猜测**：没有“按文件名相似度匹配”“按目录第 i 个文件对齐”之类的兜底；缺失就是显式错误。

## 9. 测试与核验

- `tests/unit/test_media.py`：映射完整性（50/25 连续 id）、字段良构、关键条目抽查、media root 优先级、顺序无关解析、缺失错误可执行、大小/哈希损坏、合成 zip 与 tar.gz（含内层 zip）、本地 HTTP 下载、归档校验失败不落盘、`--keep-archive` 复用、CLI 成功/失败路径。
- 2026-11-02 对真实官方归档做了一次性核验：75/75 条 `size` + `sha256` 全等；并用真实归档安装 TVSum/SumMe 各一个视频后经 `media_path(verify_sha256=True)` 解析成功。
- 改动不触碰 `selection` / `metrics` / `Video`，协议层对齐测试不受影响；三门槛（ruff / mypy strict / pytest）全绿。

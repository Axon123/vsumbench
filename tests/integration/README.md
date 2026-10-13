# 集成测试（tests/integration）

真实数据端到端测试。默认被 pytest 配置排除（`-m "not integration"`），本地显式运行：

```bash
vsumm download tvsum   # 或 summe；也可手动放置 h5 到 $VSUM_CACHE_DIR
pytest -m integration
```

数据不在缓存目录时自动 skip（不失败），CI 因此保持轻量。当前覆盖：canonical 文件
checksum 校验、TVSum/SumMe 全量加载（视频数 / 标注人数 / picks=每 15 帧 / 1024 维特征）、
canonical 5-fold 切分与加载 id 集合的精确匹配。

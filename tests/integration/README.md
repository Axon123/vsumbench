# 集成测试（tests/integration）

真实数据端到端测试（dev-plan M1 起填充）。默认被 pytest 配置排除（`-m "not integration"`），
本地显式运行：

```bash
pytest -m integration
```

需要已通过 `vsumm download` 下载（或本地路径覆盖）的真实数据。M0 阶段此目录为空。

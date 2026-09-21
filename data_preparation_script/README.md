# 作者历史预处理脚本

本目录原有 CLAM、Quilt、固定 GPU 0~3 脚本保留作审计参考，不是论文优先复现入口。

新流程由 `../scripts/run_inference.py` 串联 `trident_coords.py`、`data_processing/preprocess.py`，分别使用独立 Trident 环境、Patho-R1 通用/问题描述和 PLIP 特征；缓存有内容和配置校验。

请从 `../docs/REPRODUCTION.md` 执行 `scripts/build_manifest.py` 和 `scripts/submit_runs.py`。两个历史 multi_*.sh 已改为显式失败提示，避免误启动旧模型/固定 GPU 配置。原脚本可在作者提交 `8c2abee2cd18ca2e2c839742c9296bd8a41738c5` 中查看。

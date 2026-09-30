# reproduce_v3：输入补全与固定对照数据

本次依据线程 `01a0ee28-6967-7352-a1cc-5a6f7b7fbfcb` 的最终 plan，并以本次补充为准：
Missing 同时接收 Predict 与 Reflect，BCNB 固定分层抽取 1,000 题；CPTAC 最初只准备官方
VQA 标注，随后按用户指令补充下载其对应的 240 张官方 TCIA SVS。
未提交 GPU validation、smoke 或 benchmark；CPU 模拟回归不代表真实模型验收或准确率改善。

## 三个版本

| 项目 | v1 | v2 | v3 |
| --- | --- | --- | --- |
| 配置 | reproduce_v1.yaml（默认） | reproduce_v2.yaml | reproduce_v3.yaml |
| Perceptor System | 本地短版 | 论文长版 | 与 v2 相同 |
| Perceptor User | 区域元数据 | 基础描述指令 | 区域元数据 + v2 基础指令 |
| Zoom User | 加 missing_info | 无 missing_info | 非空时追加实际 missing_info |
| Missing 输入 | Predict + Reflect | 无前序输出 | 本轮 answer、thinking_steps 和 Reflect JSON |
| Predict / Reflect / Final / Summary / Retry | 本地旧版 | 论文提示词及本地包装 | 与 v2 相同 |
| token 上限 | 基础预算 | 2 倍 | 与 v2 相同 |
| invalid 恢复 | 重试 | 终态 | 终态 |
| 描述缓存 | data/processed | data/processed/paper-prompts-nonthinking-2x-v1/bf16 | data/processed/reproduce-v3/bf16 |

v3 协议为 `paper-prompts-context-restored-v3`。继承 v2 backend，仅覆盖两个消息构造接口。
区域元数据来自真实 `Region.metadata()`，含原图坐标、区域宽高、尺度和尺度来源。
实际发送的 User 与日志一致；Zoom 使用原始 Missing 文本；Predict 字段及 Reflect JSON 通过
`ensure_ascii=False` 序列化，保留引号、换行和中文，不增加虚构的反思理由。
Reflect=Yes 时仍跳过 Missing；新增内容及格式重试均进入 v2 的完整上下文预算，必要时只压缩描述。

论文依据：[PathAgent](https://arxiv.org/abs/2511.17052v1) 补充材料 Model Details 中说明
Missing 接收前两步的 responses；Prompt 节规定 Reflect 输出 `sufficient`。
本地原文位于 `runs/analysis/bcnb_pipeline_audit_20260929/paper_source/sec/X_suppl.tex`。
User 包装的具体措辞是本地适配，prompt manifest 明确标注为非论文逐字模板。

模型、权重 revision、BF16/FP32、checkpoint 采样、overlap=0、MPP=0.5 假设、首轮 10%、
后续 5%、最大 5 轮、单个 Zoom 子区后 Final、Perceptor 原始 `<think>` 保留均与 v2 一致。
精确字符串 `"10"`、`"20"`、`"40"` 的整数转换沿用 v2。
默认入口和既有 BCNB 双流程 launcher 仍为 v1、v1/v2；未扩展成三流程自动提交。
缓存及运行身份包含协议、prompt manifest 与源码 hash，不能混用 v2 缓存或续接 v2 输出目录。

## 固定 BCNB 样本

位置：`data/raw/bcnb_1000_sample/`。固定 seed=128，从 7,274 题按
`Task × 真实答案选项文本` 比例分层，各层至少一题；不能按随机排列的选项字母分层。
配额按比例取整、最少一题，再按最大配额缺口分配余数；层内按 SHA256(seed, ID) 排序。
`scripts/prepare_bcnb_sample.py` 可重建相同抽样，已有不同样本时拒绝覆盖。

| Task | 题数 |
| --- | ---: |
| Tumor | 145 |
| ER | 145 |
| PR | 146 |
| HER2 | 146 |
| HER2 Expression | 145 |
| Histological grading | 127 |
| Molecular subtype | 146 |
| 总计 | 1,000 |

全部 20 个分层均覆盖，共 667 张切片。`sampling.json` 固定原始标注 SHA256、样本 SHA256、
题目 ID、种子和各层配额；CSV 保留原题、原选项及标签。`images/` 是指向已有原始切片的相对链接。
`manifest.json` 保存固定题单，但 `images_verified=false`：本次没有进行图像/倍率验证，
推理入口会阻止直接使用该未验证 manifest。标签只供抽样和计分，不进入模型消息。

未来获准检查数据后，可从固定 CSV 生成单独的可运行 manifest（本次未执行）：

```bash
conda activate pathagent
python scripts/build_manifest.py bcnb \
  --dataset-config configs/datasets/bcnb_1000_sample.yaml \
  --output data/manifests/bcnb_1000_sample/full.json
```

不会重新抽样，也不覆盖原全量 manifest。之后各版本均使用这个题单，v3 选择
`--config configs/reproduce_v3.yaml` 并使用全新输出目录。
该子集来自已分析过的 benchmark，是开发性对照数据，不是未观察过的独立测试集。

## CPTAC 标注

位置：`data/raw/cptac/SlideBench-VQA-CPTAC.csv`。
来源：[官方 SlideChat 数据集固定 revision](https://huggingface.co/datasets/General-Medical-AI/SlideChat/tree/aa1aa40b78564b094e3c3305918ec7f7fc1ba5af)。
SHA256：`a82b1de1e04344ce9b1f78836a3e5bcb8c3d4d0ec56889602d46fb22bdaeb799`。

240 题、240 个切片 ID，CM、LSCC、LUAD、UCEC 各 60 题。
保留原 CSV 字节，`provenance.json` 记录来源。新增 `configs/datasets/cptac.yaml` 和数据适配器，
`data/manifests/cptac/annotations.json` 是不可用于推理的仅标注 manifest。
原文件 `Task`、Broad/Narrow Category 均为空，适配器使用本地 task 名 `Tumor subtype`。
`Answer` 用作真值；原文件的 `Output` 是已有模型预测，适配器完全忽略。

对应原始 WSI 已下载至 `data/raw/cptac/images/`：CM、LSCC、LUAD、UCEC 各 60 张，
共 97,254,925,262 bytes（90.576 GiB）。下载严格按 `Slide` ID 与 TCIA 官方包目录精确匹配；
`data/raw/cptac/download_lists/matched_wsi.csv` 固定远端路径和大小，
`data/raw/cptac/wsi_inventory.json` 保存 240 张本地文件的大小与 SHA-256。
所有文件均通过远端大小核对，数据适配器解析为 240 个唯一 slide/path。
这不包含预处理、模型 validation 或 benchmark 运行。

## 后续对照方案（未执行）

先以固定 1,000 题进行同题 v1/v2/v3 对照；与已有全量结果比较时，先按 sample_id 截取同题，
不可把样本 accuracy 直接与全量 accuracy 作版本差异。未来全量运行可继续沿用原 7,274 题。
如分片运行，保持 16 shards、每片单张同型号 RTX PRO 6000；本次没有生成或提交作业。
报告总体、论文五组及七个 task 的 accuracy、混淆矩阵、类别召回、balanced accuracy，
逐题修对/改错、Predict→Final 转移、Explore/Zoom、停止原因、invalid、token 与 GPU 小时。
不预设 v3 准确率提升。

以下候选均未加入实现或提前增加开关，每次仅在 v3 基线上改变一项：

1. 明确 BCNB H&E 身份，区分可见观察、推断与未知检测信息。
2. 单独比较 Perceptor 长/短 System，再独立研究结构化描述。
3. 保留完整输出，独立研究仅将最终描述而非 `<think>` 传给 Executor。
4. Reflect 检查证据来源；Missing 区分更多区域、更高倍率与无法取得的外部检测。
5. Final 改答需要新增证据或明确推理纠正，不强制保留首次答案。
6. 分别研究首轮最少 3 区域与首轮 20%，不要同时改变。
7. 分别研究初始 overlap、Zoom 子区数量；倍率仅在获得可靠 MPP 后修订。
8. Executor 采样、token、格式重试与迭代次数作为较低优先级的独立实验。

需要采样稳定性对照时，用 128、256、384 三个固定种子配对运行。
CPU 回归代码包含实际 Perceptor 消息截获、Missing 信息流和预算、数据标签隔离与抽样稳定性；
旧 v1/v2 prompt fixtures 保持不变。真实 validation 与 benchmark 留待后续指令。

## 本次检查状态

静态语法检查、git diff 空白检查、固定样本唯一性/分层覆盖/原图链接核对及重复准备检查通过。
完整 CPU 模拟回归尝试阻塞在共享环境的 `torch._load_global_deps` / `ctypes.CDLL` 加载，
已终止这两个测试进程，不能声明完整回归通过。对应测试代码已保留。
未启动 GPU validation、smoke 或 benchmark，也未加载任何模型 checkpoint。

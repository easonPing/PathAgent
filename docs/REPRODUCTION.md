# PathAgent 论文优先复现

本分支实施已批准的方案：PathMMU、SlideBench-VQA (BCNB) 全量推理，WSI-VQA 本轮只做真实 GPU smoke；不训练、不做消融。基线作者提交为 `8c2abee2cd18ca2e2c839742c9296bd8a41738c5`。以下“实际使用”表示当前代码与冻结配置，不表示已经取得论文指标；真实运行状态另见 `IMPLEMENTATION_STATUS.md`。

依据优先级为论文 > 作者代码 > 经验证的第三方复现 > 显式工程默认。论文未公开的细节不能被称为论文超参数。没有找到包含独立修改和可验证实验结果的第三方复现，因此没有用第三方数值填补空白。

## 超参数、算法和评测差异矩阵


| paper                                      | 作者的repo                                                  | 他人的reproduction | 实际使用                                                                                                                                       |
| ------------------------------------------ | -------------------------------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Executor：Qwen3-4B                          | Qwen3，路径需自行填写                                            | 无经验证结果          | Qwen/Qwen3-4B，revision `1cfa9a7…`，完整 SHA 见配置                                                                                               |
| Perceptor：Patho-R1；图和结果表为 7B               | 在线 Patho-R1，离线通用描述使用 Quilt                               | 无经验证结果          | WenchuanZhang/Patho-R1-7B，revision `7a69eb2…`；通用、问题相关和 Zoom 描述全部使用它                                                                        |
| Navigator：PLIP                             | 外部 PLIP 库、本地权重路径                                         | 无经验证结果          | vinid/plip，revision `67ade53…`；标准 CLIPModel/Processor，与 PLIP 库的编码和归一化方式一致                                                                  |
| 冻结模型，training-free                         | 无训练流程                                                    | 无经验证结果          | inference_mode；LLM/VLM BF16，无 4/8 bit 量化；PLIP FP32                                                                                         |
| 最多 T=5 轮                                   | 默认 5                                                     | 无经验证结果          | 5，配置校验拒绝其他值                                                                                                                                |
| 初始区域数 ceil(0.1N)                           | int(0.1N)，向下取整                                           | 无经验证结果          | 向上取整；N 是完整初始区域集合，N>=1                                                                                                                      |
| Explore 补充 ceil(0.05N)                     | int(0.05N)                                               | 无经验证结果          | 向上取整，分母始终为原始 N，排除已观察区域                                                                                                                     |
| 每轮前 5 个区域补充问题相关形态描述                        | top 5                                                    | 无经验证结果          | 每轮最多 5 个；不足时全部补充                                                                                                                           |
| 初始 5x，Zoom 到 10/20/40x                     | 像素缩放和倍率语义不完整；倍率变量可跨病例泄漏                                  | 无经验证结果          | 原图像素坐标；每题独立状态；从原图重新读取高倍率子区                                                                                                                 |
| Trident 预处理；未公开版本及切片像素参数                   | CLAM；level-0 patch/step 4096                             | 无经验证结果          | Trident 0.2.3，论文前提交 `adf3b7e…`；保持 level-0 FOV=4096、overlap=0，再重采样到 5x。该 FOV 解释是工程默认                                                        |
| 未公开分割器及阈值                                  | CLAM preset                                              | 无经验证结果          | Trident HEST 默认模型，confidence=.5，min_tissue_proportion=0，remove_holes=false，不运行额外 artifact remover；HEST revision `4d24a57…`                 |
| 未公开分割内部细节                                  | 不适用                                                      | 无经验证结果          | 固定 Trident HEST 默认：10x、512 输入、FP16；分割 batch=16；mask_to_gdf 最小轮廓面积=1000；这些是工具默认，不是论文参数                                                      |
| BCNB JPEG 的精确 MPP 未给出                      | 未解决，issue #6 无作者补充                                       | 无经验证结果          | 优先读元数据；缺失时声明 assumed MPP=.5 µm/px（名义 20x）。该假设会影响与论文的可比性                                                                                    |
| 未公开 PathMMU ROI 的物理倍率/重切协议                 | 无完整适配器                                                   | 无经验证结果          | 原始 ROI 作为 N=1、相对 scale=1；相对 Zoom={2,4,8}。不宣称原图是物理 5x；这是关键未公开协议的工程实现                                                                        |
| 通用形态描述 prompt 见正文                          | Quilt 描述                                                 | 无经验证结果          | `Please describe the pathology features in this image.`                                                                                    |
| 问题相关描述 prompt 见正文                          | 在线提示偏向直接回答问题                                             | 无经验证结果          | `Please describe the pathology features related to the question: [QUESTION] in this image.`；附原图坐标与尺度元数据；Zoom 时附 missing visual information |
| Predict → Reflect → Missing information 三步 | 三次 Qwen 调用                                               | 无经验证结果          | 三次结构化调用；若 Reflect=Yes 跳过第三次并结束；完整模板在 models/inference.py，未公开的措辞为显式工程实现                                                                     |
| Explore 使用 missing information I_t 检索      | 部分实现不一致                                                  | 无经验证结果          | 用 I_t 的 PLIP text embedding 查询未观察区域                                                                                                        |
| Algorithm 1：放大当前 X_t 全部区域；选 1 个细粒度 patch   | 仅 top-2 parent；用原问题 q 检索                                 | 无经验证结果          | 所有当前 parent 都生成子区，I_t 检索选 1 个，描述后结束循环                                                                                                      |
| §3.4 描述提到 q，Algorithm 1 写 I_t              | 使用 q                                                     | 无经验证结果          | 按已批准方案采用 Algorithm 1 的 I_t；保留该论文内部歧义                                                                                                       |
| 最终结合累积证据与先前推断                              | 丢失部分历史；日志主要是终态                                           | 无经验证结果          | 汇总所有轮次区域证据、Zoom 证据、每轮决策/推断；保存完整轨迹和每次模型原始输出                                                                                                 |
| 未给出生成长度和采样超参数                              | 三步 512/256/256；通用描述 512、在线描述 1024；final 1024；summary 512 | 无经验证结果          | 保留这些 token 上限；三步与 Perceptor 继承固定 checkpoint generation_config，实际解析值写入 resolved_generation.json                                             |
| 未给出 Qwen thinking flag                     | enable_thinking=False                                    | 无经验证结果          | false；Qwen3 checkpoint 的 do_sample/temperature/top_p/top_k 保留，不把它误写成全流程 greedy                                                             |
| 未公开 final/summary 采样                       | final/summary do_sample=False；temperature=.15/.2 实际无效    | 无经验证结果          | greedy，显式清除无效 temperature/top_p/top_k 参数                                                                                                   |
| 未公开描述压缩实现                                  | 超过 50 描述，调用端 chunk=10，summary 512                        | 无经验证结果          | threshold=50、chunk=10、greedy 512；增加 token-budget 检查，不能无损保留所有观测时显式失败，不静默截断                                                                  |
| 未公开随机种子                                    | 无可靠的阶段独立种子                                               | 无经验证结果          | 128；sample/image ID + stage + 调用序号的 SHA256 派生种子，断点和分片顺序不改变每题随机流                                                                            |
| 未公开 JSON 失败处理                              | 存在宽松提取/回退                                                | 无经验证结果          | 严格字段校验；最多额外 1 次格式重试；仍失败记 invalid，最终评测算错，smoke 判失败                                                                                          |
| PLIP 相似度检索                                 | 标准 CLIP 编码                                               | 无经验证结果          | 图/文 embedding L2 归一化、余弦排序；文本 77 token，稳定 ID 打破同分                                                                                           |
| BCNB accuracy，5 个报告分组                      | 缺少正确的整套评估入口                                              | 无经验证结果          | 7,274 QA / 1,058 原图；7 原始 task；ER/PR/HER2 合并 Receptor 微平均；同时报告各 task 和整体微平均                                                                 |
| PathMMU accuracy                           | 无完整数据准备/分域评估                                             | 无经验证结果          | 5 来源；test+test_tiny，排除 val；固定公开 release 已实测 9,677 QA / 7,280 图；此前参考数为 7,213 图。官方说明披露 Atlas 替换且图片数增加，精确对应关系仍未建立，保留差异而不伪造一致                  |
| BCNB Table 1=55.72；Tables 2/3=54.72        | 未解释差异                                                    | 无经验证结果          | 同时保留两个论文参考值，不挑更接近的作为唯一目标                                                                                                                   |
| PathMMU=53.19                              | 未提供端到端复现结果                                               | 无经验证结果          | 仅作为目标；不根据 smoke 正确率调参数                                                                                                                     |
| 选择题 accuracy                               | SequenceMatcher 以 gold 为基准，空/无效答案可误判正确                   | 无经验证结果          | 只接受有效选项标签或唯一完整选项文本；不使用 gold 参与预测解析；缺失/invalid 按完整 manifest 分母算错                                                                            |
| WSI-VQA 开放题与闭合题                            | 混合过滤、可能忽略空预测                                             | 无经验证结果          | 闭合题 accuracy；开放题 PTB tokenizer + BLEU-1~4/METEOR/ROUGE-L/CIDEr，并报告 exact match；空预测保留分母；本轮仅 smoke，不报告全量复现                                   |
| 未给出公开患者→切片完整映射                             | 依赖 filename/DX1 规则                                       | 无经验证结果          | 显式 slide_map；本轮选两个有 MPP、每例至少 2 closed+2 open QA 的公开诊断 DX 切片；按下载体积确定 smoke，不用预测结果选样                                                         |
| 未指定运行硬件/分片要求                               | 本地固定 GPU 脚本                                              | 无经验证结果          | 用户指定 account=lia-lab-members；两个全量 benchmark 各 16 个非空分片，图像所有 QA 同片；最多 32 并发；每个 benchmark 固定 GPU 型号及显存规格                                     |
| 未给出调度策略                                    | 固定 GPU 0~3                                               | 无经验证结果          | 每次提交前 live Slurm + sbatch --test-only；估计排队和最终分片完工时间；联合比较两个 benchmark 的资源竞争，估计不是预留或保证                                                       |

## 文件与入口

- `configs/reproduce.yaml`：冻结模型、算法、生成与调度参数；`configs/datasets/*.yaml`：数据版本、原图位置与预期覆盖。
- `data_processing/datasets.py`：三套原始标注的规范化、严格选项解析、阻止 gold 进入推理。
- `data_processing/regions.py`：OpenSlide/PIL 原图读取、坐标、MPP 来源、物理/相对倍率和 Zoom。
- `scripts/trident_coords.py`：独立 Trident 环境中的分割和原图坐标生成；`data_processing/preprocess.py`：按内容/配置/源码 hash 的区域、描述和 PLIP 缓存。
- `models/agent.py`：不依赖大模型加载的 Algorithm 1 状态机；`models/inference.py`：三个冻结模型、结构化提示、上下文预算和原始调用记录。
- `pathagent.py`、`scripts/run_inference.py`：真实单 GPU runner；原子写结果，严格 resume 校验，无 CPU 推理回退。
- `scripts/build_manifest.py`：覆盖校验与固定 smoke 选样；`data_processing/sharding.py`：按图像成本 LPT 分片。
- `scripts/select_gpu.py`、`scripts/submit_runs.py`：实时资源选择、三套 smoke 证明校验、两个 16-task array 提交及 job ID 回执。
- `eval/metrics.py`、`scripts/evaluate.py`：完整 manifest 分母、分组统计和严格合并；历史 `eval/metics.py` 转到新入口。

旧 CLAM/Quilt 准备脚本是作者历史实现，不参与新 runner；见其目录 README。历史源码可由上述作者提交直接查看。

## 可执行步骤

```bash
conda activate pathagent
python -m pip install -r requirements-reproduce.txt
python scripts/prepare_assets.py external
conda env create --prefix "$PWD/envs/trident" --file environment-trident.yml
python scripts/prepare_assets.py segmentation
hf auth login
python scripts/prepare_assets.py models
python scripts/prepare_assets.py annotations
python scripts/prepare_wsi_smoke.py
```

BCNB 原图放入 `data/raw/bcnb/images/`，其 stem 与 CSV Slide 对应。PathMMU 按固定 release 的 instructions.md 补齐 images.zip 未包含的外部源；仅下载 ZIP 不代表数据完整。适配器不会静默排除缺图样本。BCNB 作者 Drive 的 gdown 50 项限制尚须可访问的完整下载源；不能设置 remaining_ok=True 后当作全量。

```bash
python scripts/build_manifest.py pathmmu
python scripts/build_manifest.py bcnb
python scripts/build_manifest.py pathmmu --smoke
python scripts/build_manifest.py bcnb --smoke
python scripts/build_manifest.py wsi_vqa --smoke
python -m pytest -q tests
```

PathMMU smoke 每来源 2 题；BCNB 全部原图面积排序选最小/中位/最大三图，各保留其全部 QA；WSI-VQA 2 图，每图 2 closed+2 open QA。选样在看到模型预测前固定。

在允许访问 Slurm 的终端逐一提交，脚本每次重新发现 GPU，不沿用旧选择：

```bash
python scripts/submit_runs.py smoke --dataset pathmmu
python scripts/submit_runs.py smoke --dataset bcnb
python scripts/submit_runs.py smoke --dataset wsi_vqa
```

每次输出真实 job ID 和结果目录。检查三个 `profile.json` 的 passed=true（由完整真实 GPU 运行生成）后：

```bash
python scripts/submit_runs.py final \
  --smoke-pathmmu runs/submissions/<timestamp1>/pathmmu \
  --smoke-bcnb runs/submissions/<timestamp2>/bcnb \
  --smoke-wsi-vqa runs/submissions/<timestamp3>/wsi_vqa
```

最终入口会重新检查源码/config/manifest hash、完整数据和 smoke 结果；固定 GPU 约束；写 16 份非空 manifest，保存每次 `sbatch` 的实际回执并用 `scontrol show job` 验证接受。两个 array 已接受后，本轮按用户要求结束，不等待全量任务完成。未来评估：

```bash
python scripts/evaluate.py --manifest data/manifests/pathmmu/full.json \
  --results-dir runs/submissions/<timestamp>/pathmmu/outputs --output runs/pathmmu_metrics.json
python scripts/evaluate.py --manifest data/manifests/bcnb/full.json \
  --results-dir runs/submissions/<timestamp>/bcnb/outputs --output runs/bcnb_metrics.json
```

`--allow-incomplete` 仅用于中间查看：缺失题仍计入分母，输出明确 complete=false。不能将该结果当作最终复现。

## 已知科学与工程限制

1. PathMMU ROI 倍率、BCNB MPP、Trident 参数、精确提示及模型 release 均未由论文完整披露。当前实现将这些选择显式冻结，但不足以宣称 bitwise/精确协议复现。
2. 硬件 runtime ratio 是显式先验：A100-80=1、A100-40=.95、A6000/A40=.45、RTX PRO 6000=1.4、B200=2。初始 smoke 先验为 A100 上 4 小时，可通过 `--estimate-seconds` 改变资源估计；不影响科学参数。全量使用真实 smoke 时间按区域/问题量推算，WSI 区域数使用完整网格的保守上界。调度器的预期起始时间、运行任务的墙钟到期时间与未来竞争会引入误差，不是实测每种 GPU 的性能。
3. 主流程在加载 LLM 之前运行 Trident，避免分割与 LLM 同时占用显存。全量运行需要充分的磁盘、CPU RAM 和作业 walltime；超过 GPU partition 的时长上限时停止提交，不能静默截断样本。
4. 配置与源码变更会使旧 smoke 证明、旧结果目录和缓存失效。作业排队期间修改执行源码也会触发拒绝；这用于保护结果来源的一致性。
## 原始依据

- [论文 arXiv v1](https://arxiv.org/html/2511.17052v1)
- [作者固定提交](https://github.com/G14nTDo4/PathAgent/tree/8c2abee2cd18ca2e2c839742c9296bd8a41738c5)
- [issue #4 倍率](https://github.com/G14nTDo4/PathAgent/issues/4)、[issue #5 baseline 采样](https://github.com/G14nTDo4/PathAgent/issues/5)、[issue #6 BCNB patch 协议](https://github.com/G14nTDo4/PathAgent/issues/6)
- [PathMMU 官方](https://github.com/PathMMU-Benchmark/PathMMU)、[固定数据 release](https://huggingface.co/datasets/jamessyx/PathMMU/tree/054e64e56e599e9636024f1471d49ecae4a2784f)
- [BCNB 官方原始数据入口](https://github.com/bupt-ai-cz/BALNMP/blob/main/download_dataset.md)、[SlideChat BCNB 标注](https://huggingface.co/datasets/General-Medical-AI/SlideChat/blob/c8128b91b9bcb38b395961633232dba4dfca81f2/SlideBench-VQA-BCNB.csv)
- [WSI-VQA 官方](https://github.com/cpystan/WSI-VQA)、[GDC API](https://docs.gdc.cancer.gov/API/Users_Guide/Getting_Started/)
- [固定 Trident 提交](https://github.com/mahmoodlab/TRIDENT/tree/adf3b7e8fdd7c815f33bec79da0d8d265200c2d6)、[PLIP 官方](https://github.com/PathologyFoundation/plip)

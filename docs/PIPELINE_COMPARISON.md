# 同一分支切换新旧 BF16 pipeline

新增 v3：`configs/reproduce_v3.yaml` 在 v2 上恢复区域元数据、Zoom missing_info，
并将本轮 Predict 与 Reflect 结果送入 Missing。三版本差异、固定 BCNB 1,000 题和 CPTAC
标注说明见 [REPRODUCE_V3.md](REPRODUCE_V3.md)。默认入口仍为 v1，下面的双流程 launcher 仍只运行 v1/v2。

两套流程共用当前分支、`conda activate pathagent` 环境、模型权重、原始数据和本地 Slurm。
无需切换分支、创建 worktree 或安装另一台机器的 Conda。

配置现统一命名为 `reproduce_v1.yaml`（旧）和 `reproduce_v2.yaml`（新），默认入口使用 v1。
这次重命名不改变配置参数及内部 protocol 标识，也不包含审计报告提出的下一版算法修改。
历史运行目录中的冻结配置和源码保持其提交时的名字与内容。

| 项目 | 默认旧流程 | Paper 流程（本机适配） |
| --- | --- | --- |
| 配置 | `configs/reproduce_v1.yaml` | `configs/reproduce_v2.yaml` |
| protocol | 缺省即 `legacy-v1` | `paper-prompts-nonthinking-2x-v1` |
| 参考实现 | `fe18766` 的推理 backend，保留原实现 | GitHub `1e4d0377a4438b19d0e2dde6803db29f804ac8f4` |
| 提示词 / 消息格式 | 本地原提示词 | 固定提交中的公开提示词及消息封装 |
| Predict / Reflect / Missing token 上限 | 512 / 256 / 256 | 1024 / 512 / 512 |
| 通用 / 问题及 Zoom 描述 token 上限 | 512 / 1024 | 1024 / 2048 |
| Final / Summary token 上限 | 1024 / 512 | 2048 / 1024 |
| Perceptor 采样 | 继承固定 checkpoint | **同样继承固定 checkpoint** |
| Executor 采样 | Predict/Reflect/Missing 继承 checkpoint；Final/Summary greedy | 相同 |
| Executor thinking | false | false |
| Perceptor 输入 | 区域元数据；Zoom 追加 missing_info | 原图及描述提示词；元数据仅记日志 |
| JSON 处理 | 原解析器及 `zoom_level` | Executor 排除 `<think>` 内 JSON；`recommended_zoom_level` 映射回状态机 |
| invalid 恢复 | 再次推理 | 保留为终态；评测仍算错 |
| 描述缓存根目录 | `data/processed` | `data/processed/paper-prompts-nonthinking-2x-v1/bf16` |

**用户指定的差异：** GitHub 参考代码显式给 Perceptor 传 `do_sample=False`；本机 Paper
配置和旧配置都为 `perceptor_sampling: checkpoint`，生成调用不覆盖 `do_sample`、
temperature、top_p 或 top_k。继承 checkpoint 不代表一定启用随机采样；实际值取决于
固定权重的生成配置与 Transformers 默认值，记录在 `resolved_generation.json` 中。
本次 BCNB 实测两套 Perceptor 均为 `do_sample=False`；Executor 前三步均为
`do_sample=True, temperature=0.6, top_p=0.95, top_k=20`，Final/Summary 均为 greedy。
运行的 `prompt_manifest.json` 同时记录固定上游提交和这项适配，不能将此配置当成未经修改的线上运行。

两套流程均使用单张 GPU：Qwen3-4B / Patho-R1-7B 为 BF16，PLIP 为 FP32；模型
revision、Algorithm 1、Trident 分割参数、原图坐标和倍率逻辑相同。两套配置都保留
本机 `workers: 1`。Trident 直接使用 `envs/trident/bin/python`，不经过线上 `activate.sh`。
本次不包含线上双 A5000、NF4、四分片本地 launcher 或其旧实验结果依赖。

## 直接运行

从项目根目录激活环境。以下推理命令需在已分配的 GPU 中执行；输出目录名称可按实验修改，
新旧流程必须使用不同目录。相同 manifest 保证同题对照，硬件也应保持同型号与显存规格。

```bash
conda activate pathagent

python pathagent.py \
  --config configs/reproduce_v1.yaml \
  --manifest data/manifests/bcnb/full.json \
  --output runs/compare/experiment_01/legacy

python pathagent.py \
  --config configs/reproduce_v2.yaml \
  --manifest data/manifests/bcnb/full.json \
  --output runs/compare/experiment_01/paper
```

`--config` 缺省时始终使用旧配置。YAML 未填写 protocol 时仍是旧协议；未知名称、
不符合 Paper 协议的 token 上限和非本次支持的精度/设备配置会被拒绝。

## Preflight 与 Slurm

以下现有入口均支持 `--config`，默认仍为 `configs/reproduce_v1.yaml`。
Preflight 检查资产，不运行推理；输出缺数据或缺权重时退出码为 2。

```bash
python scripts/preflight.py --config configs/reproduce_v2.yaml \
  --output runs/preflight-paper.json

# BCNB smoke manifest 尚未生成时，先按既有规则生成一次；两套流程使用同一份。
python scripts/build_manifest.py bcnb --smoke
python scripts/submit_runs.py smoke --dataset bcnb --config configs/reproduce_v1.yaml
python scripts/submit_runs.py smoke --dataset bcnb --config configs/reproduce_v2.yaml
```

`scripts/submit_runs.py final` 同样支持 `--config`，仍需提供同一配置、同一源码下
三套有效的 smoke 证明。旧证明不能为另一套协议背书。
`slurm/submit_benchmark.py` 也支持 `--config`，保留已有的 16/128 分片参数、
`--skip-smoke` 显式选择和 `--submit` 开关；此次适配不改变其验收或提交条件。

Slurm 脚本使用提交时的 `sys.executable` 并保存所选配置，因此提交前需先激活
`pathagent`。不会在作业里寻找另一台机器的 Miniforge 或自动切换到 base。

## 缓存、恢复与核对

- 模型权重和原图复用；描述缓存身份包含协议、提示词、生成参数与模型信息，不能混用。
- 保留源码 hash 校验。此次增加源码后，历史结果/缓存不会被当作新运行续接；新实验使用新目录。
  旧协议行为保留，不承诺重新运行与历史输出逐字一致。
- 同目录续跑必须匹配源码、配置、协议、提示词和 manifest 身份。Paper 的 invalid 保留为错题，
  不通过重复恢复重新抽样；旧流程保持原先对 invalid 重跑的行为。
- 每个输出目录记录 `run_lock.json`、`prompt_manifest.json`、`resolved_generation.json`、
  `environment.json`、`image_inventory.json`、逐题结果和轨迹、`metrics.json` 及 `profile.json`。
  Paper 的逐次调用另记录实际生成参数、长度、输入尺寸与图像 hash；通用描述调用保存在
  inventory 所指向的 observation cache 中。
- v2 按用户要求将精确匹配的字符串 `"10"`、`"20"`、`"40"` 转为整数，再执行原有合法倍率校验。
  其他字符串不转换。Perceptor 原始描述中的 `<think>` 内容保留；Executor 的排除规则不套用于它。

CPU 回归命令：

```bash
conda activate pathagent
python -m pytest -q tests
```

固定参考 fixtures 来自本地 Git 中的 `fe18766` 和 `1e4d037`，包括原始 prompt manifest，
以及固定模拟回复、固定 evidence 下 Predict/Reflect/Missing/Final 的消息和预算。
测试用模拟模型验证生成调用，不下载模型，也不提交 GPU 作业；它们不能代替真实 GPU smoke。

## BCNB 双流程自动验收与全量运行

`slurm/submit_bcnb_comparison.py` 是只运行 BCNB 的专用入口。使用 `pathagent` 环境，
`prepare --root <新目录>` 保存执行源码、两套配置及相同的样本/分片快照；
`preview --root <目录>` 仅执行 Slurm test-only；`submit --root <目录>` 提交两个
validation 和一个依赖两者结束的控制作业。此入口使用本次规划保存的历史逐切片耗时
`runs/planning/bcnb_cs6501_20260928/historical_costs.json` 均衡分片，该文件必须存在。

两套 validation 都须 Slurm `COMPLETED/0:0`，并且 21 道题均为有效输出、运行身份和
GPU 规格一致，控制作业才会提交**两套**全量数组。通过表示运行和输出协议通过验收，
不是要求答案准确率达到 100%。缺失、失败或过期结果都会阻止两套全量提交。
每套数组固定 16 shards，不设置 `%` 并发节流；每个 shard 使用一张 RTX PRO 6000、
8 CPU、64 GB 内存，账号 `cs6501-cbx8wm`、QOS `class`。验证各限时 60 分钟，
全量每片 legacy 210 分钟、paper 294 分钟，实际并发及启动时间取决于调度器。

控制作业保存 `gate_decision.json`；每次提交立即保存 `receipts/*.json`，重复调用会
复用已有提交记录。两套数组分别全部成功结束后自动汇总至 `legacy/metrics.json` 和
`paper/metrics.json`。全量计分保留 invalid 为错误答案，不删除题目以缩小分母。
验证和全量共用 `gpu_fleet.json` 检查实际型号/显存；代码从保存的 `source/` 执行，
模型、原图及 conda 环境继续使用本地资产。

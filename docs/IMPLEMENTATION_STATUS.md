# 实施状态

## BCNB shard 0 Trident 内存失败（2026-09-22 22:05 EDT）

- 数组任务 **20342742_0** 于 2026-09-22 15:14:12 EDT 启动，运行 1 分 20 秒后以 exit code 1 失败；其余 `1-127` 仍为 `PENDING / Priority`，调度器当前不提供 StartTime。自动评分 **20342743** 仍为 `PENDING / Dependency`。
- 失败发生在 BCNB `613.jpg` 的 Trident HEST 组织分割阶段：`ImageWSI.segment_tissue -> _segment_semantic -> DataLoader worker start -> os.fork` 抛出 `OSError: [Errno 12] Cannot allocate memory`。该任务配置为 Trident `workers=4`、8 CPU、64GB host RAM；sacct 记录 batch step MaxRSS=28,388,620K。这是 host memory / worker fork 失败，不是 CUDA OOM。
- shard 0 尚未生成任何题目结果或 `profile.json`，因此当前有效进度为 **0/128 shards、0/7,274 QA**。由于评分依赖为 `afterok:20342742`，即使其余分片成功，当前评分依赖也无法满足。
- 当前尚未修改配置、取消剩余数组或重新提交。建议先把 Trident worker 降到 0 或 1，对失败 shard 做真实 GPU 验证，再以新的源码/config 身份提交全量数组和评分依赖。

## 当前 BCNB 作业：128 分片，A6000，最多 32 卡（2026-09-21 18:30 EDT）

- 按用户最新指令提交 **20342742，array=`0-127%32`**，固定 A6000 48GB；每片 8 CPU / 64GB RAM / 12:18:00，account=`lia-lab-members`。当前 PENDING，实际 StartTime 尚未给出。
- **自动评分作业 20342743** 依赖全部 128 个分片成功。完整评测覆盖 **7,274 道题 / 1,058 张图**；检查无重复、无遗漏，同图 QA 不跨片。
- 验证新 array 和评分依赖后，已按用户指令取消旧 BCNB **20341940**、旧评分 **20341941** 和旧 smoke **20336395**；sacct 已确认取消。
- 提交目录：`runs/submissions/20260921_182927_645806_bcnb_full/`，包含配置、manifest、回执、`replacement.json`、`completion_forecast.json`；结果位于 `outputs/shard_*/results/`，汇总指标为 `metrics.json`。
- 命令：`python slurm/submit_benchmark.py bcnb --skip-smoke --shards 128 --concurrency 32 --gpu-feature a6000 --submit`。布局校验支持用户授权的 16/128 分片；论文算法和模型参数未改。
- 12 项测试通过；额外检查了 128 分片完整性、配置/源码身份以及实际提交参数 `--array=0-127%32`。
- 同规格额外请求的 test-only 预测首次启动为 **9 月 23 日 07:16 EDT**；假设届时起持续获得 32 张卡且估算成立，约 **9 月 24 日 14:04** 完成计算；16 卡持续并发则约 **9 月 25 日 20:42**。这不是现有数组的实际启动承诺；吞吐未经 smoke 实测，不含精确加载/I/O/重跑及评分排队耗时。
- 以下旧 BCNB 作业记录只作历史参考。本次没有提交 PathMMU 全量作业。

## 最新进展：按官方说明补 PathMMU 原图（2026-09-21）

- 本轮用户重新要求处理 PathMMU，因此先前“只提交 BCNB、暂停 PathMMU”的指示已由本轮任务更新；本轮未提交新的 PathMMU GPU 作业。
- 已从官方或可核对原始文件名的公开镜像补齐 CRC100K、BACH、LC25000、SICAPv2、SkinCancer、Osteo、WSSS4LUAD、Atlas。PubMed、EduContent 和 PathCLS 原已随固定 release 提供的图片保留。逐源来源与恢复方法见 `PATHMMU_DATA.md`。
- MHIST 已从保留原始文件名的公开归档补齐 **80/80** 张，并记录固定 revision、归档 SHA256 和逐图 SHA256。用户已收到官方访问邮件，但尚未用邮件中的原始 ZIP 做字节级比对。
- SocialPath 无需 X API 凭据，已按原帖公开网页的图片顺序恢复 **1,239/1,318** 张。剩余 **79** 张的原帖当前返回 404 或不提供图片数据；重复尝试后数量稳定，未以其他图片代替。
- `python data_tools/audit_pathmmu.py --verify` 对所有现有图片完成 Pillow 格式校验，**0 张无效**。含验证集共 **7,715/7,794** 张；正式测试 `test` + `test_tiny` 共 **7,207/7,280** 张。覆盖与逐图缺失映射见 `runs/assets/pathmmu_sources/coverage.json`。
- `python scripts/build_manifest.py pathmmu` 如预期因缺图拒绝生成完整 manifest；因此不能提交 PathMMU 全量 benchmark 或声称全量数据就绪。下面旧状态只作历史记录。

## 最新进展：跳过 smoke，提交正式 benchmark（2026-09-21 17:37 EDT）

- 用户明确授权先假设 smoke 成功，直接申请 GPU 跑 benchmark；这次不再以 smoke 验收作为提交前提，也没有生成虚假的通过证明。
- **BCNB 数据已齐全**：重新验证 1,058 张原图、7,274 道题，生成完整 manifest。16 个分片覆盖全部题目且无重复，同图问题不跨片。
- **BCNB 全量 array 已实际提交：20341940**，`0-15%16`，每片 A100 40GB ×1 / 8 CPU / 64GB RAM / 44 小时 9 分钟，account=`lia-lab-members`；提交后为 `PENDING / Priority`。
- **自动评分作业：20341941**，CPU 作业，依赖全部分片成功；完整分母评分写入本次提交目录的 `metrics.json`。
- 回执、配置、manifest、GPU 选择快照与授权记录：`runs/submissions/20260921_173656_813116_bcnb_full/`。
- 新入口：`python slurm/submit_benchmark.py bcnb --skip-smoke --submit`。它仍强制检查全量数据完整性、固定模型、16 个分片以及同一 benchmark GPU 型号/显存一致性。未运行 GPU smoke，因此内存和吞吐未实测；时间申请来自明确标记的工程估算。
- 原 smoke **20336395 保留**，推理源码 hash 未改变。新增提交与数据恢复工具放在独立目录，不影响既有作业的源码身份检查。
- **用户最新指示：先不考虑 PathMMU，本轮只提交 BCNB**。PathMMU 全量没有提交；补图工作暂停，已下载文件保留。公开 HF 镜像核查后仅覆盖已有 PubMed/EduContent，不能补缺。恢复时仍需先检查全量原图完整性。
- 下面各节为历史记录；与本节冲突时以本节为准。

## 最新进展：PathMMU 先行测试（2026-09-21）

- 用户已解决 HF 登录；三个固定模型均下载完成并通过资产校验，Patho-R1 不再阻塞。恢复 huggingface-hub 0.36.2 后，Transformers 4.51.0 导入和 pip check 均通过，登录凭据保留。
- PathMMU 标注和 images.zip 已下载解压。固定 release 实测 9,677 QA / 7,280 个唯一图像名；此前参考 7,213 与本 release 不一致。官方说明披露 Atlas 替换且图数增加，不能据此断言与论文原始数据完全相同。审计文件：`runs/assets/pathmmu_release_audit.json`。
- ZIP 覆盖 PubMed 全部 3,068 题、EduContent 全部 1,938 题、PathCLS 中 74 题；SocialPath、Atlas 和其余 PathCLS 外部原图尚未补齐。
- 按本轮“先用 PathMMU 跑一下”的范围，从已下载原图中按稳定 ID hash 每来源选 2 题，共 6 题，manifest：`data/manifests/pathmmu/smoke_available.json`。这是 availability-restricted execution check，不代替完整五来源 smoke 验收，不是全量成绩。
- 真实 GPU 测试已提交并经 scontrol 验证：**Job 20336395**，account=`lia-lab-members`，GPU=`A6000 48GB ×1`，8 CPU / 64GB host RAM，walltime=1:55:00。提交后为 **PENDING**；尚无模型推理结果。
- 回执和选卡记录：`runs/submissions/20260921_142431_957767/receipt.json`、`gpu_selection.json`。预期输出目录：`runs/submissions/20260921_142431_957767/pathmmu/`。
- BCNB 按用户最新要求暂不测试；本次未提交 WSI-VQA 或全量数组。下面保留上一轮的历史状态，以上更新优先。


本文件记录当前实际完成的工作，不把 CPU 测试、合成评估输入或 `sbatch --test-only` 当作真实 GPU smoke。

## 已完成

- 论文优先的配置、三数据集适配器、原图/尺度处理、Trident 接口、Patho-R1/PLIP 缓存、Algorithm 1 状态机、完整轨迹、严格 resume、严格 MCQ 和开放题评估。
- 每 benchmark 16 个非空分片的实现；同图所有 QA 同片；实时 GPU/队列查询、联合完工时间估计、型号及显存一致性检查、最终 array 提交与回执验证入口。
- 12 项针对性测试通过；包括算法取整、Explore/Zoom 查询和范围、五轮上限、状态隔离、完整证据、无效答案和缺失分母、原图 Zoom、稳定分片、manifest 篡改拒绝以及繁忙 GPU 的保守排队估计。
- 主环境与独立 Trident 环境 `pip check` 均通过。Trident 环境为 Python 3.10，torch 2.7.0+cu128、torchvision 0.22.0+cu128、timm 0.9.16、segmentation-models-pytorch 0.4.0。最初选择的 SMP 0.3.4 与 timm 冲突，已实际修复。
- Qwen3-4B、PLIP、固定 HEST checkpoint 和 HEST 初始化 backbone 已下载，模型锁保存 SHA256。HEST 在 CPU 上加载成功；这不等于 GPU 分割验证。
- BCNB CSV 已校验固定 SHA256，实际 7,274 QA / 1,058 图；WSI-VQA 测试标注实际 735 QA / 86 病例。
- WSI-VQA smoke 两张原图已通过 GDC 文件大小、MD5 和 SHA256 校验，8 题 manifest 已生成。病例为 TCGA-AC-A2FG、TCGA-AN-A0XW，MPP 分别 .252 和 .2525。另一个更小文件缺少尺度元数据，已显式排除，并保留下载/排除记录。
- 开放题评估以明确标记的合成预测完成 Java/PTB、BLEU、METEOR、ROUGE-L、CIDEr 接口检查，文件为 `runs/validation/synthetic_evaluator_check.json`；不是模型输出，也不是 benchmark 成绩。
- Slurm 选择器已执行真实只读资源查询和 `sbatch --test-only`，生成 `runs/assets/gpu_selection_validation.json`；没有实际提交 GPU 作业。
- 完整四列超参数/算法差异矩阵及运行命令位于 `REPRODUCTION.md`。

## 当前阻塞

1. **Patho-R1 权重缺失**：未检测到 Hugging Face 登录凭据。需要在本机用已获批账号执行 `hf auth login`；不要把 token 放到聊天或仓库中。
2. **PathMMU 原始数据缺失**：同样需要已获批 HF 账号；取得固定 release 后仍需按其说明补齐外部图片源。现阶段不能验证全量覆盖，也不能生成正式 smoke/full manifest。
3. **BCNB 原图缺失**：作者公开 Google Drive 的 WSIs 子目录超过 gdown 的 50 项列表上限，下载器已明确失败，没有接受部分列表。作者备用 OneDrive 接口返回 401（另一个入口报告 migrated）。需要完整原图目录或可访问的完整压缩包/下载入口。

数据完整性阻塞记录在 `runs/preflight.json`。执行 `python scripts/preflight.py` 可刷新；缺资产时退出码 2。

## 尚未完成的验收

- 三套真实 GPU smoke：**均未执行/未通过验收**。
- PathMMU 与 BCNB 的最终 16-shard array：**均未提交，没有真实 job ID**。
- 真实大模型生成的格式遵循、峰值显存、吞吐、Trident GPU 分割，以及完整数据上的答案准确率：**尚未验证**。

以上步骤必须在数据和模型就绪后继续。最终提交入口会拒绝缺失、失败或过期的 smoke 证明，不会以单元测试替代。当前结果应称为“复现实现和 CPU/接口验证已完成，真实 GPU 验收被资产访问阻塞”，不能称为“论文已复现”。

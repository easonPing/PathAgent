# 实施状态

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

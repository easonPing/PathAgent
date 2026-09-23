# PathMMU 原图补全记录

标注和 `images.zip` 固定为 `jamessyx/PathMMU@054e64e56e599e9636024f1471d49ecae4a2784f`，原始路径以 `data/raw/pathmmu/data.json` 的 `source_img` 为准。评估只使用 `test` 和 `test_tiny`，补图工具也覆盖 `val`。目标目录是 `data/raw/pathmmu/images/`。

官方 [additional data instructions](https://github.com/PathMMU-Benchmark/PathMMU/blob/main/data/instructions.md) 要求从各原始来源获取 PathCLS、SocialPath 和 Atlas 图片。恢复工具只写入映射中指定的文件名，TIFF 按官方 `construct_pathcls.py` 转为 RGB JPEG。压缩包恢复记录输入与输出 SHA256，Parquet 恢复记录输出 SHA256；报告在 `runs/assets/pathmmu_sources/`。

| 来源 | 图像数（含 val） | 使用的数据源 | 状态 |
| --- | ---: | --- | --- |
| PubMed | 2,262 | 固定 release `images.zip` | 齐 |
| EduContent | 1,547 | 固定 release `images.zip` | 齐 |
| CRC100K | 358 | [作者 Zenodo](https://zenodo.org/records/1214456) | 齐 |
| BACH | 160 | [作者 Zenodo](https://zenodo.org/records/3632035) | 齐 |
| LC25000 | 200 | [作者 Zenodo](https://zenodo.org/records/14998042)，不足的 80 张由 [完整镜像](https://huggingface.co/datasets/1aurent/LC25000/tree/2387ffed63932faa8fd05c679ffec85fc3153f61) 补齐 | 齐 |
| SICAPv2 | 160 | [作者 Mendeley](https://data.mendeley.com/datasets/9xxm58dvs3/1) | 齐 |
| SkinCancer | 629 | [heiDATA](https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/7QCR8S) | 齐 |
| Osteo | 120 | [TCIA 原始项目](https://www.cancerimagingarchive.net/collection/osteosarcoma-tumor-assessment/) 的 [完整镜像](https://huggingface.co/datasets/CAIR-M3LLM/OsteosarcomaTumorAssessment/tree/4580d282487fa9ebfac025a85260159f13733d25) | 齐 |
| WSSS4LUAD | 120 | [原始挑战](https://wsss4luad.grand-challenge.org/) 的 [完整镜像](https://huggingface.co/datasets/MedOtter/WSSS4LUAD-v2/tree/aa035a16f230622a17f51bc6b5180209a04bec7b) | 齐 |
| Atlas | 762 | [ARCH `books_set`](https://warwick.ac.uk/fac/cross_fac/tia/data/arch/books_set.zip) | 齐 |
| MHIST | 80 | [公开归档](https://huggingface.co/datasets/NYCU-PCSxNTHU-MIS/MHIST/tree/465df791910a55ff93683ac914170f0b32de2013) 中保留原始文件名的 `mhist.tar.gz` | 齐，80/80 |
| SocialPath | 1,318 | 映射中的原始 X 帖子公开网页及图片 CDN | 已恢复 1,239/1,318；79 帖当前未提供目标图片 |

PathCLS 中还有 78 张没有 `source_img` 的图像，已由固定 release 提供。上述分项不能直接相加成总体，因为这 78 张单列于说明中。

2026-09-21 的逐图校验结果：包含 `val` 的 7,794 张中已具备 7,715 张；`test` 与 `test_tiny` 的 7,280 张中已具备 7,207 张。现有图像均通过 Pillow 格式校验。剩余 79 张均为 SocialPath，其中测试集缺 73 张。

镜像匹配依据为 `source_img` 的原始文件名。LC25000 的 120 张 Zenodo 图像已重新逐张读取并与本地文件做 SHA256 对比，全部字节一致；另外 80 张不存在于该 Zenodo 包，由完整镜像补齐。校验记录分别为 `runs/assets/pathmmu_sources/LC25000_zenodo_verified.json` 和 `LC25000_parquet_restored.json`。Osteo 和 WSSS4LUAD 镜像未提供原始项目逐文件校验和，因此不能声称已做官方字节级核对。

使用 Atlas 部分时还需按 PathMMU 官方说明引用 Gamper 与 Rajpoot 的 *Multiple Instance Captioning: Learning Representations from Histopathology Textbooks and Articles*（CVPR 2021）。从 Parquet 镜像恢复图片需要安装 `pyarrow`；压缩包恢复工具使用项目现有的 Pillow 和 requests。

## MHIST 与 SocialPath 恢复记录

1. 用户已收到 [MHIST 官方访问邮件](https://bmirds.github.io/MHIST/)，但截图中的四小时临时链接无法准确复制。已从上述公开归档的固定 revision 取得 `mhist.tar.gz`，按 `source_img` 精确匹配原始文件名，恢复 80/80 张，全部为 224×224。归档 SHA256 为 `e6426329fa3d67c0b861b1f348a085fb16ebb593eb164b1c37b1c7a25b09f0eb`；逐图哈希及原始归档成员名在 `MHIST_restored.json`。尚未与作者邮件中的 `images.zip` 做字节级比对。复现命令：

   ```bash
   python data_tools/restore_pathmmu_images.py --archive /path/to/mhist.tar.gz --source-prefix MHIST --source-url https://huggingface.co/datasets/NYCU-PCSxNTHU-MIS/MHIST/tree/465df791910a55ff93683ac914170f0b32de2013
   ```

2. SocialPath 按 `data/raw/pathmmu/socialpath_mapping.json` 的 `tw_id` 和从 1 开始的 `img_position`，读取原帖公开网页中属于该帖子本身的有序图片地址，再从 `pbs.twimg.com` 下载原图。该方法不需要 X API 凭据。三次执行后恢复 1,239/1,318 张，剩余 79 张在最新尝试中有 69 个帖子返回 404、10 个帖子页面没有图片数据；记录在 `SocialPath_restored.json`。脚本仅按原帖 ID 和位置保存，未用其他图片代替。

   ```bash
   python data_tools/restore_pathmmu_socialpath.py --workers 3
   ```

3. 查到的 [公开 PathMMU TSV 镜像](https://huggingface.co/datasets/mobiushy/PathMMU/tree/main) 虽内嵌图片，但 [Dataset Viewer 分类统计](https://datasets-server.huggingface.co/statistics?dataset=mobiushy%2FPathMMU&config=default&split=train) 仅有 PubMed 3,020 条和 EduContent 1,829 条，没有 SocialPath。PathMMU [维护者说明](https://huggingface.co/datasets/jamessyx/PathMMU/discussions/4) 原图因再分发限制需从原始来源获取。

4. 每次补图后核查：

   ```bash
   python data_tools/audit_pathmmu.py --verify
   python scripts/build_manifest.py pathmmu
   ```

覆盖报告在 `runs/assets/pathmmu_sources/coverage.json`，其中逐项列出缺图的映射。缺图时审计脚本退出码为 2；完整 manifest 也会拒绝缺图。

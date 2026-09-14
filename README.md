# PanCancer-Dependency

泛癌多组学功能依赖预测，以及透明细胞肾细胞癌候选脆弱性研究。

当前迁入的是共享岭回归、留一癌系验证和模块分析代码；尚未实现预训练或患者跨域适配。模块搜索保留用于复现历史比较，DDQN/MORL 不属于当前主线。

## 当前工作：公开数据基线审计（2026-09-14）

研究约束：目前仅使用公开数据。目标暂定为患者相关的 ccRCC 候选依赖优先级；患者域预测和生存关联不能单独证明患者功能依赖。

目的：确认现有数据和模型能否作为泛癌学习、患者域迁移之前的可靠基线，先核实样本身份、候选范围和验证边界。

本次实测事实（读取共享 DepMap 数据、原始 CSV 表头和历史结果，未读取 TCGA 锁定 Test）：

- 处理后数据包含 883 个模型；四种模态行索引一致且无重复。依赖矩阵为 883×261，存在 39 个缺失值；表达、拷贝数、突变矩阵均为 883×266，无缺失值。
- 当前 LOLO 排除 30 个 common-essential 靶点和 3 个其余依赖标签不完整的靶点，剩余 228 个，覆盖 19 个样本数不少于 20 的癌系。这是基因过滤，尚不是独立的依赖背景校正模型。
- Kidney 有 24 个模型，其中 RCC 20 个、明确 ccRCC 12 个。Kidney 内含一个标注为 Non-Cancerous 的永生化胚胎肾模型 ACH-001310；历史 Kidney 汇总不应被解释为纯 ccRCC 或纯肿瘤队列。
- `build_depmap_bridge.py` 的候选来自 TCGA train 中至少 3 次、频率不超过 5% 的突变基因，并排除驱动背景基因。因此当前预测范围不是全基因组，不能据此评估普适依赖发现能力。
- 本地原始 CRISPRGeneEffect CSV 有 17,917 列（含索引列），另外三种组学原始文件也存在；本次仅检查文件及表头，尚未完成全范围基因交集、质量与身份审计。
- 共享输入的 7 个文件 SHA-256 均与历史 LOLO run.json 一致。历史 shared_multiomics 的癌系平均 ΔR² 为 0.06574，Kidney 为 0.08860，均相对外层训练逐靶点均值；这些不是本次重新训练的结果。

判断：现有结果支持继续检查多组学预测信号，但低频突变预筛选、小规模 ccRCC 样本和 Kidney 身份混杂限制了外推。当前数据不支持直接宣称已完成泛癌预训练或患者功能依赖验证。

首个正式实验的拟定范围（数据构建已完成，正式训练尚未执行）：

1. 从原始 DepMap 建立独立版本的广覆盖基因输入，不再使用 TCGA 低频突变筛选限定全部预测靶点；记录模态交集、缺失与筛选规则。
2. 按 ModelID 和疾病注释审计样本，排除明确 Non-Cancerous 模型参与肿瘤训练与主评价，单独记录明确 ccRCC、其他 RCC 和未定亚型。永生化模型不充当正常成人肾脏安全性证据。
3. 先比较训练集逐靶点均值、背景和共享岭回归；报告绝对依赖预测及样本差异预测，common-essential 与其他基因分层评价。任何数据驱动的预处理、背景估计或调参仅使用相应训练折。
4. 分开报告跨癌系迁移与 ccRCC 子集表现；历史 ccRCC 已参与探索，不能重标为独立确认集。复杂预训练须证明相对基线及无预训练模型的增益。
5. 患者适配之前审计公开外部功能数据的 ccRCC 覆盖、共享细胞系和来源。若缺少足够独立功能样本，明确保留候选预测结论，不用相关性验证替代功能验证。

验证：原有 23 项合成数据测试及 LOLO dry-run 通过。审查发现原 LOLO 防泄漏测试仅重复相同输入，现已改为实际扰动外层留出标签，检查内层调参分数、所选 alpha 和最终预测均不改变；修改后 23/23 项测试再次通过。正式训练仍由用户手动启动。

## 广覆盖 DepMap 数据基线 v1（2026-09-14）

目的：解除 TCGA 低频突变候选范围限制，建立只依赖 DepMap 原始数据的癌症模型输入，并保留 common-essential、缺失和样本身份信息供后续严格评价。

入口：`scripts/build_depmap_baseline.py`。已执行数据构建，未训练模型、未读取 TCGA 数据。输出为 `data/processed/depmap_baseline_24q4_v1/`，不覆盖旧项目共享数据或历史结果。

| 项目 | 本次实测结果 |
| --- | --- |
| 四模态共有模型 | 883 个；排除明确标注 Non-Cancerous 的 10 个，保留 873 个 |
| 四模态共有 HGNC 标准基因 | 15,835 个，保留驱动基因，无突变频率或依赖强度筛选 |
| common-essential | 1,328 个，保留发布版注释，不过滤或统一扣除 |
| Kidney / RCC / 明确 ccRCC | 23 / 20 / 12 个模型 |
| 依赖标签缺失 | 86,148 个值，约 0.6232%；保留为 NaN |
| 表达、拷贝数、突变缺失 | 共有基因与保留模型范围内均为 0 |
| 同患者多模型 | 24 位患者；PatientID 无缺失，未发现同患者跨癌系 |
| 输出规模 | 6 个文件，合计约 86.53 MiB |

保留文件：`matrices.npz`（四个 float32 矩阵及 model_ids、genes，无 pickle）、`models.csv`（保留样本注释）、`model_audit.csv`（逐模型纳入与排除原因）、`gene_coverage.csv`（基因模态覆盖、缺失与 essential 注释）、`gene_mapping.csv.gz`（原始列到标准基因的映射）、`audit.json`（来源校验和、规则、统计、输出哈希）。未填补缺失、标准化、拟合或按结局选择基因；同名标准基因的连续值取均值，突变计数取最大值后转为是否大于 0，无法唯一映射到 HGNC 标准基因的列不纳入。

验证：6 个 DepMap 原始文件大小及 MD5 均匹配迁入脚本固定的 24Q4 清单；输出数据文件 SHA-256 核对通过；四矩阵维度、类型、模型顺序、基因唯一性、缺失计数和突变二值检查通过；抽取驱动基因、TLN1、边界基因及有重复映射的基因，与原始 CSV 跨全部保留模型核对通过。确认非癌模型均已排除，`tests/` 已删除。

限制与下一步：这是四模态交集，不能称为全基因组无偏覆盖；广覆盖没有增加 ccRCC 独立样本量。common-essential 注释来自整个发布版，用于分层描述，不能称为训练折内估计。保留 EngineeredModel 标记，尚未将所有工程化模型自动排除。下一步实现读取 NPZ 的基线训练入口，处理缺失标签、训练折内预处理、同患者分组及 ccRCC 子集评价；旧 TSV 版 LOLO 不兼容此 NPZ，也不应直接按全量靶点逐个回归扩展而忽视计算成本。

复现命令（已有输出时脚本拒绝覆盖；需重建时使用新的版本目录）：

```bash
/home/liliang/miniconda3/envs/rl_genrisk/bin/python scripts/build_depmap_baseline.py \
  --raw-dir /mnt/e/projects/rl-genrisk-main/data/raw/depmap_24q4 \
  --hgnc /mnt/e/projects/rl-genrisk-main/outputs/reassessment_20260911/hgnc_complete_set.tsv \
  --output-dir data/processed/depmap_baseline_24q4_v1
```

## 环境与运行检查

在 WSL 中执行，沿用已验证的 Python 环境：

```bash
cd /mnt/e/Projects/PanCancer-Dependency
/home/liliang/miniconda3/envs/rl_genrisk/bin/python scripts/run_depmap_lolo_validation.py --input-dir /mnt/e/projects/rl-genrisk-main/data/processed/depmap_bridge_24q4 --dry-run
```

该命令仅检查历史 DepMap 输入，不拟合模型。精简依赖见 requirements.txt，现阶段未验证全新安装环境。

按用户要求，2026-09-14 删除了本项目的 `tests/` 及其全部内容；上文及迁移清单的测试结果仅记录删除前的历史验证。后续使用必要的运行检查和一次性数值检查，不保留测试目录。删除会失去可重复运行的自动回归检查，并不表示原测试没有作用。

## 历史基线的手动运行

训练由用户手动启动。以下命令会重新拟合完整 LOLO，仅在决定开展该实验后执行；不需要为迁移而重跑：

```bash
cd /mnt/e/Projects/PanCancer-Dependency
/home/liliang/miniconda3/envs/rl_genrisk/bin/python scripts/run_depmap_lolo_validation.py --input-dir /mnt/e/projects/rl-genrisk-main/data/processed/depmap_bridge_24q4 --output-dir outputs/depmap_lolo_validation_manual
```

脚本逐癌系、内层折打印进度。禁止覆盖既有结果；正式结果仍需审查。

## 数据、结果与隔离规则

- 大型数据共用旧项目路径；路径清单见 configs/shared_data_paths.json。该 JSON 是路径记录，现有脚本不会自动读取它，运行时必须显式传入对应参数。
- 共享数据应保持只读；修改处理方式时另建版本。不得删除旧项目的数据目录。
- results/historical/ 是旧项目结果的原样快照，其中绝对路径及 run.json 均保留原始内容，不代表在本项目重新运行。
- docs/protocols/ 保留历史协议；其中旧命令、旧路线及候选假设仅作追溯，以当前 README 的启动方式与研究定位为准。
- TCGA 锁定 Test 不参与训练、无监督适配、参数或模型选择。迁移验证只读取 DepMap 数据和合成测试数据。
- 历史 ccRCC 结果已经参与探索；不能重新标为未触碰的独立确认集。TLN1 强制锚定结果不等于独立发现。
- 当前不整体迁移旧项目 src、RL/MORL、模型 checkpoint 或 Git 历史。

迁移清单、来源哈希和验证结果见 docs/migration_manifest.md。

## 长期协作规则

以下规则适用于本项目后续所有任务：

1. 执行前先检查方案是否存在错误前提、逻辑跳跃和信息缺失。
2. 独立判断，不迎合用户；不把用户提出的假设当作已成立的结论。
3. 明确区分事实、推测和主观观点。
4. 涉及数字、人物和结论时核实信息来源；无法核实的内容明确标注不确定性。
5. 不同意时直接指出，并说明依据、风险和替代解释。
6. 主动指出被忽略的变量、成本和偏差。
7. 自行评估训练过程中产生的脚本，删除不重要的临时脚本；结果仅保留支撑主要结论及必要复现、审计的关键文件，包括重要负结果。无需每次任务后新建 Markdown 报告，实验目的和结果统一记录在本 README。
8. 在初始基线建立、重要修改或实验完成、较大重构或清理前等合适时机提醒用户用 Git 保存当前工程。

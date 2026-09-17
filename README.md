# PanCancer-Dependency

## 项目概览

本项目研究一个问题：**能否利用大量癌细胞系中已经测量的基因表达和基因依赖数据，学习一套可迁移的方法，再从透明细胞肾细胞癌（clear cell renal cell carcinoma，ccRCC）患者的肿瘤表达数据中筛选值得实验验证的依赖候选？**

这里的“基因依赖”是指癌细胞在某个基因被敲除或抑制后，生长或存活受到影响。“功能依赖”需要基因扰动实验直接证明。患者肿瘤的 RNA 表达只能帮助提出候选，不能代替功能实验。

项目只使用公开数据。主线已经从泛癌细胞系训练推进到跨平台验证、患者表达迁移、候选冻结和多层证据审查。当前形成的是一套**候选优先级流程和冻结的实验候选集**，尚未证明患者特异的 ccRCC 功能依赖，也没有建立药物疗效、安全性或正常肾治疗窗。DDQN 和多目标强化学习不属于当前研究主线。

## 研究目的

项目依次回答五个问题：

1. 泛癌细胞系中的分子信息能否预测一个细胞系依赖哪些基因？
2. 模型学到的是普遍必需基因，还是具有细胞系差异的选择性依赖？
3. 这种信号能否从 Broad/DepMap CRISPR 数据迁移到独立的 Sanger CRISPR 和 DRIVE RNAi 平台？
4. 在没有患者功能标签的条件下，模型给出的 ccRCC 患者候选能否在独立患者分组中保持稳定？
5. 冻结候选是否同时具备独立功能证据、ccRCC 特异性和可接受的正常肾风险？

预期目标不是直接从计算结果宣布治疗靶点，而是建立一条可审计的筛选流程，缩小后续基因扰动实验需要验证的范围，并清楚标明每个结论的证据等级。

## 整体流程

```text
DepMap 泛癌 CRISPR 与多组学数据
              │
              ▼
完整癌系留出训练与模态消融
              │
              ▼
去除 common-essential 后预测选择性依赖
              │
              ▼
Sanger CRISPR 与 DRIVE RNAi 跨平台复现
              │
              ▼
TCGA-KIRC 患者表达跨域映射
              │
              ▼
Train 发现 → Validation 稳定性 → 锁定 Test 一次性确认
              │
              ▼
文献、正常肾表达、类器官风险与可成药性审查
              │
              ▼
冻结 20 个候选及其证据边界
```

训练和外部验证始终按患者或完整癌系隔离，避免同一患者来源模型同时进入训练与评价。患者端没有基因敲除后的真实结果，因此只评价排序稳定性和输入敏感性，不把它们称为预测准确率。

## 数据和评价方法

- **DepMap 24Q4**：873 个癌细胞系、15,835 个共有基因，包含表达、拷贝数、突变和 CRISPR 依赖；其中明确注释的 ccRCC 模型为 12 个。
- **Sanger Project Score**：用于检验 Broad/DepMap 上训练的规律能否迁移到另一套 CRISPR 实验平台。
- **DRIVE-only DEMETER2**：使用独立 RNAi 扰动技术复查候选方向，避免只依赖 CRISPR。
- **TCGA-KIRC**：156 个 Train、53 个 Validation、51 个锁定 Test 患者；Test 在协议和代码提交后只正式访问一次。
- **HPA、GTEx 和正常肾类器官文献**：定位正常肾脏和肾单位细胞中的表达或功能风险。
- **Open Targets 与同行评议文献**：核查可成药性、直接 loss-of-function 支持和反向证据。

主要排序指标为 NDCG，它衡量真正强依赖基因是否被排在候选列表前部；数值越高，头部排序越好。Spearman 衡量两份完整排序或候选频率的一致程度。Top-10/Top-100 重叠衡量两种设置给出的头部候选有多少相同。所有这些指标都必须结合评价数据是否具有功能真值来解释。

## 阶段成果

### 阶段一：建立泛癌功能依赖基线

在 19 个癌系上进行完整癌系留出：每次用其他癌系训练，再预测完全未参与训练的癌系。多组学模型在 19/19 个癌系上优于训练基因均值。非 common-essential 基因的癌系等权 NDCG 从 0.6167 提升到 0.7009，Top-10 依赖命中率从 0.7131 提升到 0.8305。

模态消融显示，表达提供了绝大多数可复现的预测信息：仅表达模型的 NDCG 为 0.6967，与完整多组学模型的 0.7009 接近；去掉表达后降至 0.6418。拷贝数提供小幅但可检测的泛癌排序增益，突变矩阵在当前线性模型中没有显示额外稳定增益。这个结论只适用于本项目的模型和输入，不能解释为突变或拷贝数在癌症生物学中不重要。

相关结果位于 `outputs/depmap_baseline_lolo_v1/`、`outputs/depmap_baseline_lolo_v2/`、`outputs/depmap_baseline_lolo_v3/`、`outputs/depmap_ablation_v1/`、`outputs/depmap_ablation_bootstrap_v1/` 和 `outputs/depmap_cn_control_v1/`。

### 阶段二：从普遍依赖转向选择性依赖

早期外部验证发现，模型能提高 Top-10 排序，但全基因相关性几乎不变，说明结果容易被所有细胞共同需要的基因主导。因此后续评价排除了 release-wide common-essential，并预测相对训练基因均值的依赖残差。

在 805 个可评价模型、19 个完整癌系留出中，表达残差模型相对训练选择性先验的癌系等权 NDCG 从 0.2248 提升到 0.4178，增量为 +0.1930；患者等权 bootstrap 95% 区间为 +0.1792 至 +0.2060，19/19 个癌系方向为正。这证明表达对“哪些模型比平均水平更依赖某个基因”的排序有稳定信息。

12 个 ccRCC 模型的增量更大，但相对其他 Kidney 模型的额外增益区间跨 0，因此没有证明该方法对 ccRCC 特别有效。结果位于 `outputs/selective_dependency_lolo_v1/` 和 `outputs/selective_dependency_analysis_v1/`。

### 阶段三：跨实验平台验证

在预测前冻结的 66 个纯 Sanger 泛癌模型上，Broad 直接表达模型相对训练选择性先验的 ΔNDCG 为 +0.1065，95% 区间为 +0.0776 至 +0.1360，54/66 个模型和 15/15 个癌系方向为正。这说明表达驱动的选择性排序能够跨 CRISPR 平台复现。

但绝对 NDCG 只有 0.1670，真实 Top-10 重叠只有 0.0303，说明外部平台上的具体基因名单仍不够准确。需要表达映射的 17 个模型中，ΔNDCG 为 +0.0576，区间 -0.0099 至 +0.1285，未达到稳定复现。单个肾癌模型表现较好也不能证明 ccRCC 特异性。

结果位于 `outputs/sanger_phase_a_769p_v1/`、`outputs/sanger_phase_b_v1/`、`outputs/sanger_external_controls_v1/`、`outputs/sanger_external_controls_analysis_v1/` 和 `outputs/sanger_selective_dependency_v1/`。

### 阶段四：患者表达迁移和候选冻结

模型使用 850 个非 Kidney DepMap 模型训练，再把 TCGA-KIRC 患者表达映射到细胞系表达空间。患者端没有依赖标签，因此这一阶段只能检验候选排序是否稳定。

Train 与 Validation 的活跃候选频率 Spearman 为 0.5645，Top-100 重叠为 0.7400。driver 中性化几乎不改变排序，但使用非 Kidney 或 Kidney 表达偏移进行映射时，Top-10 重叠只有 0.3651，说明跨域映射是主要不确定性。项目在查看外部候选证据之前冻结了 discovery rank 前 20，后续没有根据 Validation、Test、文献或正常组织结果重排。

冻结候选依次为：SEPHS2、GRB2、HNF1B、SEPSECS、CFLAR、YRDC、PAX8、EEFSEC、SLC33A1、CHMP7、YPEL5、HSD17B12、PSTK、FOXA1、UBR5、CCND1、GCN1、UXS1、FERMT2、WDR73。

结果位于 `data/processed/tcga_kirc_expression_bridge_v1/`、`outputs/tcga_patient_transfer_v1/` 和 `outputs/tcga_candidate_evidence_v2/`。

### 阶段五：锁定 Test 一次性确认

Test 协议、模型、输入映射、候选、指标和重点基因预期方向先以 Git 提交 `73f9972` 固定，随后正式访问 51 个 Test 患者一次，没有使用 Test 调参、选映射或重排候选。

Train–Test 活跃基因频率 Spearman 为 0.5450，Top-100 重叠为 0.75；Validation–Test 分别为 0.4500 和 0.70。冻结前 20 的 Validation–Test 频率 Spearman 为 0.7984，说明固定候选在未见队列中的相对出现频率较稳定。

映射不确定性同样在 Test 中出现：两种预设映射的患者 Top-10 平均重叠为 0.3608，95% 区间为 0.3157 至 0.4098。14 个配对肿瘤—邻近正常样本中，PAX8、FERMT2 和 CCND1 的预设表达方向复现且区间不跨 0；HNF1B 中位数方向一致，但区间跨 0，不能称为确认。该 Test 没有功能标签，只确认未见患者中的稳定性和表达方向。

协议见 `configs/tcga_locked_test_protocol_20260916.json`，结果位于 `outputs/tcga_locked_test_v1/`，版本化快照位于 `results/historical/tcga_locked_test_v1/`。

### 阶段六：候选的独立功能和正常肾风险审查

DRIVE-only RNAi 数据包含 397 个模型，其中映射出 8 个 ccRCC。冻结候选中有 10 个被该文库覆盖。PAX8、HNF1B 和 FERMT2 的 ccRCC 依赖残差区间完全低于 0，支持 Kidney/ccRCC 谱系依赖方向；只有 PAX8 相对其他 Kidney 模型的区间也完全低于 0。由于 DRIVE 与当前 CRISPR 基线存在部分模型重叠，这属于独立扰动平台复现，不是完全独立患者队列验证。

正常组织证据同时暴露出风险。GTEx v11 中 PAX8 的 Kidney Cortex/Medulla 中位表达为 179.12/351.62 TPM，HNF1B 为 51.63/97.90 TPM，FERMT2 的肾脏最高中位表达为 33.46 TPM。HPA 肾脏单细胞数据也显示三者位于多种肾单位细胞。公开正常肾类器官扰动文献进一步显示，PAX8 或 HNF1B 被抑制会影响肾类器官形成、分化或细胞周转。这些证据提示风险，但不能直接换算成人体药物毒性。

同行评议的直接功能文献支持集中在 PAX8 和 CCND1；YPEL5 存在反向结果，即在 786-O 细胞中敲低后增殖、迁移和侵袭增强。CCND1 具有肿瘤高表达和临床阶段 tractability 记录，但 DRIVE 的绝对依赖区间跨 0。

结果位于 `outputs/candidate_external_validation_v1/`、`outputs/candidate_external_adjudication_v1/`、`outputs/candidate_orthogonal_validation_v1/` 和 `outputs/candidate_celltype_window_v1/`。

### 阶段七：患者来源功能数据审计

项目检查了公开的 Sanger 肿瘤类器官 CRISPR、Broad NextGen Dependency Map、ccRCC 类器官、PDX 和相关筛选研究。预先要求数据必须同时满足：人患者来源 ccRCC、候选基因 loss-of-function、细胞生长或肿瘤终点，并独立于现有传统细胞系。

在冻结审计范围内，符合条件的公开资源为 0。现有 ccRCC 类器官研究主要提供药物或 CAR-T 反应，相关基因筛选大多仍在 786-O、Caki-1 或其他传统细胞系完成。这个结果表示目前缺少可用数据，不表示候选一定无效。

结果位于 `outputs/ccrcc_primary_functional_audit_v1/`，版本化快照位于 `results/historical/ccrcc_primary_functional_audit_v1/`。

### 阶段八：证据整合、图件和复现封装

项目把 20 个冻结候选的计算排序、CRISPR、RNAi、直接文献、TCGA 表达、HPA/GTEx 正常肾暴露、正常肾类器官风险和可成药性整理成不使用综合分数的证据矩阵。历史 v1 生成在锁定 Test 之前并保留为审计版本；当前 v2 已加入 Test 候选频率、映射敏感性和四个预设基因的表达方向，但没有改变候选顺序、功能证据等级或结论上限。

论文图件 v2 包含 4 张 300 dpi PNG 和 4 份矢量 PDF，分别展示公平 baseline、内部与外部复现、Locked Test 稳定性、证据递减、候选证据矩阵和功能—正常肾暴露关系。发布校验脚本已核对 6 个版本化结果目录中的 38 个文件哈希、冻结候选、结论上限、两份冻结协议、环境版本和图像文件签名。外部输入校验覆盖 9 个主流程文件，共 1.33 GiB。关键脚本已移除个人机器绝对路径，并固定公开数据与软件版本。

相关产物位于 `outputs/final_evidence_synthesis_v2/`、`outputs/final_figures_v2/`、`results/historical/` 和 `configs/publication_release_manifest.json`。v1 仍保留用于追踪 Test 解锁前后的变化。

### 阶段九：强 baseline、公平比较与既有工作定位

比较协议在运行前以 Git 提交 `170d62e` 固定。19 个 DepMap 完整癌系使用相同训练/留出模型、相同患者隔离、相同非 common-essential 基因、相同选择性残差目标和相同 NDCG@10 主指标。非平凡 baseline 的参数只在对应外层训练数据内按完整癌系分组选择，不读取 Sanger 或 TCGA Test 标签。比较方法包括训练选择性先验、仅注释岭回归、表达近邻、低秩表达 PCR-ridge、历史冻结表达核岭和训练内调参表达核岭。

癌系等权 NDCG 分别为 0.2248、0.1713、0.3350、0.4411、0.4177 和 0.4401。历史冻结核岭仍显著优于选择性先验，患者等权 ΔNDCG 为 +0.1926（95%区间 +0.1792 至 +0.2059），也优于表达近邻 +0.0771（+0.0684 至 +0.0857）；但它稳定低于 PCR-ridge，冻结核岭减 PCR 的差为 -0.0233（-0.0288 至 -0.0180）。因此原结论“表达包含可迁移的选择性依赖信息”成立，但“历史固定核岭是最强实现”不成立。训练内调参核岭达到 0.4401，说明主要差异来自正则化和低秩控制，而不是需要更复杂的深度模型。

同一批 DepMap 所选参数随后原样用于冻结 Sanger 外部队列。Pleura 和 Prostate 各只有一个外部模型，但在 DepMap 中没有达到20模型的合法调参癌系，因此没有借用其他癌系参数；公平比较覆盖 64/66 个模型、13个癌系。癌系等权 NDCG 中，选择性先验为 0.0566、表达近邻 0.1820、PCR-ridge 0.2287、冻结核岭 0.1982、调参核岭 0.2367。冻结核岭减 PCR 的患者等权差为 -0.0355（95%区间 -0.0577 至 -0.0148），确认 PCR 的相对优势能够跨到 Sanger；但所有外部绝对指标仍不足以直接给出患者治疗靶点。

与 Shi 等人在 2024 年 Nature Cancer 发表的 TCGA-DEPMAP 工作相比，两者都发现表达是依赖预测的主要信息，并把细胞系模型迁移到 TCGA。该工作使用较早 DepMap 的绝对 CERES 依赖、逐靶点 elastic net、筛选后的 1,966 个可预测模型，以及 quantile normalization 和 contrastive PCA，覆盖泛癌 TCGA、PDX、GTEx、药物反应和合成致死实验。本项目使用 DepMap 24Q4，重点预测 common-essential 校正后的选择性残差，采用更严格的完整癌系留出、Sanger CRISPR 和 DRIVE RNAi 复现、ccRCC 冻结候选、一次性患者 Test 与正常肾风险审计。DeepDEP 则通过 TCGA 无标签预训练和深度网络完成泛癌迁移。由于三者的数据版本、靶点集合、目标定义和划分不同，本项目没有把论文中已发表的指标伪装成同数据胜负比较，也不宣称全面优于 TCGA-DEPMAP 或 DeepDEP。差异表位于 `configs/prior_work_comparison_20260916.csv`。

结果位于 `outputs/selective_dependency_benchmark_v1/`、`outputs/sanger_baseline_benchmark_v1/`、`outputs/final_evidence_synthesis_v2/` 和 `outputs/final_figures_v2/`，冻结协议见 `configs/dependency_benchmark_protocol_20260916.json`。

### 阶段十：同数据高级模型复现（进行中）

本阶段检验增加算法复杂度能否在完全相同的 DepMap 24Q4、完整癌系留出和选择性残差评价中超过 PCR-ridge。预注册方法包括逐靶点 Elastic Net、使用官方表达编码器与 CGP 指纹结构的 Exp-DeepDEP 同数据适配版，以及共享靶点低秩结构的多任务 reduced-rank ridge。主比较限定在官方 DeepDEP 1,298 个默认靶点经 HGNC 规范化后与当前 DepMap 相交的 1,204 个靶点，其中 911 个非 common-essential 为主评价集；所有方法共享输入、外层模型和评价基因，不读取或重新使用 TCGA KIRC Locked Test。冻结协议见 `configs/advanced_model_benchmark_protocol_20260917.json`。

## 最终成果

### 方法层面的结论

1. 表达是当前线性模型预测泛癌选择性依赖的主要信息来源。
2. 表达残差模型相对简单选择性先验的增益在完整癌系留出中稳定，并在独立 Sanger CRISPR 平台上得到有限但方向一致的复现。
3. common-essential 校正是必要步骤；未校正时，较好的 Top-10 指标会被跨模型共享依赖显著影响。
4. 患者迁移可以产生具有一定队列稳定性的候选，但结果对表达域映射明显敏感。
5. 锁定 Test 支持候选频率和部分表达方向的未见队列稳定性，没有提供患者功能准确率。
6. 历史冻结核岭不是最强 baseline；PCR-ridge 和训练内调参核岭在内部和 Sanger 外部均更好。项目贡献应定位为严格评价、跨平台审计和 ccRCC 候选证据整合，而不是新的最优预测算法。

### 候选层面的结论

| 候选 | 已有支持 | 主要冲突或缺口 | 当前合理定位 |
|---|---|---|---|
| PAX8 | CRISPR、RNAi、直接 loss-of-function 文献方向最一致；RNAi 中显示相对其他 Kidney 的负向信号 | 正常肾高表达、正常肾类器官功能风险、肿瘤相对正常低表达 | 最值得做机制与治疗窗实验的候选，尚不是已验证治疗靶点 |
| HNF1B | RNAi 绝对依赖复现，非同行评议直接扰动资料支持 | ccRCC 相对其他 Kidney 区间跨 0；正常肾暴露和类器官风险；Test 表达区间跨 0 | Kidney 谱系机制候选 |
| FERMT2 | RNAi 绝对依赖复现，Test 表达方向复现 | 缺少亚型特异性、直接功能文献和患者来源验证 | Kidney 谱系候选 |
| CCND1 | 直接文献、肿瘤高表达和 tractability 记录；Test 表达方向复现 | DRIVE 绝对依赖区间跨 0，未建立 ccRCC 特异依赖 | 可成药性与表达候选，功能复现不足 |
| YPEL5 | 计算排序入选 | 存在同行评议直接反向证据 | 不宜作为优先抑制靶点 |

GRB2、CFLAR、YRDC 和 CHMP7 目前只有三个冻结 Sanger 肾癌模型一致的细胞系跨平台信号。其余候选仍主要是计算假设，不能把“没有检索到证据”解释为无效。

### 项目最终能够支持的表述

> 本项目建立了一套经过完整癌系留出和外部平台检查的、由表达驱动的选择性依赖优先排序流程，并得到一个保持原始发现顺序的 ccRCC 实验候选集。患者队列结果支持候选稳定性，但不构成患者功能依赖验证。

当前结果**不能**支持以下表述：

- 已发现或证实患者特异的 ccRCC 功能依赖；
- 某个候选已经具有正常肾治疗窗；
- 某个候选已经被证明具有临床疗效或安全性；
- 患者表达排序等同于药物反应或基因敲除结果；
- PAX8、HNF1B、FERMT2 或 CCND1 已经是可直接转化的治疗靶点。

## 现有不足和偏差来源

1. **缺少患者功能真值。** TCGA 只有表达和临床信息，没有对同一患者肿瘤进行候选基因敲除后的生长结果。这是当前最关键且无法通过增加模型复杂度解决的缺口。
2. **患者跨域映射不稳定。** 两种合理表达映射的 Top-10 重叠约为 0.36，说明具体患者候选可能随映射假设改变。
3. **外部平台绝对准确率仍低。** Sanger 上平均增益为正，但真实 Top-10 重叠很低，模型更适合缩小实验范围，不适合直接给出治疗决定。
4. **ccRCC 样本量有限。** DepMap 中明确 ccRCC 只有 12 个，DRIVE 中映射得到 8 个；亚型特异性区间普遍较宽。
5. **正常组织表达不是毒性实验。** HPA 和 GTEx 能提示正常肾暴露，不能估算药物剂量、安全窗或人体不良反应。
6. **细胞系不能完整代表患者肿瘤。** 培养条件、克隆选择、免疫和微环境缺失都会限制外推。
7. **common-essential 注释来自整个 DepMap 发布版。** 它不是每个训练折内部重新估计，因此存在轻微的信息边界问题。
8. **候选文献审查不是系统综述。** 固定关键词可能漏掉摘要中未写出基因或扰动方式的研究，零命中不等于零证据。
9. **baseline 范围仍有限。** 本次已公平比较先验、注释岭回归、表达近邻、PCR-ridge 和核岭，但没有在相同 DepMap 24Q4 与完整癌系留出下重建 TCGA-DEPMAP 的逐靶点 elastic net 或 DeepDEP。现有结果不能用于宣称优于这些已发表方法。
10. **尚未完成第二台机器 clean-room 重跑。** 哈希、环境和入口校验均已通过，但还不能宣称从原始公开数据到最终结果已经被独立机器完整复现。
11. **Locked Test 不能再用于新模型确认。** Test 已按冻结协议正式访问一次；v2 可以汇总原冻结模型的既有结果，但之后开发的 PCR 或调参核岭即使在 Test 上运行，也只能标为事后探索，不能作为新的前瞻验证。

## 下一步工作

当前正在进行同数据高级模型复现，以判断逐靶点 Elastic Net、Exp-DeepDEP 同数据适配版和多任务低秩模型能否在冻结口径下超过 PCR-ridge。该阶段完成前，不把模型复杂度写成性能改进；新模型不得再次使用已经访问的 Locked Test 进行选择或确认。

如果目标是把论文结论从“计算优先排序”提升为“ccRCC 功能靶点”，下一步所需的不是继续训练相似模型，而是新的实验数据：优先在患者来源 ccRCC 类器官或短期原代模型中对 PAX8、HNF1B、FERMT2、CCND1 等候选进行 loss-of-function 验证，并在正常肾类器官中使用相同扰动和可比较终点评估治疗窗。

## 使用说明

在 WSL 中使用 `/home/liliang/miniconda3/envs/rl_genrisk/bin/python`。精简依赖见 `requirements.txt`；全新安装环境尚未验证。

- 数据构建入口：`scripts/build_depmap_baseline.py`。
- 基线训练入口：`scripts/run_depmap_baseline.py`，通过 `--help` 查看参数，`--dry-run` 只检查输入与分组；`--alpha-scan` 额外对外层全部 α 打分并输出排序-α 诊断表，`--ablation` 增加模态消融配置（留一与单模态，共用分块核矩阵）。两者默认关闭、不改变 `metrics.csv`。消融配置中所有方法均保留 lineage 与 5 个 driver 指示作为共享注释基线，`drop_X` 同时去掉 X 的基因级矩阵与逐模型均值；因此"仅突变"应读作"注释基线 + 突变模态"，5 个 driver 本身即突变信息。
- 选择性残差入口：`scripts/run_selective_dependency.py`；GPU 聚类不确定性分析为 `scripts/analyze_selective_dependency.py`。前者固定 α=1e5 和训练折内第 10 百分位先验，拒绝覆盖已有输出；后者以患者和癌系分别作为 bootstrap 单位。
- 模型计算默认 `--device cuda`，使用现有环境中的 PyTorch CUDA；不可用时明确报错，不自动退回 CPU。文件读写、注释处理和结果整理仍使用 CPU。运行记录保存实际设备与数值精度。
- 新版输入：`data/processed/depmap_baseline_24q4_v1/`，包含 873 个癌症模型和 15,835 个四模态共有基因，明确 ccRCC 12 个模型。来源、筛选规则和校验和见该目录的 `audit.json`。
- 新版入口读取 NPZ；历史 `run_depmap_lolo_validation.py` 读取旧 TSV，二者输入不兼容。
- 正式训练由用户手动启动。输出目录已存在时拒绝覆盖。训练入口用于评价，不保存可部署的最终模型。
- 主要结果为 `metrics.csv`，必要明细与运行信息保存在同目录的压缩表、`tuning.csv`、`splits.csv` 和 `run.json`；启用 `--alpha-scan` 时另有 `alpha_scan.csv`。`run.json` 的 `feature_configs` 逐方法记录该配置包含哪些特征块，是消融结果可复现的必要依据。
- 原始输入和常规运行输出不纳入 Git；关键论文结果与图件的版本化快照保存在 `results/historical/`。
- 一条命令核验发布快照：`python scripts/verify_publication_release.py`。
- 完整重跑前设置 `PANCANCER_SOURCE_ROOT=/absolute/path/to/rl-genrisk-main`，再用 `python scripts/verify_external_inputs.py` 核验外部输入。

## 长期协作规则

1. 执行前先检查方案中的错误前提、逻辑跳跃和信息缺失。
2. 独立判断，不迎合用户，不把假设当作已成立的结论。
3. 明确区分事实、推测和主观观点。
4. 涉及数字、人物和结论时核实来源；无法核实则明确说明。
5. 不同意时直接指出，给出依据、风险和替代解释。
6. 主动指出被忽略的变量、成本和偏差。
7. 自行清理不重要的临时脚本与冗余结果，保留主要结论、重要负结果及必要复现与审计材料。
8. README 只在大型任务开始和最终完成时记录目的、范围及结果；不按每次对话或子任务追加中间过程，不另建 Markdown 过程报告。长期规则和必要使用说明发生变化时可更新相应内容。
9. 每次任务结束，需要用户手动执行的操作必须在回复中直接给出可复制命令，不能只让用户查看 README；明确指出下一步要做什么。没有手动操作时不编造操作要求。
10. 在初始基线、重要修改或实验完成、较大重构或清理前等合适时机提醒用户用 Git 保存工程。
11. 后续运行任务使用简洁中文终端输出，清楚显示当前阶段、进度、耗时、关键评价结果及保存位置；明确区分程序检查和正式实验。
12. 后续模型训练及可加速的数值计算优先并默认使用 GPU；不静默退回 CPU。文件读写等不适合 GPU 的操作仍由 CPU 处理。

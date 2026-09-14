# DepMap 泛癌迁移、模块稳定性与外部验证协议（2026-09-12）

## 0. 结论边界

- 当前证据支持继续检验“多组学能否预测个体细胞系的功能依赖”，但尚不能把它写成已经成立的泛癌规律。
- 当前证据不支持序列决策优于即时或静态方法；DDQN 暂停，且不参与本协议前两阶段。
- TLN1 锚定模块仍是探索性对象。20-seed 稳定性和 Sanger 复现通过前，论文标题不得把它表述为已验证机制。
- 锁定的 TCGA test 在所有模型、超参数、模块成员、药物假设和阈值冻结前不得读取，只允许最终评估一次。

## 1. 泛癌 leave-one-lineage-out（LOLO）

入口：`scripts/run_depmap_lolo_validation.py`

### 冻结口径

- 外层：DepMap `OncotreeLineage` 样本数不少于 20 的每个癌系整体留出。
- 内层：剩余癌系按完整 lineage 分组做 5-fold；留出癌系不得参与 alpha 选择。
- 比较：`global_mean`、`background`、`targetwise_multiomics`、`shared_multiomics`。
- 主调参指标：内层 aggregate SSE 最小；因此 ΔR² 是主指标，其余排序指标为次指标。
- ΔR²：相对仅由外层训练癌系估计的逐靶点 global mean。
- NDCG@10：相关性定义为 `max(0, -Chronos GeneEffect)`。
- Top-10 overlap：预测 Top-10 与真实最强依赖 Top-10 的交集比例。
- Dependency precision：预测 Top-10 中 `GeneEffect <= -0.5` 的比例。
- Regret：预测 Top-10 的平均 GeneEffect 减去 oracle Top-10；越低越好。
- Kidney 只与其他同样 LOLO 的癌系比较。脚本只报告其百分位和其他癌系分布，不自动宣布“普遍迁移”。

### 执行顺序

```bash
/home/liliang/miniconda3/envs/rl_genrisk/bin/python scripts/run_depmap_lolo_validation.py --dry-run

/home/liliang/miniconda3/envs/rl_genrisk/bin/python scripts/run_depmap_lolo_validation.py \
  --output-dir outputs/depmap_lolo_validation_20260912
```

正式运行会逐个打印外层癌系、内层 fold 和被选择的 alpha。

## 2. 20-seed 模块稳定性

入口：`scripts/run_depmap_module_search.py`

### 冻结口径

- 默认 20 个预先固定的 CV split seed：20260911–20260930。
- 比较 `static_cv_ppi`、`greedy_ppi`、`beam_ppi`；`frequency_ppi` 保留为额外透明基线。
- 随机模块同时匹配事件基因数和 PPI 总节点数。
- TLN1、AJUBA、CTNNA1、DSP、DSC3、TP53 分别报告：作为事件基因入选频率、作为任意模块节点入选频率。连接节点不得误写成被模型选中的事件基因。
- 每个 seed 内，adaptive 方法只能根据 non-kidney CV objective 在 greedy/beam 中选择；ccRCC holdout 不参与选择。
- 主比较：adaptive ccRCC ΔR² 减 `static_cv_ppi` ccRCC ΔR²。
- 95% 区间由 20 个算法/CV split seed bootstrap 得到，只表示算法不稳定性，不代表 20 个独立生物队列。
- 停止门：均值 `<0.005`，或 bootstrap 下界 `<=0`，则输出 `STOP_DDQN_ROUTE`。

### 执行

```bash
/home/liliang/miniconda3/envs/rl_genrisk/bin/python scripts/run_depmap_module_search.py
```

关键输出：

- `method_stability_summary.csv`
- `tracked_gene_selection_frequency.csv`
- `adaptive_vs_static_gate.csv`
- `random_module_controls.csv.gz`

## 3. 外部验证：必须在模块冻结后

### Project Score / Sanger CRISPR

- 先固定 TLN1 模块成员和方向，再下载/读取 Sanger 结局。
- 先审计 Broad 与 Sanger 的细胞系交集；共享细胞系复现和非共享细胞系迁移分开报告。
- 报告连续 gene-fitness 效应、依赖比例和跨平台相关，不只报告显著性。
- Project Score 与 Broad DepMap 是不同实验管线，但共享细胞系时不能称为完全独立样本验证。

### PRISM / GDSC

- “模块状态”计算公式必须先在 DepMap/TCGA development 数据中冻结。
- 药物或药物类别必须按预先固定的靶点映射选择；若扫描全部药物，必须明确作为 discovery，并做 FDR 校正，再用另一药物数据集确认。
- PRISM 与 GDSC 的 assay、药物覆盖和细胞系覆盖不同，不能直接合并 AUC 数值。

### TCGA

- development 数据只用于冻结纯度校正、分期协变量、模块分数和缺失值规则。
- 推荐预先固定 OS 与 PFI，Cox 模型至少调整年龄、性别、分期和纯度；检查比例风险假设。
- 生存关联不是功能依赖或治疗获益的因果证明。
- 锁定 test 最终只运行一次，并完整报告效应量、置信区间和失败结果。

## 4. RL 条件路线

- 单基因 DepMap 是全反馈矩阵，监督学习是主方法。
- 只有公开双基因 CRISPR 数据与当前候选有足够交集，且能构造严格的逐轮预算模拟时，才启动 RL 分支。
- SCHEMATIC 覆盖 67×176 个基因、7 个细胞系；必须先取得正式基因表并计算与冻结候选的交集，不能仅根据论文摘要假设可用。
- NAIAD 是主动学习方法，不是数据集名称；应使用其公开代码所引用的组合扰动数据，并单独记录数据来源。
- 奖励使用预先定义的双基因非加和效应。比较 random、MPE、UCB、Thompson sampling、contextual bandit、DDQN。
- 只有 DDQN 在强协同组合发现数和样本效率上跨数据集/seed 稳定超过 UCB 与 contextual bandit，才保留 RL。


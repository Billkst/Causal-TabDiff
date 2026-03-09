# 数据与评估管线审计报告

生成日期：2026-03-08

## 结论摘要

- 当前训练主表只使用 `prsn`，即 [src/data/data_module.py](src/data/data_module.py#L83-L90) 中的 `self.prsn_df`；`screen`、`ctab`、`ctabc`、`canc` 均未并入训练特征。
- 当前 `Y` 来自 `prsn` 表的 `cancyr`，并在训练/评估中通过 `y > 0.5` 二值化；因此正类实际定义为 `cancyr ∈ {1,2,3,4,5,6,7}`，负类为 `0`。
- 这不是“1 年窗”标签，也不是“within y years”可配标签；当前实现更接近“基线后任意随访年首次肺癌为正类”，且 `T0` 癌症 (`cancyr=0`) 会被并入负类。
- 训练输入张量是 `[B, 3, 49]`，但 3 个时间步来自对同一份 `prsn` 静态宽表的重复拷贝；评估时又压成 `[B, 29]`。因此当前实现不属于严格意义上的纵向轨迹模型。

---

## 1. `data/` 与 `docs/` 全部文件清单

### 1.1 `data/`

- [data/nlst.780.idc.delivery.052821/columns.txt](data/nlst.780.idc.delivery.052821/columns.txt) — 15 行
- [data/nlst.780.idc.delivery.052821/nlst_780_canc_idc_20210527.csv](data/nlst.780.idc.delivery.052821/nlst_780_canc_idc_20210527.csv) — 2151 行，2150 行数据，34 列
- [data/nlst.780.idc.delivery.052821/nlst_780_ctab_idc_20210527.csv](data/nlst.780.idc.delivery.052821/nlst_780_ctab_idc_20210527.csv) — 177488 行，177487 行数据，12 列
- [data/nlst.780.idc.delivery.052821/nlst_780_ctabc_idc_20210527.csv](data/nlst.780.idc.delivery.052821/nlst_780_ctabc_idc_20210527.csv) — 31047 行，31046 行数据，10 列
- [data/nlst.780.idc.delivery.052821/nlst_780_prsn_idc_20210527.csv](data/nlst.780.idc.delivery.052821/nlst_780_prsn_idc_20210527.csv) — 53453 行，53452 行数据，39 列
- [data/nlst.780.idc.delivery.052821/nlst_780_screen_idc_20210527.csv](data/nlst.780.idc.delivery.052821/nlst_780_screen_idc_20210527.csv) — 75139 行，75138 行数据，20 列

### 1.2 `docs/`

- [docs/Metrics_Confirmation_Checklist.md](docs/Metrics_Confirmation_Checklist.md) — 71 行
- [docs/dataset/dictionaries/dictionary_idc_canc_idc-20210527.md](docs/dataset/dictionaries/dictionary_idc_canc_idc-20210527.md) — 146 行
- [docs/dataset/dictionaries/dictionary_idc_ctab_idc-20210527.md](docs/dataset/dictionaries/dictionary_idc_ctab_idc-20210527.md) — 124 行
- [docs/dataset/dictionaries/dictionary_idc_ctabc_idc-20210527.md](docs/dataset/dictionaries/dictionary_idc_ctabc_idc-20210527.md) — 122 行
- [docs/dataset/dictionaries/dictionary_idc_prsn_idc-20210527.md](docs/dataset/dictionaries/dictionary_idc_prsn_idc-20210527.md) — 229 行
- [docs/dataset/dictionaries/dictionary_idc_screen_idc-20210527.md](docs/dataset/dictionaries/dictionary_idc_screen_idc-20210527.md) — 129 行
- [docs/dataset/数据集详尽说明文档.md](docs/dataset/%E6%95%B0%E6%8D%AE%E9%9B%86%E8%AF%A6%E5%B0%BD%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md) — 94 行
- [docs/deployment/experiment_commands.md](docs/deployment/experiment_commands.md) — 82 行
- [docs/proposal/causal_tabdiff_outcome_readout_v2.md](docs/proposal/causal_tabdiff_outcome_readout_v2.md) — 221 行
- [docs/proposal/方案二：开题报告.md](docs/proposal/%E6%96%B9%E6%A1%88%E4%BA%8C%EF%BC%9A%E5%BC%80%E9%A2%98%E6%8A%A5%E5%91%8A.md) — 388 行
- [docs/proposal/方案二：开题报告图5.png](docs/proposal/%E6%96%B9%E6%A1%88%E4%BA%8C%EF%BC%9A%E5%BC%80%E9%A2%98%E6%8A%A5%E5%91%8A%E5%9B%BE5.png) — 二进制图片，行数不适用
- [docs/proposal/方案二：开题报告图6.png](docs/proposal/%E6%96%B9%E6%A1%88%E4%BA%8C%EF%BC%9A%E5%BC%80%E9%A2%98%E6%8A%A5%E5%91%8A%E5%9B%BE6.png) — 二进制图片，行数不适用
- [docs/proposal/方案二：开题报告图7.png](docs/proposal/%E6%96%B9%E6%A1%88%E4%BA%8C%EF%BC%9A%E5%BC%80%E9%A2%98%E6%8A%A5%E5%91%8A%E5%9B%BE7.png) — 二进制图片，行数不适用
- [docs/proposal/方案二：开题报告图8.png](docs/proposal/%E6%96%B9%E6%A1%88%E4%BA%8C%EF%BC%9A%E5%BC%80%E9%A2%98%E6%8A%A5%E5%91%8A%E5%9B%BE8.png) — 二进制图片，行数不适用

---

## 2. 原始 CSV 统计：行数、列数、主键/连接键、时间字段、标签字段、每列缺失率

缺失率单位：`%`

### 2.1 `prsn`

文件：[data/nlst.780.idc.delivery.052821/nlst_780_prsn_idc_20210527.csv](data/nlst.780.idc.delivery.052821/nlst_780_prsn_idc_20210527.csv)

- 行/列：53452 × 39
- 主键：`pid`（唯一）
- 连接键：`pid`
- 时间字段：`scr_days0`, `scr_days1`, `scr_days2`, `candx_days`, `canc_free_days`
- 标签字段：`cancyr`, `can_scr`
- 缺失率：
  - `race` 0.0
  - `cigsmok` 0.0
  - `gender` 0.0
  - `age` 0.0
  - `loclhil` 96.1498
  - `locllow` 96.1498
  - `loclup` 96.1498
  - `locrhil` 96.1498
  - `locrlow` 96.1498
  - `locrmid` 96.1498
  - `locunk` 96.1498
  - `locrup` 96.1498
  - `locoth` 96.1498
  - `locmed` 96.1498
  - `loclmsb` 96.1498
  - `locrmsb` 96.1498
  - `loccar` 96.1498
  - `loclin` 96.1498
  - `lesionsize` 96.6512
  - `de_type` 96.1947
  - `de_grade` 96.1498
  - `de_stag` 96.1517
  - `scr_res0` 0.0
  - `scr_res1` 0.0
  - `scr_res2` 0.0
  - `scr_iso0` 0.0
  - `scr_iso1` 0.0
  - `scr_iso2` 0.0
  - `cancyr` 96.1498
  - `can_scr` 0.0
  - `canc_rpt_link` 0.0
  - `pid` 0.0
  - `dataset_version` 0.0
  - `scr_days0` 2.0411
  - `scr_days1` 8.6807
  - `scr_days2` 11.2175
  - `candx_days` 96.1498
  - `canc_free_days` 0.0
  - `de_stag_7thed` 96.219

### 2.2 `screen`

文件：[data/nlst.780.idc.delivery.052821/nlst_780_screen_idc_20210527.csv](data/nlst.780.idc.delivery.052821/nlst_780_screen_idc_20210527.csv)

- 行/列：75138 × 20
- 主键：`(pid, study_yr)`（唯一）
- 连接键：`pid`, `study_yr`
- 时间字段：`study_yr`
- 标签字段：无
- 缺失率：
  - `pid` 0.0
  - `ctdxqual` 0.004
  - `study_yr` 0.0
  - `techpara_kvp` 0.0053
  - `techpara_ma` 13.1837
  - `techpara_fov` 0.378
  - `techpara_effmas` 52.2479
  - `ct_recon_filter1` 0.0053
  - `ct_recon_filter2` 63.2676
  - `ctdxqual_breath` 97.2411
  - `ctdxqual_motion` 97.2411
  - `ctdxqual_resp` 97.2411
  - `ctdxqual_techpara` 97.2411
  - `ctdxqual_inadeqimg` 97.2411
  - `ctdxqual_artifact` 97.2411
  - `ctdxqual_graininess` 97.2411
  - `ctdxqual_other` 97.2411
  - `ct_recon_filter3` 99.8123
  - `ct_recon_filter4` 99.9947
  - `dataset_version` 0.0

### 2.3 `ctab`

文件：[data/nlst.780.idc.delivery.052821/nlst_780_ctab_idc_20210527.csv](data/nlst.780.idc.delivery.052821/nlst_780_ctab_idc_20210527.csv)

- 行/列：177487 × 12
- 主键：`(pid, study_yr, sct_ab_num)`（唯一）
- 连接键：`pid`, `study_yr`, `sct_ab_num`
- 时间字段：`study_yr`
- 标签字段：无
- 缺失率：
  - `sct_ab_desc` 0.0
  - `sct_ab_num` 0.0
  - `sct_epi_loc` 80.0521
  - `sct_long_dia` 80.1518
  - `sct_margins` 80.0521
  - `sct_perp_dia` 80.1704
  - `sct_pre_att` 80.0526
  - `study_yr` 0.0
  - `sct_slice_num` 80.0521
  - `sct_found_after_comp` 0.0011
  - `pid` 0.0
  - `dataset_version` 0.0

### 2.4 `ctabc`

文件：[data/nlst.780.idc.delivery.052821/nlst_780_ctabc_idc_20210527.csv](data/nlst.780.idc.delivery.052821/nlst_780_ctabc_idc_20210527.csv)

- 行/列：31046 × 10
- 主键：`(pid, study_yr, sct_ab_num)`（唯一）
- 连接键：`pid`, `study_yr`, `sct_ab_num`
- 时间字段：`study_yr`, `visible_days`
- 标签字段：无
- 缺失率：
  - `study_yr` 0.0
  - `sct_ab_preexist` 0.0
  - `pid` 0.0
  - `sct_ab_attn` 33.3827
  - `sct_ab_gwth` 33.3795
  - `sct_ab_invg` 81.7046
  - `sct_ab_num` 0.0
  - `sct_ab_code` 0.0097
  - `dataset_version` 0.0
  - `visible_days` 15.8925

### 2.5 `canc`

文件：[data/nlst.780.idc.delivery.052821/nlst_780_canc_idc_20210527.csv](data/nlst.780.idc.delivery.052821/nlst_780_canc_idc_20210527.csv)

- 行/列：2150 × 34
- 主键：`(pid, lc_order)`（唯一）；`(pid, study_yr)` 不唯一
- 连接键：`pid`；如需区分多次肺癌记录，使用 `pid + lc_order`
- 时间字段：`study_yr`, `candx_days`
- 标签字段：无显式训练标签列
- 缺失率：
  - `pid` 0.0
  - `lc_topog` 0.0
  - `topog_source` 13.6279
  - `de_type` 1.2093
  - `de_grade` 0.0
  - `de_stag` 0.0465
  - `path_stag` 29.2558
  - `clinical_stag` 1.2558
  - `stage_sum` 29.2558
  - `valcsg` 36.6512
  - `clinical_t_7thed` 4.9302
  - `clinical_n_7thed` 3.6279
  - `clinical_m_7thed` 3.6279
  - `path_t_7thed` 31.7209
  - `path_n_7thed` 31.814
  - `path_m_7thed` 31.7209
  - `de_stag_7thed` 2.8372
  - `first_lc` 0.0
  - `lesionsize` 12.6977
  - `lc_morph` 0.0
  - `lc_behav` 0.0
  - `lc_grade` 0.0
  - `source_best_stage` 0.0
  - `clinical_t` 2.4186
  - `path_t` 29.4884
  - `clinical_n` 2.093
  - `path_n` 30.4651
  - `clinical_m` 2.0
  - `path_m` 30.2791
  - `stage_only` 26.3256
  - `study_yr` 0.0
  - `dataset_version` 0.0
  - `lc_order` 0.0
  - `candx_days` 0.0

### 2.6 `columns.txt`

文件：[data/nlst.780.idc.delivery.052821/columns.txt](data/nlst.780.idc.delivery.052821/columns.txt)

- 行数：15
- 内容是 5 个 CSV 的列头转储，不是训练特征定义文件

---

## 3. 从原始 CSV 到训练集：`y` 的构造逻辑

### Step 1：原始字典定义

在 [docs/dataset/dictionaries/dictionary_idc_prsn_idc-20210527.md](docs/dataset/dictionaries/dictionary_idc_prsn_idc-20210527.md#L194-L200)：

- `cancyr` = “Study year associated with first confirmed lung cancer”
- 编码：`.N, 0=T0, 1=T1, ..., 7=T7`

这说明原始 `cancyr` 不是天然二分类，而是“首次肺癌对应的研究年份”。

### Step 2：训练数据只取 `prsn`

在 [src/data/data_module.py](src/data/data_module.py#L83-L90)：

- 读取了 `prsn/screen/ctab/canc` 路径
- 但实际只执行：
  - `self.prsn_df = pd.read_csv(prsn_path)`
  - `self.merged_df = self.prsn_df.copy()`

即：训练主表只有 `prsn`，其他原始表没有并入训练特征。

### Step 3：标签列名

在 [src/data/generate_metadata.py](src/data/generate_metadata.py#L20-L25)：

- `y_col = 'cancyr'`

在 [src/data/data_module.py](src/data/data_module.py#L43-L45)：

- `self.y_col = self.metadata['y_col']['name']`

因此标签列名固定为 `cancyr`。

### Step 4：缺失处理

在 [src/data/data_module.py](src/data/data_module.py#L92-L96)：

- `self.merged_df[self.y_col] = self.merged_df[self.y_col].fillna(0).astype(int)`

即：

- `NaN` 被改成 `0`
- `cancyr` 保留为整数年份标签

### Step 5：最终张量中的 `y`

在 [src/data/data_module.py](src/data/data_module.py#L183-L189)：

- `self.y = self.merged_df[self.y_col].values.reshape(-1, 1)`

因此数据集中的原始 `y` 形状是 `[N, 1]`，值域是 `{0,1,2,3,4,5,6,7}`。

### Step 6：训练/评估时的二值化

在 [src/baselines/wrappers.py](src/baselines/wrappers.py#L1029-L1037) 与 [run_baselines.py](run_baselines.py#L136-L141)：

- `y_binary = (y > 0.5).float()`
- `real_y_bounds = (real_y_flat > 0.5).astype(float)`

因此：

- 正类定义：`cancyr >= 1`
- 负类定义：`cancyr == 0`

含义：

- `T1~T7` 首次肺癌 -> 正类
- `NaN -> 0` -> 负类
- `T0` 首次肺癌 (`cancyr=0`) 也被并入负类

### Step 7：预测时间窗

当前实现不是：

- 不是 “1 年”
- 不是 “多年内 y years 可配置”
- 不是 “ever within y years”

当前实现更接近：

- `ever after baseline at T1..T7`
- 同时把 `T0` 癌症错误并入负类

### Step 8：样本单位

当前不是：

- 不是 “一个人-一年”
- 不是 “一个人-多年的真实观测序列”

而是：

- 一个受试者一条样本
- 再在 [src/data/data_module.py](src/data/data_module.py#L149-L180) 中人为复制成 `T=3` 个时间步
- 这 3 个时间步并非来自 `screen/ctab/ctabc` 的真实年度序列，而是同一张 `prsn` 宽表的重复编码

因此样本单位应表述为：

- “一个人 + 3 个重复的伪时间步静态拷贝”

---

## 4. 模型实际输入张量 shape

### 4.1 数据集输出

由 [src/data/data_module.py](src/data/data_module.py#L149-L189) 可得：

- `x`: `[B, 3, 49]`
- `x_cat_raw`: `[B, 3, 23]`
- `alpha_target`: `[B, 1]`
- `y`: `[B, 1]`

其中：

- `T = 3`
- analog/编码维度 `D = 49`
- 语义列数 `D_semantic = 29`

### 4.2 主模型训练输入

在 [run_experiment.py](run_experiment.py#L47-L49) 与 [src/baselines/wrappers.py](src/baselines/wrappers.py#L1025-L1043)：

- `CausalTabDiff` 训练时直接使用 `x = batch['x']`
- 即主模型训练输入为 `[B, 3, 49]`

### 4.3 为什么这不是真纵向轨迹

因为 [src/data/data_module.py](src/data/data_module.py#L158-L180) 在 `for t in range(self.T)` 中始终从同一个 `merged_df` 取列，没有基于 `study_yr` 切片，也没有将 `screen/ctab/ctabc` 融入时序结构。

因此：

- 张量 rank 的确是 3
- 但时间语义是伪造出来的
- 当前实现不是严格意义上的纵向轨迹模型

### 4.4 评估脚本送入指标的 shape

在 [run_baselines.py](run_baselines.py#L485-L512)：

- `real_x = real_x_raw_t[:, -1, :]`
- `fake_x` 也被约束为与 `real_x` 同宽

随后在 [run_baselines.py](run_baselines.py#L97-L109)：

- `real_x_flat = real_x.reshape(real_x.shape[0], -1)`
- `fake_x_flat = fake_x.reshape(fake_x.shape[0], -1)`

由于此时 `real_x` 与 `fake_x` 已是二维，评估输入实际上为：

- `[B, 29]`

### 4.5 综合结论

- 训练张量：`[B, 3, 49]`
- 评估张量：`[B, 29]`
- 当前整体实现不是纵向轨迹评估管线

### 4.6 各 baseline 的实际输入形态

- `CausalForest` / `STaSy`：先将 `x` flatten 到 `[B, 147]`，见 [src/baselines/wrappers.py](src/baselines/wrappers.py#L47-L56) 与 [src/baselines/wrappers.py](src/baselines/wrappers.py#L286-L300)
- `TSDiff`：把 `[B, 147]` 拼上 `T,Y` 后伪装成 `[B,1,149]`，见 [src/baselines/wrappers.py](src/baselines/wrappers.py#L420-L438)
- `TabSyn` / `TabDiff`：内部做联合拼接，但本质上也是时间展开后的建模，见 [src/baselines/wrappers.py](src/baselines/wrappers.py#L518-L542) 与 [src/baselines/wrappers.py](src/baselines/wrappers.py#L796-L819)

---

## 5. 评估脚本中 `TSTR_AUC`、`TSTR_PR_AUC`、`TSTR_F1`、`TSTR_F1_RealPrev`、`TSTR_F1_FakePrev` 的计算与阈值

代码位置：[run_baselines.py](run_baselines.py#L191-L258)

### 5.1 基础标签处理

- `real_y_class = (real_y > 0.5)`
- `fake_y_class = (fake_y > 0.5)`

### 5.2 TSTR 分类器

- `tstr_model = XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=neg/pos)`
- 训练：`tstr_model.fit(fake_x_flat, fake_y_class)`
- 预测概率：`t_pred_proba = tstr_model.predict_proba(real_x_flat)[:, 1]`
- 预测类别：`t_pred_class = tstr_model.predict(real_x_flat)`

### 5.3 指标定义

- `TSTR_AUC = roc_auc_score(real_y_class, t_pred_proba)`
- `TSTR_PR_AUC = average_precision_score(real_y_class, t_pred_proba)`
- `TSTR_F1 = f1_score(real_y_class, t_pred_class)`

这里 `TSTR_F1` 的阈值并非自定义，而是 XGBoost `predict()` 的默认二分类阈值 `0.5`。

### 5.4 prevalence-aware 阈值函数

阈值函数见 [run_baselines.py](run_baselines.py#L191-L194)：

$$
\text{target\_rate} = \mathrm{mean}(\text{target\_binary}), \quad
q = 1 - \text{target\_rate}, \quad
\theta = \mathrm{quantile}(\text{score\_vec}, q)
$$

对应实现：

- `prevalence_threshold(score_vec, target_binary)` 返回 `score_vec` 的 `1 - prevalence` 分位点

随后：

- `real_prev_threshold = prevalence_threshold(t_pred_proba, real_y_class)`
- `fake_prev_threshold = prevalence_threshold(t_pred_proba, fake_y_class)`

对应指标：

- `TSTR_F1_RealPrev = f1_score(real_y_class, t_pred_proba >= real_prev_threshold)`
- `TSTR_F1_FakePrev = f1_score(real_y_class, t_pred_proba >= fake_prev_threshold)`

### 5.5 单类退化分支

若 `fake_y` 或 `real_y` 只有单类，见 [run_baselines.py](run_baselines.py#L201-L208)：

- `TSTR_AUC = 0.5`
- `TSTR_PR_AUC = mean(real_y_class)`
- `TSTR_F1 = 0.0`
- `TSTR_F1_RealPrev = 0.0`
- `TSTR_F1_FakePrev = 0.0`

---

## 6. `markdown_report.md` 全文

文件：[markdown_report.md](markdown_report.md)

```markdown
# Baseline Evaluation Results

| Model                 | ATE_Bias        | Wasserstein     | CMD             | TSTR_AUC        | TSTR_PR_AUC     | TSTR_F1         | TSTR_F1_RealPrev   | TSTR_F1_FakePrev   | Params(M)       | AvgInfer(ms/sample)   |
|-----------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|--------------------|--------------------|-----------------|-----------------------|
| Causal-TabDiff (Ours) | 0.0019 ± 0.0015 | 0.5959 ± 0.1562 | 0.4213 ± 0.1346 | 0.6342 ± 0.0444 | 0.1522 ± 0.0708 | 0.1835 ± 0.1045 | 0.1751 ± 0.0681    | 0.1721 ± 0.0648    | 0.0936 ± 0.0000 | 5.0671 ± 0.5431       |
```

---

## 7. 与癌症结局、随访年份相关的数据字典条目

### 7.1 `prsn` 中的肺癌结局条目

见 [docs/dataset/dictionaries/dictionary_idc_prsn_idc-20210527.md](docs/dataset/dictionaries/dictionary_idc_prsn_idc-20210527.md#L194-L200)

- `can_scr`：首次确认肺癌对应的筛查结果  
  `0=No Cancer, 1=Positive Screen, 2=Negative Screen, 3=Missed Screen, 4=Post Screening`
- `canc_free_days`：随机化到最后已知无肺癌日期的天数
- `canc_rpt_link`：肺癌诊断是否与阳性筛查关联
- `cancyr`：首次确认肺癌对应研究年份  
  `.N, 0=T0, 1=T1, ..., 7=T7`
- `candx_days`：随机化到首次肺癌诊断的天数

### 7.2 `canc` 中的癌症结局与诊断年份

见 [docs/dataset/dictionaries/dictionary_idc_canc_idc-20210527.md](docs/dataset/dictionaries/dictionary_idc_canc_idc-20210527.md#L102-L140)

- `candx_days`：随机化到肺癌诊断天数
- `clinical_stag` / `path_stag` / `de_stag`：临床/病理/综合分期
- `first_lc`：是否该受试者的第一例肺癌
- `lc_order`：该肺癌在该受试者所有肺癌中的顺序
- `study_yr`：诊断研究年份  
  `0=T0, 1=T1, ..., 7=T7`

### 7.3 `screen` 中的随访年份

见 [docs/dataset/dictionaries/dictionary_idc_screen_idc-20210527.md](docs/dataset/dictionaries/dictionary_idc_screen_idc-20210527.md#L115-L120)

- `study_yr`：筛查研究年份  
  `0=T0, 1=T1, 2=T2`

### 7.4 `ctab` 中的随访年份与连接键

见 [docs/dataset/dictionaries/dictionary_idc_ctab_idc-20210527.md](docs/dataset/dictionaries/dictionary_idc_ctab_idc-20210527.md#L105-L114)

- `sct_ab_num`：异常编号；与 `pid + study_yr` 联合可用于匹配
- `study_yr`：筛查研究年份  
  `0=T0, 1=T1, 2=T2`

### 7.5 `ctabc` 中的随访年份与可见时间

见 [docs/dataset/dictionaries/dictionary_idc_ctabc_idc-20210527.md](docs/dataset/dictionaries/dictionary_idc_ctabc_idc-20210527.md#L107-L113)

- `sct_ab_gwth`：异常是否发生间隔生长
- `study_yr`：筛查研究年份  
  `0=T0, 1=T1, 2=T2`
- `visible_days`：从随机化到最早可见日期的天数

---

## 8. 附：训练特征语义维度

根据 [src/data/dataset_metadata.json](src/data/dataset_metadata.json)，当前训练语义特征共 29 列：

`race`, `cigsmok`, `gender`, `age`, `loclhil`, `locllow`, `loclup`, `locrhil`, `locrlow`, `locrmid`, `locunk`, `locrup`, `locoth`, `locmed`, `loclmsb`, `locrmsb`, `loccar`, `loclin`, `lesionsize`, `scr_res0`, `scr_res1`, `scr_res2`, `scr_iso0`, `scr_iso1`, `scr_iso2`, `can_scr`, `scr_days0`, `scr_days1`, `scr_days2`

对应：

- 连续特征数：6
- 类别特征数：23
- 编码后 analog 维度总和：49

---

## 9. 最终判定

1. 当前 `Y` 不是严格定义好的临床二分类结局，而是把多分类年份标签 `cancyr` 通过 `> 0.5` 粗暴压成二值标签。
2. 当前样本单位不是“人-年”，而是“人 + 伪 3 时间步静态复制”。
3. 当前训练张量虽然是 `[B, T, D]`，但时间维不承载真实纵向轨迹；评估还原后又退化为 `[B, D]`。
4. 因此，当前实现更准确地说是“带伪时间轴的静态表格生成模型”，而非真实 longitudinal trajectory model。

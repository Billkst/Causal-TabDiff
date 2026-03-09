# Causal Treatment Candidate Audit

- Generated at: 2026-03-07T08:41:47.772858
- Data path: /home/UserData/ljx/Project_2/Causal-TabDiff/data/nlst.780.idc.delivery.052821/nlst_780_prsn_idc_20210527.csv
- Metadata path: /home/UserData/ljx/Project_2/Causal-TabDiff/src/data/dataset_metadata_noleak.json
- Sample size: 53452
- Binary outcome rate $P(cancyr>0)$: 0.0293
- Noleak kept columns: race, cigsmok, gender, age

## Verdict

- **Recommended current treatment candidate:** `cigsmok`
- **Reason:** it is the only clinically meaningful pre-treatment exposure that is both present in the current primary wide table and retained by the noleak policy.
- **Blocked candidates:** `scr_res0`, `scr_iso0`, `scr_days0`, `can_scr`
- **Reason for blocking:** they are post-randomization process variables, screening readouts, or explicit descendants of the cancer outcome.
- **Best theoretical candidate if later recovered:** `rndgroup` (screening arm), but it is not present in the current working wide table.

## Candidate Audit Table

| Column | Present | In noleak | Candidate Type | Verdict | Reason |
| --- | --- | --- | --- | --- | --- |
| cigsmok | True | True | pre_treatment_exposure | accept | T0 吸烟状态，属于基线暴露，且位于 noleak 保留列内。 |
| scr_res0 | True | False | post_treatment_screen_result | block | T0 官方筛查结果，属于筛查后的观测结果，接近中介/早期指征。 |
| scr_iso0 | True | False | post_treatment_screen_result | block | T0 isolation read，同样发生在筛查之后，不适合充当外生 treatment。 |
| scr_days0 | True | False | post_randomization_process | block | 随机化后到筛查的天数，反映流程执行，不是稳定暴露。 |
| can_scr | True | False | outcome_descendant | block | 癌症诊断关联到哪类筛查结果，直接是 outcome descendant，严格泄漏。 |
| rndgroup | False | False | ideal_trial_arm_if_available | missing | 若存在则是最干净的随机化干预臂，但当前主宽表中不可见。 |

## Empirical Snapshot

### cigsmok

- Unique values: 2
- Top counts: {0: 27692, 1: 25760}

| level | y0_rate | y1_rate |
| --- | ---: | ---: |
| 0.0 | 0.9779 | 0.0221 |
| 1.0 | 0.9629 | 0.0371 |

### scr_res0

- Unique values: 13
- Top counts: {1: 20629, 2: 18659, 4: 9578, 3: 3479, 15: 953, 17: 73, 11: 41, 10: 16, 23: 8, 24: 7}

| level | y0_rate | y1_rate |
| --- | ---: | ---: |
| 1.0 | 0.9769 | 0.0231 |
| 2.0 | 0.9730 | 0.0270 |
| 3.0 | 0.9669 | 0.0331 |
| 4.0 | 0.9534 | 0.0466 |
| 10.0 | 1.0000 | 0.0000 |
| 11.0 | 1.0000 | 0.0000 |
| 13.0 | 1.0000 | 0.0000 |
| 15.0 | 0.9759 | 0.0241 |
| 17.0 | 0.9589 | 0.0411 |
| 23.0 | 1.0000 | 0.0000 |
| 24.0 | 1.0000 | 0.0000 |
| 95.0 | 1.0000 | 0.0000 |

### scr_iso0

- Unique values: 13
- Top counts: {1: 20550, 2: 18491, 4: 9794, 3: 3510, 15: 953, 17: 73, 11: 41, 10: 16, 23: 8, 24: 7}

| level | y0_rate | y1_rate |
| --- | ---: | ---: |
| 1.0 | 0.9770 | 0.0230 |
| 2.0 | 0.9731 | 0.0269 |
| 3.0 | 0.9667 | 0.0333 |
| 4.0 | 0.9534 | 0.0466 |
| 10.0 | 1.0000 | 0.0000 |
| 11.0 | 1.0000 | 0.0000 |
| 13.0 | 1.0000 | 0.0000 |
| 15.0 | 0.9759 | 0.0241 |
| 17.0 | 0.9589 | 0.0411 |
| 23.0 | 1.0000 | 0.0000 |
| 24.0 | 1.0000 | 0.0000 |
| 95.0 | 1.0000 | 0.0000 |

### can_scr

- Unique values: 5
- Top counts: {0: 51394, 1: 928, 4: 867, 2: 181, 3: 82}

| level | y0_rate | y1_rate |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.0000 |
| 1.0 | 0.4375 | 0.5625 |
| 2.0 | 0.3702 | 0.6298 |
| 3.0 | 0.2073 | 0.7927 |
| 4.0 | 0.0000 | 1.0000 |

## Static Causal Recommendation

1. Keep the current noleak feature set unchanged for baseline fairness.
2. Replace synthetic `alpha_target` with `cigsmok` only for the next controlled audit branch.
3. Treat `race`, `gender`, and `age` as baseline confounders/background covariates, not treatments.
4. Do **not** promote `scr_res0`, `scr_iso0`, or `can_scr` into treatment, because that would reopen leakage or mediator conditioning.
5. If a future branch restores a genuine randomized arm column such as `rndgroup`, re-run this audit before switching treatment definitions.

# STaSy Leakage Diagnostic Report

- samples: 100
- device: cuda
- label_prevalence_real: 0.0200
- label_prevalence_fake: 0.0200
- label_gap: 0.0000
- tstr_auc: 0.5000
- tstr_f1: 0.0000
- tstr_ap: 0.0200
- tstr_f1_best: 0.0392
- tstr_best_threshold: 0.5000
- shuffle_tstr_auc: 0.5000
- shuffle_tstr_f1: 0.0000
- shuffle_tstr_ap: 0.0200
- shuffle_tstr_f1_best: 0.0392
- shuffle_tstr_best_threshold: 0.5000
- domain_auc: 0.9689
- nn_mem_ratio: 954947.3302
- calibrated_fake_y: True
- calibrated_fake_x: True

## Gate Verdict: FAIL
- Blocking reasons:
  - domain AUC过高(0.9689 > 0.9000)

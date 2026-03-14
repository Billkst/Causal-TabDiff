# Baseline Fix All Pass Plan

## Goal

Repair the remaining failing baselines under the official 5-seed protocol without changing seeds or the formal threshold rule, then regenerate the official tables and rewrite the final report in Chinese.

## Scope

- Keep formal seeds fixed to `42, 52, 62, 72, 82`
- Keep protocol fixed to `validation-set F1 threshold selection + fixed test threshold`
- Focus on remaining failures and likely collapse behavior:
  - `TabDiff TSTR`
  - `SSSD TSTR`
  - Chinese rewrite of `BASELINE_COMPARISON_REPORT.md`

## Steps

1. Fix label-generation collapse in `TabDiff` and collapse-prone label generation in `SSSD`
2. Keep failure markers only if a model still collapses after the fix
3. Rerun only affected 5-seed TSTR experiments for `TabDiff` and `SSSD`
4. Re-evaluate those reruns under the formal validation-F1 protocol
5. Regenerate all five formal CSV outputs
6. Rewrite `BASELINE_COMPARISON_REPORT.md` in Chinese to match final outputs
7. Run consistency checks and consult Oracle before final conclusion

## Verification

- `outputs/b2_baseline/summaries/baseline_protocol_consistency_check.csv` should show all rows passing if all baselines formally pass
- `outputs/b2_baseline/summaries/baseline_layer1_tstr.csv` should no longer contain failed `TabDiff` rows if the fix succeeds
- `BASELINE_COMPARISON_REPORT.md` should be fully Chinese and consistent with the CSV files

## Stop Condition

If `TabDiff` or `SSSD` still collapses after the minimal fix and 5-seed rerun, retain explicit failed-baseline evidence instead of fabricating a pass.

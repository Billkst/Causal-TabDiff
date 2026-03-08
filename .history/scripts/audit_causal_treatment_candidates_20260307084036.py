import json
import os
from datetime import datetime

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    'data',
    'nlst.780.idc.delivery.052821',
    'nlst_780_prsn_idc_20210527.csv',
)
NOLEAK_METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    'src',
    'data',
    'dataset_metadata_noleak.json',
)
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    'logs',
    'testing',
    'causal_treatment_audit.md',
)


def binary_outcome_rate(series: pd.Series, y_binary: pd.Series) -> pd.DataFrame:
    table = pd.crosstab(series.fillna('MISSING'), y_binary, normalize='index')
    table.columns = ['y0_rate', 'y1_rate']
    return table.reset_index().rename(columns={table.index.name or 'row_0': 'level'})


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    with open(NOLEAK_METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    kept_columns = metadata['feature_policy']['kept_columns']
    y_binary = (df['cancyr'].fillna(0).astype(float) > 0).astype(int)
    y_rate = float(y_binary.mean())

    candidate_specs = [
        {
            'name': 'cigsmok',
            'kind': 'pre_treatment_exposure',
            'reason': 'T0 吸烟状态，属于基线暴露，且位于 noleak 保留列内。',
        },
        {
            'name': 'scr_res0',
            'kind': 'post_treatment_screen_result',
            'reason': 'T0 官方筛查结果，属于筛查后的观测结果，接近中介/早期指征。',
        },
        {
            'name': 'scr_iso0',
            'kind': 'post_treatment_screen_result',
            'reason': 'T0 isolation read，同样发生在筛查之后，不适合充当外生 treatment。',
        },
        {
            'name': 'scr_days0',
            'kind': 'post_randomization_process',
            'reason': '随机化后到筛查的天数，反映流程执行，不是稳定暴露。',
        },
        {
            'name': 'can_scr',
            'kind': 'outcome_descendant',
            'reason': '癌症诊断关联到哪类筛查结果，直接是 outcome descendant，严格泄漏。',
        },
        {
            'name': 'rndgroup',
            'kind': 'ideal_trial_arm_if_available',
            'reason': '若存在则是最干净的随机化干预臂，但当前主宽表中不可见。',
        },
    ]

    lines = []
    lines.append('# Causal Treatment Candidate Audit')
    lines.append('')
    lines.append(f'- Generated at: {datetime.now().isoformat()}')
    lines.append(f'- Data path: {DATA_PATH}')
    lines.append(f'- Metadata path: {NOLEAK_METADATA_PATH}')
    lines.append(f'- Sample size: {len(df)}')
    lines.append(f'- Binary outcome rate $P(cancyr>0)$: {y_rate:.4f}')
    lines.append(f'- Noleak kept columns: {", ".join(kept_columns)}')
    lines.append('')
    lines.append('## Verdict')
    lines.append('')
    lines.append('- **Recommended current treatment candidate:** `cigsmok`')
    lines.append('- **Reason:** it is the only clinically meaningful pre-treatment exposure that is both present in the current primary wide table and retained by the noleak policy.')
    lines.append('- **Blocked candidates:** `scr_res0`, `scr_iso0`, `scr_days0`, `can_scr`')
    lines.append('- **Reason for blocking:** they are post-randomization process variables, screening readouts, or explicit descendants of the cancer outcome.')
    lines.append('- **Best theoretical candidate if later recovered:** `rndgroup` (screening arm), but it is not present in the current working wide table.')
    lines.append('')
    lines.append('## Candidate Audit Table')
    lines.append('')
    lines.append('| Column | Present | In noleak | Candidate Type | Verdict | Reason |')
    lines.append('| --- | --- | --- | --- | --- | --- |')

    for spec in candidate_specs:
        present = spec['name'] in df.columns
        in_noleak = spec['name'] in kept_columns
        verdict = 'accept' if spec['name'] == 'cigsmok' and present and in_noleak else 'block'
        if spec['name'] == 'rndgroup' and not present:
            verdict = 'missing'
        lines.append(
            f"| {spec['name']} | {present} | {in_noleak} | {spec['kind']} | {verdict} | {spec['reason']} |"
        )

    lines.append('')
    lines.append('## Empirical Snapshot')
    lines.append('')

    for name in ['cigsmok', 'scr_res0', 'scr_iso0', 'can_scr']:
        if name not in df.columns:
            continue
        rates = binary_outcome_rate(df[name], y_binary)
        lines.append(f'### {name}')
        lines.append('')
        lines.append(f'- Unique values: {df[name].nunique(dropna=True)}')
        lines.append(f'- Top counts: {df[name].fillna("MISSING").value_counts().head(10).to_dict()}')
        lines.append('')
        lines.append('| level | y0_rate | y1_rate |')
        lines.append('| --- | ---: | ---: |')
        for _, row in rates.head(12).iterrows():
            lines.append(f"| {row['level']} | {row['y0_rate']:.4f} | {row['y1_rate']:.4f} |")
        lines.append('')

    lines.append('## Static Causal Recommendation')
    lines.append('')
    lines.append('1. Keep the current noleak feature set unchanged for baseline fairness.')
    lines.append('2. Replace synthetic `alpha_target` with `cigsmok` only for the next controlled audit branch.')
    lines.append('3. Treat `race`, `gender`, and `age` as baseline confounders/background covariates, not treatments.')
    lines.append('4. Do **not** promote `scr_res0`, `scr_iso0`, or `can_scr` into treatment, because that would reopen leakage or mediator conditioning.')
    lines.append('5. If a future branch restores a genuine randomized arm column such as `rndgroup`, re-run this audit before switching treatment definitions.')
    lines.append('')

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Wrote audit report to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()

"""
B1-1: 5-Table Integration Pipeline for Landmark-Based Modeling
构建从原始 5 表到统一主建模表的完整流水线
"""
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

# Leakage blacklist - 严格执行
BLACKLIST_EXACT = [
    'cancyr', 'candx_days', 'can_scr', 'canc_rpt_link',
    'clinical_stag', 'path_stag', 'histology', 'grade',
    'lesionsize', 'vital_status', 'fup_days'
]

BLACKLIST_PATTERNS = [
    r'^canc_.*',   # Cancer-related
    r'^de_.*',     # Death-related  
    r'^loc.*',     # Location (diagnosis-time)
]

def is_leakage(col_name):
    """检查字段是否为 leakage"""
    if col_name in BLACKLIST_EXACT:
        return True
    for pattern in BLACKLIST_PATTERNS:
        if re.match(pattern, col_name):
            return True
    return False

class LandmarkTableBuilder:
    def __init__(self, data_dir, output_dir, debug_mode=False, debug_n_persons=1000):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_mode = debug_mode
        self.debug_n_persons = debug_n_persons
        self.sampled_pids = None
        
    def load_raw_tables(self):
        """加载 5 张原始表"""
        base = os.path.join(self.data_dir, 'nlst.780.idc.delivery.052821')
        
        print("Loading raw tables...")
        self.prsn = pd.read_csv(f'{base}/nlst_780_prsn_idc_20210527.csv')
        
        if self.debug_mode:
            self.sampled_pids = self.prsn['pid'].sample(n=min(self.debug_n_persons, len(self.prsn)), random_state=42)
            self.prsn = self.prsn[self.prsn['pid'].isin(self.sampled_pids)]
            print(f"  Debug mode: sampled {len(self.sampled_pids)} persons")
        
        self.screen = pd.read_csv(f'{base}/nlst_780_screen_idc_20210527.csv')
        self.ctab = pd.read_csv(f'{base}/nlst_780_ctab_idc_20210527.csv')
        self.ctabc = pd.read_csv(f'{base}/nlst_780_ctabc_idc_20210527.csv')
        self.canc = pd.read_csv(f'{base}/nlst_780_canc_idc_20210527.csv')
        
        if self.debug_mode:
            self.screen = self.screen[self.screen['pid'].isin(self.sampled_pids)]
            self.ctab = self.ctab[self.ctab['pid'].isin(self.sampled_pids)]
            self.ctabc = self.ctabc[self.ctabc['pid'].isin(self.sampled_pids)]
            self.canc = self.canc[self.canc['pid'].isin(self.sampled_pids)]
        
        print(f"  prsn: {len(self.prsn)} rows, {len(self.prsn.columns)} cols")
        print(f"  screen: {len(self.screen)} rows, {len(self.screen.columns)} cols")
        print(f"  ctab: {len(self.ctab)} rows, {len(self.ctab.columns)} cols")
        print(f"  ctabc: {len(self.ctabc)} rows, {len(self.ctabc.columns)} cols")
        print(f"  canc: {len(self.canc)} rows, {len(self.canc.columns)} cols")
        
    def build_person_baseline_table(self):
        """中间表 1: person_baseline_table"""
        print("\n[1/5] Building person_baseline_table...")
        
        # 安全基线特征 (只保留非 leakage 字段)
        safe_cols = ['pid', 'age', 'gender', 'race', 'ethnic', 'bmi',
                     'cigsmok', 'smokeage', 'smokeyr', 'cigsperday', 'smokeday', 'smokequit',
                     'copd', 'emphysema', 'chronic_bronchitis', 'fhx_lung_cancer', 'prior_cancer']
        
        available_cols = [c for c in safe_cols if c in self.prsn.columns]
        baseline = self.prsn[available_cols].copy()
        
        # 剔除 leakage 字段检查
        for col in baseline.columns:
            if is_leakage(col):
                raise ValueError(f"LEAKAGE DETECTED in baseline: {col}")
        
        baseline.to_pickle(self.output_dir / 'person_baseline_table.pkl')
        print(f"  Saved: {len(baseline)} persons, {len(baseline.columns)} features")
        return baseline
    
    def build_person_year_screening_summary(self):
        """中间表 2: person_year_screening_summary (pid × study_yr)"""
        print("\n[2/5] Building person_year_screening_summary...")
        
        screen_agg = self.screen.groupby(['pid', 'study_yr']).agg({
            'ctdxqual': 'first',
            'techpara_kvp': 'mean',
            'techpara_ma': 'mean',
            'techpara_fov': 'mean'
        }).reset_index()
        
        screen_agg.to_pickle(self.output_dir / 'person_year_screening_summary.pkl')
        print(f"  Saved: {len(screen_agg)} person-year records")
        return screen_agg
    
    def build_person_year_abnormality_summary(self):
        """中间表 3: person_year_abnormality_summary (pid × study_yr)"""
        print("\n[3/5] Building person_year_abnormality_summary...")
        
        abn_agg = self.ctab.groupby(['pid', 'study_yr']).agg(
            abnormality_count=('sct_ab_num', 'count'),
            max_long_dia=('sct_long_dia', 'max'),
            max_perp_dia=('sct_perp_dia', 'max'),
            has_spiculated=('sct_margins', lambda x: (x == 4).any() if len(x) > 0 else False)
        ).reset_index()
        
        abn_agg.to_pickle(self.output_dir / 'person_year_abnormality_summary.pkl')
        print(f"  Saved: {len(abn_agg)} person-year records")
        return abn_agg
    
    def build_person_year_change_summary(self):
        """中间表 4: person_year_change_summary (pid × study_yr)"""
        print("\n[4/5] Building person_year_change_summary...")
        
        change_agg = self.ctabc.groupby(['pid', 'study_yr']).agg(
            has_growth=('sct_ab_gwth', lambda x: (x == 2).any() if len(x) > 0 else False),
            has_attn_change=('sct_ab_attn', lambda x: (x == 2).any() if len(x) > 0 else False),
            change_count=('sct_ab_num', 'count')
        ).reset_index()
        
        change_agg.to_pickle(self.output_dir / 'person_year_change_summary.pkl')
        print(f"  Saved: {len(change_agg)} person-year records")
        return change_agg
    
    def build_event_label_table(self):
        """中间表 5: event_label_table"""
        print("\n[5/5] Building event_label_table...")
        
        all_pids = self.prsn[['pid']].copy()
        all_pids['cancyr'] = self.prsn['cancyr'] if 'cancyr' in self.prsn.columns else np.nan
        
        if len(self.canc) > 0 and 'pid' in self.canc.columns:
            canc_events = self.canc.groupby('pid')['study_yr'].min().reset_index()
            canc_events.columns = ['pid', 'cancyr_from_canc']
            all_pids = all_pids.merge(canc_events, on='pid', how='left')
            all_pids['cancyr'] = all_pids['cancyr_from_canc'].combine_first(all_pids['cancyr'])
            all_pids.drop(columns=['cancyr_from_canc'], inplace=True)
        
        event_table = all_pids
        event_table.to_pickle(self.output_dir / 'event_label_table.pkl')
        print(f"  Saved: {len(event_table)} persons")
        print(f"    Cancer cases: {event_table['cancyr'].notna().sum()}")
        print(f"    No cancer: {event_table['cancyr'].isna().sum()}")
        return event_table
    
    def build_unified_landmark_table(self, baseline, screen_agg, abn_agg, change_agg, event_table):
        """构建统一主建模表: unified_person_landmark_table"""
        print("\n[FINAL] Building unified_person_landmark_table...")
        
        samples = []
        excluded_count = 0
        
        for _, person in baseline.iterrows():
            pid = person['pid']
            event_row = event_table[event_table['pid'] == pid]
            cancyr = event_row['cancyr'].values[0] if len(event_row) > 0 else np.nan
            
            for landmark in [0, 1, 2]:
                if pd.notna(cancyr) and cancyr <= landmark:
                    excluded_count += 1
                    continue
                
                y_2year = 1 if (pd.notna(cancyr) and cancyr > landmark and cancyr <= landmark + 2) else 0
                
                future_event_years = np.zeros(7, dtype=np.float32)
                if pd.notna(cancyr) and cancyr > landmark:
                    event_offset = int(cancyr - landmark - 1)
                    if 0 <= event_offset < 7:
                        future_event_years[event_offset] = 1.0
                
                screen_hist = screen_agg[(screen_agg['pid'] == pid) & (screen_agg['study_yr'] <= landmark)]
                abn_hist = abn_agg[(abn_agg['pid'] == pid) & (abn_agg['study_yr'] <= landmark)]
                change_hist = change_agg[(change_agg['pid'] == pid) & (change_agg['study_yr'] <= landmark)]
                
                sample = {'pid': pid, 'landmark': landmark, 'y_2year': y_2year, 'cancyr': cancyr}
                
                for col in baseline.columns:
                    if col != 'pid':
                        sample[f'baseline_{col}'] = person[col]
                
                for t in range(landmark + 1):
                    screen_t = screen_hist[screen_hist['study_yr'] == t]
                    if len(screen_t) > 0:
                        sample[f'screen_t{t}_ctdxqual'] = screen_t['ctdxqual'].values[0]
                    
                    abn_t = abn_hist[abn_hist['study_yr'] == t]
                    if len(abn_t) > 0:
                        sample[f'abn_t{t}_count'] = abn_t['abnormality_count'].values[0]
                        sample[f'abn_t{t}_max_dia'] = abn_t['max_long_dia'].values[0]
                    
                    change_t = change_hist[change_hist['study_yr'] == t]
                    if len(change_t) > 0:
                        sample[f'change_t{t}_has_growth'] = change_t['has_growth'].values[0]
                
                sample['trajectory_target'] = future_event_years
                samples.append(sample)
        
        unified = pd.DataFrame(samples)
        unified.to_pickle(self.output_dir / 'unified_person_landmark_table.pkl')
        print(f"  Saved: {len(unified)} samples from {unified['pid'].nunique()} persons")
        print(f"  Excluded: {excluded_count} samples (cancyr <= landmark)")
        print(f"  Positive rate: {unified['y_2year'].mean():.4f}")
        return unified
    
    def generate_statistics_report(self, unified):
        """生成数据统计报告"""
        print("\n" + "="*60)
        print("DATA STATISTICS REPORT")
        print("="*60)
        
        total_persons = unified['pid'].nunique()
        total_samples = len(unified)
        
        print(f"\n总患者数: {total_persons}")
        print(f"总样本数: {total_samples}")
        print(f"平均每人样本数: {total_samples/total_persons:.2f}")
        
        print("\n各 landmark 样本分布:")
        for lm in [0, 1, 2]:
            lm_samples = unified[unified['landmark'] == lm]
            pos_rate = lm_samples['y_2year'].mean()
            print(f"  T{lm}: {len(lm_samples)} 样本, 阳性率 {pos_rate:.4f}")
        
        print(f"\n排除样本数 (cancyr <= t): {total_persons * 3 - total_samples}")
        
        stats = {
            'total_persons': total_persons,
            'total_samples': total_samples,
            'landmark_distribution': {
                f'T{lm}': {
                    'count': len(unified[unified['landmark'] == lm]),
                    'positive_rate': float(unified[unified['landmark'] == lm]['y_2year'].mean())
                } for lm in [0, 1, 2]
            },
            'feature_dimensions': len([c for c in unified.columns if c not in ['pid', 'landmark', 'y_2year', 'cancyr']])
        }
        
        import json
        with open(self.output_dir / 'statistics_report.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def run_full_pipeline(self):
        """执行完整的 5 表整合流水线"""
        self.load_raw_tables()
        
        baseline = self.build_person_baseline_table()
        screen_agg = self.build_person_year_screening_summary()
        abn_agg = self.build_person_year_abnormality_summary()
        change_agg = self.build_person_year_change_summary()
        event_table = self.build_event_label_table()
        
        unified = self.build_unified_landmark_table(baseline, screen_agg, abn_agg, change_agg, event_table)
        
        stats = self.generate_statistics_report(unified)
        
        print("\n" + "="*60)
        print("B1-1 PIPELINE COMPLETE")
        print("="*60)
        print(f"输出目录: {self.output_dir}")
        
        return unified, stats

if __name__ == '__main__':
    builder = LandmarkTableBuilder(
        data_dir='data',
        output_dir='data/landmark_tables',
        debug_mode=True
    )
    unified, stats = builder.run_full_pipeline()

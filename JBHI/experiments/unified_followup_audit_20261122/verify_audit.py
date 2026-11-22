"""Re-read model checkpoints and public tables and validate the material used by the report."""
import itertools
import re
from common import *
from frozen_inputs import verify
import pandas as pd

def table(path):
    return pd.read_csv(path,float_precision='round_trip')


def main():
    verify()
    archive=table(FIX/'statistics/metrics.csv').query("model=='R'").set_index(['fold','seed'])
    measure=table(FIX/'statistics/measurement.csv').query("model=='R'").set_index(['fold','seed'])
    public=table(STATS/'param_sensitivity_metrics.csv').set_index(['fold','seed','arm'])
    tau=table(STATS/'tau_sweep.csv')
    pairs=obj(STATS/'param_sensitivity_paired.json')
    complex_df=table(STATS/'complexity_benchmark.csv')
    assert len(public)==63 and len(public.index.unique())==63 and public.notna().all().all()
    assert (public.notice==NOTICE).all() and (tau.notice==NOTICE).all() and (complex_df.notice==NOTICE).all()
    assert pairs['notice']==NOTICE and pairs['draws']==2000 and pairs['seed']==20261122
    assert set(r['arm'] for r in pairs['contrasts'])==set(ARMS)-{'R'} and len(pairs['contrasts'])==6
    assert all(c['lo']<=c['delta']<=c['hi'] for c in pairs['contrasts'])
    assert set(tau.tau)==set(TAUS) and len(tau.query("kind=='cell'"))==54 and len(tau.query("kind=='mean'"))==6
    for t in TAUS:
        c=tau[(tau.tau==t)&(tau.kind=='cell')]
        assert len(c[['fold','seed']].drop_duplicates())==9
        avg=tau[(tau.tau==t)&(tau.kind=='mean')].iloc[0]
        for col in measure.columns.drop('model'):
            assert abs(c[col].mean()-avg[col])<=1e-12
    max_tau=max_ref=0.
    evaluation_count=0
    for f in range(3):
        cal,ca=engine.read_role(f,1)
        ev,idx=engine.read_role(f,2)
        for s,name in itertools.product(SEEDS,ARMS):
            row=public.loc[(f,s,name)]
            assert (row.aux_weight,row.gamma)==ARMS[name]
            net=load_net(f,s,name)
            path=source(f,s) if name=='R' else PRIVATE/'fits'/group(f,s)/name
            threshold=cutoff(cal['y'],predict(net,cal,torch.arange(len(ca))))
            with np.load(path/'predictions.npz') as z:
                np.testing.assert_array_equal(z['rows'],idx)
                p=predict(net,ev,torch.arange(len(idx)))
                np.testing.assert_array_equal(p,z['p'])
            met=endpoints(ev['y'].numpy().astype('int64'),p,threshold)
            aux,_=engine.measurement(net,ev)
            for k,v in {**met,**aux}.items():
                assert abs(row[k]-v)<=1e-12,(f,s,name,k)
            if name=='R':
                for col,val in met.items():
                    max_ref=max(max_ref,abs(row[col]-archive.loc[(f,s),col]),abs(val-row[col]))
                cell=tau[(tau.kind=='cell')&(tau.tau==.5)&(tau.fold==str(f))&(tau.seed==str(s))].iloc[0]
                for col,val in aux.items():
                    max_tau=max(max_tau,abs(cell[col]-measure.loc[(f,s),col]),abs(cell[col]-val))
                assert row.selected_epoch==obj(path/'fit.json')['selected']
            else:
                record=obj(path/'evaluation.json')
                assert record['threshold']==threshold
                lines=(LOGS/f'{name}-dev{f}-{s}.log').read_text().splitlines()
                aps=[float(re.search(r' cal_ap=([.0-9]+)',line)[1]) for line in lines]
                assert len(aps)==record['epoch'] and row.selected_epoch==record['selected_epoch']
                assert aps[int(row.selected_epoch)-1]==max(aps)
                assert (len(aps)-row.selected_epoch==CFG['patience'] or len(aps)==CFG['finetune_epochs'])
            evaluation_count+=1
    assert max_tau<=1e-12 and max_ref==0
    assert len(complex_df)==11 and set(complex_df.model)==set((*'ABCDRN',*DEEP))
    for r in complex_df.itertuples():
        actual=obj(PRIVATE/f'timing-{r.model}.json')['times_ms']
        assert (r.batch_size,r.threads,r.warmups,r.repeats,len(actual))==(1000,1,5,20,20)
        np.testing.assert_array_equal(np.quantile(actual,[.25,.5,.75]),[r.q25_ms,r.median_ms,r.q75_ms])
        assert 0<r.deployment_parameters<=r.registered_parameters
        assert r.training_graph_parameters==r.training_network_parameters+r.loss_parameters
    cases=obj(PRIVATE/'case_display.json')
    assert set(cases)=={'1','2','3'} and obj(STATS/'case_selection.json')['distinct_cases']==3
    assert all(re.fullmatch(r'\d{2}–\d{2}',r['age']) and set(r['models'])==set('BCRN') for r in cases.values())
    assert cases['3']['next_mm'] is None and cases['3']['outcome']==1
    private_doc=PRIVATE/'case_studies_local_only.md'
    assert private_doc.stat().st_mode & 0o077==0 and private_doc.read_text().count('## Case ')==3
    save_json(STATS/'verification.json',dict(notice=NOTICE,tau_max_abs=max_tau,r_metrics_max_abs=max_ref,
              frozen_r_weights_replayed=9,all_evaluation_checkpoints_replayed=evaluation_count,new_fits=54,
              tau_cells=54,tau_means=6,paired_contrasts=6,paired_draws=2000,models_benchmarked=11,
              subjects=10594,landmarks=31587,sealed_records_read=0,patient_computation='local only',
              distinct_local_cases=3,archive_hashes_verified=True))
    print('Verified all 63 checkpoints, public tables, local case structure, and frozen input digests',flush=True)


if __name__=='__main__':
    main()

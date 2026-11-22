"""Deterministic case selection. Export only counts to the public statistics directory."""
import argparse
from common import *
from model import Unified
from scipy.stats import rankdata

SITES=('右上','右中','右下','左上','左下','其他')
ARMS4='BCRN'


def violation(v):
    return (np.abs(v[:,1:7].max(1)-v[:,0])>1e-5), (v[:,-1]>v[:,0]+1e-5)


def inspect_fold(f,s):
    d,rows=engine.read_role(f,2)
    mask=d['mask'].numpy()
    size,ranks={},{}
    for name in ARMS4:
        path=source(f,s,name) if name in 'RN' else OLD/'runs'/group(f,s)/name
        with np.load(path/'predictions.npz') as p:
            np.testing.assert_array_equal(rows,p['rows'])
            np.testing.assert_array_equal(rows[mask],p['aux_rows'])
            ranks[name]=100*(rankdata(p['p'],method='average')-.5)/len(rows)
            size[name]=np.zeros((len(rows),8))
            size[name][mask]=p['sizes_mm'] if name in 'RN' else np.expm1(p['sizes'])
    return d,rows,mask,size,ranks


def choose():
    # Selection frame is locked to the preferred dev0/42. Other folds are inspected
    # only when even the stated fallback for a case has no candidate.
    chosen={}
    summary={}
    aggregates={}
    used=set()
    with np.load(OLD/'private/registry.npz') as bank:
        who,landmark,ct=(bank[k] for k in ('who','lm','ct'))
    person_ct=np.zeros((int(who.max())+1,3),dtype=bool)
    person_ct[who,landmark]=ct
    for f,s in [(0,42),(1,42),(2,42)]:
        if len(chosen)==3:
            break
        d,rows,mask,size,ranks=inspect_fold(f,s)
        truth=d['aux'].expm1().numpy()
        y=d['y'].numpy().astype(bool)
        lm=d['lm'].numpy()
        delta={n:abs(v[:,0]-truth[:,0]) for n,v in size.items()}
        bv,_=violation(size['B'])
        rv,rg=violation(size['R'])
        dom=d['dom'].numpy()
        extra=truth[:,1:]>0
        extra[np.flatnonzero(dom),dom[dom>0]-1]=False
        missed=(extra & (size['R'][:,1:]==0)).any(1)
        a=mask & (lm<2) & bv & ~rv & ~rg & (delta['B']>delta['R'])
        c2=mask & (lm<2) & missed & (delta['R']<delta['N'])
        gap=ranks['N']-ranks['R']
        c3=(lm==2) & y & ~mask
        if (f,s)==(0,42):
            low=load_net(f,s,scan=True)
            low.tau=.10
            _,dec=engine.measurement(low,d)
            missed_sites=extra[mask] & (size['R'][mask,1:]==0)
            aggregates=dict(eligible=int(mask.sum()),b_max_violations=int((bv&mask).sum()),
                            r_max_violations=int((rv&mask).sum()),r_growth_violations=int((rg&mask).sum()),
                            extra_positive_sites=int(extra[mask].sum()),missed_extra_sites=int(missed_sites.sum()),
                            recovered_extra_sites=int((missed_sites & (dec['sizes_mm'][:,1:]>0)).sum()),
                            positive_t2=int(c3.sum()),both_outside_top5=int((c3&(ranks['R']<95)&(ranks['N']<95)).sum()))
        alternatives=[
            ([(a&y&(ranks['R']>=90),'阳性且前10%'),(a&y&(ranks['R']>=80),'阳性且前20%'),
              (a&(dom>0)&(ranks['R']>=90),'真实正径且前10%'),(a&(dom>0)&(ranks['R']>=80),'真实正径且前20%')],delta['B']-delta['R']),
            ([(c2&y&(gap>0),'阳性且R分位低于N'),(c2,'最大N减R分位差')],gap),
            ([(c3&(ranks['R']<95)&(ranks['N']<95),'均未进前5%'),(c3,'阳性T2的最大绝对分位差')],abs(gap))]
        for k,(tiers,score) in enumerate(alternatives,1):
            if k in chosen:
                continue
            for cand,label in tiers:
                key=f'Case {k} / dev{f} / seed{s} / {label}'
                cand=cand & ~np.isin(who[rows],list(used))
                ix=np.flatnonzero(cand)
                summary[key]=len(ix)
                if len(ix):
                    i=ix[np.argmax(score[ix])]
                    used.add(who[rows[i]])
                    chosen[k]=derive(k,f,s,i,d,mask,truth,size,ranks,label,extra,person_ct[who[rows[i]]])
                    break
    assert len(chosen)==len(used)==3
    save_json(PRIVATE/'case_display.json',chosen)
    save_json(STATS/'case_selection.json',dict(notice=NOTICE,counts=summary,distinct_cases=3,dev0_seed42=aggregates))


@torch.no_grad()
def derive(k,f,s,i,d,mask,truth,size,ranks,label,extra,ct):
    lm=int(d['lm'][i])
    age=int(d['x'][i,0,0])//5*5
    history=[]
    for t in range(3):
        if t>lm:
            assert (d['x'][i,t]==0).all()
            history.append('未来轮次已掩码')
        else:
            x=d['x'][i,t]
            history.append(f'已观察，{("无CT记录","有CT记录")[int(ct[t])]}。异常计数 {int(x[7])}，历史最大长径 {float(x[8]):.1f} mm，生长标记 {int(x[11])}')
    models={}
    for n in ARMS4:
        if mask[i]:
            v=size[n][i]
        else:
            net=load_net(f,s,n) if n in 'RN' else Unified(n)
            if n in 'BC':
                net.load_state_dict(engine.state(OLD/'runs'/group(f,s)/n/'best.pt'))
            net.eval()
            h=net.encode(*inputs(d,torch.tensor([i])))
            v=net.report(h)[1].expm1()[0].numpy()
        maxv,grow=violation(v[None,:])
        models[n]=dict(rank=float(ranks[n][i]),mm=v.tolist(),viol=[bool(maxv[0]),bool(grow[0])])
    out=dict(fold=f,seed=s,selection=label,age=f'{age}–{age+4}',history=history,
             smoking=('既往吸烟','当前吸烟')[int(d['a'][i])],outcome=int(d['y'][i]),
             next_mm=truth[i].tolist() if mask[i] else None,models=models)
    if k==2:
        net=load_net(f,s,scan=True)
        # Match the exact eligible batch used in the tau sweep.
        active=torch.where(d['mask'])[0]
        batch=next(b for b in active.split(512) if (b==i).any())
        loc=int(torch.where(batch==i)[0].item())
        net.tau=.10
        v=net.report(net.encode(*inputs(d,batch)))[1].expm1()[loc].numpy()
        out.update(tau=.10,tau_mm=v.tolist(),missed=[(*SITES,'生长')[j] for j in
                   np.flatnonzero(extra[i] & (size['R'][i,1:]==0))])
    return out


def mm(v):
    return ' / '.join(f'{x:.2f}' for x in v)


def render():
    records=obj(PRIVATE/'case_display.json')
    lines=['# 三个脱敏病例 · 受限本地详表','',NOTICE,'',
           '尺寸均为毫米，顺序为全肺、右上、右中、右下、左上、左下、其他、生长标记最大径。风险百分位相对于同折同模型的全部预测时点，越高越靠前。同分取中秩。两年阳性指预测时点后的研究年终点。']
    for k,case in records.items():
        lines.extend(['',f'## Case {k} · '+['机制正向例','障碍与终点分化例','纯历史部署边界例'][int(k)-1],'',
                      f'开发折 {case["fold"]}，种子 {case["seed"]}。筛选层级为“{case["selection"]}”。','',NOTICE+'。','',
                      '| 时间点或条件 | 可见信息 |','|---|---|',
                      *[f'| T{i} | {h} |' for i,h in enumerate(case['history'])],
                      f'| 基线条件 | 年龄 {case["age"]} 岁，{case["smoking"]} |',
                      f'| 两年真实结局 | {("阴性","阳性")[case["outcome"]]} |',
                      '| 下一轮 CT（仅作事后对照） | '+(mm(case['next_mm']) if case['next_mm'] is not None else '无下一轮 CT 监督')+' |',
                      '', '| 模型 | 风险百分位 | 全肺及各部位预测尺寸（mm） | 最大径违例 | 生长径违例 |',
                      '|---|---:|---|---|---|'])
        for n,r in case['models'].items():
            lines.append(f'| {n} | {r["rank"]:.2f} | {mm(r["mm"])} | {("无","有")[r["viol"][0]]} | {("无","有")[r["viol"][1]]} |')
        if k in ('1','2'):
            primary='B' if k=='1' else 'N'
            truth=case['next_mm'][0]
            e1=abs(case['models'][primary]['mm'][0]-truth)
            e2=abs(case['models']['R']['mm'][0]-truth)
            lines.extend(['',f'{primary} 与 R 的全肺误差分别为 {e1:.2f}、{e2:.2f} mm，R 的误差减少 {e1-e2:.2f} mm。'])
        if k=='2':
            before=case['models']['R']['mm'][1:]
            after=case['tau_mm'][1:]
            changed=[f'{n} {lo:.2f}→{hi:.2f} mm' for n,lo,hi in zip((*SITES,'生长'),before,after) if abs(lo-hi)>.005]
            recovered=[n for n,val in zip((*SITES,'生长'),after) if n in case['missed'] and val>0]
            lines.extend(['','降阈值后变化的项目为'+('、'.join(changed) if changed else '无')+'。原先漏报的真实额外项中，'+('、'.join(recovered)+'恢复非零输出。' if recovered else '本例无一恢复非零输出。')])
            lines.extend(['','默认门漏报的真实额外项为'+'、'.join(case['missed'])+'。τ=0.10 时全套预测尺寸为 '+mm(case['tau_mm'])+' mm。'])
        if k=='3':
            lines.extend(['','T2 已达最后一个协议轮次。表中风险采用已保存的外折分数，尺寸为冻结诊断头的历史前缀回放，缺少实测对照，无法计算尺寸误差。'])
    (PRIVATE/'case_studies_local_only.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['select','render-local'])
    if p.parse_args().action=='select':
        choose()
    else:
        render()

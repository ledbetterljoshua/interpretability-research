"""Verify and summarize saved offset experiments without loading a model."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[1]

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def finite(x):
    if isinstance(x,float): assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values(): finite(v)
    elif isinstance(x,list):
        for v in x: finite(v)
def aggregate(ms):
    return dict(n=len(ms),correct=sum(m['rank']==1 for m in ms),p=st.mean(m['p'] for m in ms))
def main():
    mdir=ROOT/'data/offset_mechanism'
    rdir=ROOT/'data/conditional_routes'
    m,r=read(mdir/'run.json'),read(rdir/'run.json')
    assert m['status']==r['status']=='complete'
    for p,key in [('experiments/offset_mechanism.py','source_sha256'),
                  ('experiments/capital_generalization.py','helper_sha256'),
                  ('notes/2026-09-05-offset-mechanism-plan.md','plan_sha256'),
                  ('data/offset_mechanism/preflight.json','preflight_sha256'),
                  ('data/offset_mechanism/offsets.npz','offsets_sha256')]:
        assert sha(ROOT/p)==m[key],p
    for p,h in r['hashes'].items(): assert sha(ROOT/p)==h,p
    pf=read(mdir/'preflight.json')
    fit={c for c,_ in pf['fitting']}
    rep={c for c,_ in pf['replication']}
    val={c for c,_ in pf['fresh']}
    assert len(fit)==len(rep)==len(val)==10
    assert not (fit&rep or fit&val or rep&val)
    assert len(m['cases'])==len(r['cases'])==20
    mechanism=[read(mdir/(n+'.json')) for n in m['cases']]
    routes=[read(rdir/(n+'.json')) for n in r['cases']]
    for c in mechanism+routes:
        finite(c)
        assert max(c['controls'].values())<1e-4
    for c in routes:
        assert [x['mask'] for x in c['records']]==list(range(256))
        prior=next(d for d in mechanism if d['country']==c['country'])
        for mask,old in [(255,prior['conditions']['capital']['full']),
                         (0,prior['conditions']['capital']['direct']),
                         (85,prior['mediation']['all_mlp']),
                         (170,prior['mediation']['all_attention'])]:
            new=c['records'][mask]['actual']
            assert new['rank']==old['rank'] and abs(new['p']-old['p'])<1e-5
        for row in c['records']:
            assert row['kl']>=-1e-5
            for kind in ('actual','predicted'):
                q=row[kind]
                assert 0<=q['p']<=1 and 1<=q['rank']<=50257
    summary={'mechanism':{},'routes':{},'scope':{},'resources':dict(
        seconds=m['elapsed_seconds']+r['elapsed_seconds'],peak_rss_mib=max(m['peak_rss_mib'],r['peak_rss_mib']))}
    # Selection reads development records only; fixed rule in the prospective plan.
    dev=[c for c in routes if c['split']=='replication']
    candidates=[]
    for mask in range(256):
        ms=[c['records'][mask]['actual'] for c in dev]
        if sum(q['rank']==1 for q in ms)>=9:
            candidates.append((mask.bit_count(),-st.mean(q['p'] for q in ms),mask))
    selected=min(candidates)[2] if candidates else None
    summary['selection']=dict(mask=selected,active=dev[0]['records'][selected]['active'] if selected is not None else [],
                              rule='Fewest active with >=9/10 development correct; ties by mean probability, then mask')
    for split in ('replication','fresh'):
        cs=[c for c in mechanism if c['split']==split]
        conditions={mode:{kind:aggregate([c['conditions'][mode][kind] for c in cs])
                    for kind in ('full','direct','output_bias','natural')} for mode in cs[0]['conditions']}
        summary['mechanism'][split]=dict(baseline=aggregate([c['baseline'] for c in cs]),conditions=conditions,
            mediation={k:aggregate([c['mediation'][k] for c in cs]) for k in cs[0]['mediation']},
            attribution={k:st.mean(c['attribution_terms'][k] for c in cs) for k in cs[0]['attribution_terms']})
        cs=[c for c in routes if c['split']==split]
        per_country=[]
        for c in cs:
            rs=c['records']
            per_country.append(dict(country=c['country'],correctness_disagreement=st.mean((x['actual']['rank']==1)!=(x['predicted']['rank']==1) for x in rs),
                top_disagreement=st.mean(x['actual']['top_id']!=x['predicted']['top_id'] for x in rs),
                margin_mae=st.mean(abs(x['actual']['margin']-x['predicted']['margin']) for x in rs),
                mean_kl=st.mean(x['kl'] for x in rs)))
        summary['routes'][split]=dict(per_country=per_country,
            means={k:st.mean(c[k] for c in per_country) for k in per_country[0] if k!='country'},
            masks={str(mask):dict(actual=aggregate([c['records'][mask]['actual'] for c in cs]),
                                 predicted=aggregate([c['records'][mask]['predicted'] for c in cs]),
                                 active=cs[0]['records'][mask]['active']) for mask in range(256)})
    assert len(r['scope_cases'])==30
    for form in ('possessive','qa','language'):
        cs=[c for c in r['scope_cases'] if c['form']==form and c['eligible']]
        summary['scope'][form]=dict(eligible=len(cs),excluded=[c for c in r['scope_cases'] if c['form']==form and not c['eligible']],
            modes={mode:dict(**aggregate([c['results'][mode] for c in cs]),
                capitals_top=sum(c['results'][mode]['capital_top'] for c in cs)) for mode in ('baseline','capital','neutral','other_attribute')})
    response=read(rdir/'response.json')
    assert response['status']=='complete' and len(response['cases'])==20
    assert sha(ROOT/'experiments/conditional_response.py')==response['source_sha256']
    assert sha(ROOT/'notes/2026-09-05-response-plan.md')==response['plan_sha256']
    assert sha(mdir/'offsets.npz')==response['input_sha256']
    finite(response)
    for c in response['cases']:
        assert max(c['controls'].values())<1e-4
        old=next(q for q in routes if q['country']==c['country'])['records'][170]
        for new,prior in [(c['adapted'],old['actual']),(c['freeze_to_full']['all'],old['predicted'])]:
            assert new['rank']==prior['rank'] and abs(new['p']-prior['p'])<1e-5
    summary['response']={split:dict(adapted=aggregate([c['adapted'] for c in response['cases'] if c['split']==split]),
        freeze_to_full={k:aggregate([c['freeze_to_full'][k] for c in response['cases'] if c['split']==split]) for k in ('all','mlp8','mlp9','mlp10','mlp11')}) for split in ('replication','fresh')}
    summary['resources']['seconds']+=response['elapsed_seconds']
    summary['resources']['peak_rss_mib']=max(summary['resources']['peak_rss_mib'],response['peak_rss_mib'])
    saved_selection=read(rdir/'development_selection.json')
    assert saved_selection['selection'][2]==selected
    assert saved_selection['validation_records_available_at_selection']==0
    finite(summary)
    (rdir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('Verified provenance, 30 disjoint countries, 5,120 planned subset interventions, 120 shadow controls, 80 cross-run intervention agreements, and all scope records.')
    for split in ('replication','fresh'):
        print(split, 'prediction errors:',summary['routes'][split]['means'])
        print('Selected route:',summary['selection'],summary['routes'][split]['masks'][str(selected)])
        print('Attention frozen:',summary['routes'][split]['masks']['170'])
    print('Scope:',json.dumps(summary['scope']))
    print('Resources:',summary['resources'])

if __name__=='__main__': main()

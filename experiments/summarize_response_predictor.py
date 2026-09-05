"""Validate saved response predictors and their forecasts; no model loading."""
import hashlib
import json
from pathlib import Path
import statistics as st
from summarize_offset_routes import finite

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/response_predictor'

def read(p): return json.loads(p.read_text())
def main():
    m=read(D/'run.json')
    assert m['status']=='complete'
    for p,h in m['hashes'].items(): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    assert len(m['original'])==20 and len(m['scope'])==30
    original=[read(D/(n+'.json')) for n in m['original']]
    scope=[read(D/(n+'.json')) for n in m['scope']]
    for c in original+scope:
        finite(c)
        assert max(c['controls'].values())<1e-4
        for r in c['records']:
            for p in r['predictions'].values():
                assert 0<=p['p']<=1 and 1<=p['rank']<=50257
    for c in original: assert [r['mask'] for r in c['records']]==list(range(256))
    for c in scope: assert [r['mask'] for r in c['records']]==m['scope_masks']
    def score(cs,exclude_endpoints=False):
        result={}
        for mode in m['modes']:
            country_stats=[]
            for c in cs:
                rs=[r for r in c['records'] if not exclude_endpoints or r['mask'] not in (0,255)]
                country_stats.append(dict(country=c['country'],n=len(rs),
                    correctness_disagreement=st.mean((r['actual']['rank']==1)!=(r['predictions'][mode]['rank']==1) for r in rs),
                    top_disagreement=st.mean(r['actual']['top_id']!=r['predictions'][mode]['top_id'] for r in rs),
                    probability_mae=st.mean(abs(r['actual']['p']-r['predictions'][mode]['p']) for r in rs),
                    margin_mae=st.mean(abs(r['actual']['margin']-r['predictions'][mode]['margin']) for r in rs)))
            result[mode]=dict(cases=len(cs),interventions=sum(c['n'] for c in country_stats),
                **{k:st.mean(c[k] for c in country_stats) for k in country_stats[0] if k not in ('country','n')},per_case=country_stats)
        return result
    result=dict(original={split:score([c for c in original if c['split']==split]) for split in ('replication','fresh')},
        scope={form:score([c for c in scope if c['form']==form],True) for form in ('possessive','qa','language')},
        scope_pooled=score(scope,True),resources=dict(seconds=m['elapsed_seconds'],peak_rss_mib=m['peak_rss_mib']))
    orig=result['original']['fresh']
    result['forecasts']=dict(last_mlp_reduces_old_error_by_5pp=orig['frozen']['correctness_disagreement']-orig['last_mlp']['correctness_disagreement']>=.05,
        all_mlps_reduce_old_error_by_10pp=orig['frozen']['correctness_disagreement']-orig['all_mlps']['correctness_disagreement']>=.10,
        both_reduce_new_scope_error=all(result['scope_pooled'][k]['correctness_disagreement']<result['scope_pooled']['frozen']['correctness_disagreement'] for k in ('last_mlp','all_mlps')))
    finite(result)
    (D/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for group,values in list(result['original'].items())+list(result['scope'].items())+[('scope_pooled',result['scope_pooled'])]:
        print(group,{k:{q:round(v[q],5) for q in ('correctness_disagreement','top_disagreement','probability_mae','margin_mae')} for k,v in values.items()})
    print('Forecasts:',result['forecasts']);print('Resources:',result['resources'])
    print('Verified hashes, 15,360 original predictions, 810 scope predictions against 270 measured outcomes, and all numerical controls. New-outcome scoring excludes 60 endpoint controls.')

if __name__=='__main__': main()

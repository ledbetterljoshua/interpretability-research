"""Fixed 18-contrast analysis; question uncertainty, never model-population claims."""
import numpy as np
import stratified_protocol as sp
from budget_statistics import paired,paired_intervals,holm,decision_comparison

SPLITS=('arc_test','openbook_test')
DIAGNOSTICS=('random_write_1215','random_write_1216','random_write_1217','final_only','context_only')
PROJECTED=('symmetric_write','symmetric_unit_write')
STRATUM={n:k for k,values in sp.STRATA.items() for n in values}


def methods(name,abstained):
    result=['ordinary','prompt_only','decoded','raw']
    if not abstained:
        result.extend(DIAGNOSTICS)
        if name=='reference-widened-base':result.extend(PROJECTED)
    if name in sp.CONSTRUCTED:result.append('own_code')
    return (*result,'sft')


def summarize(vectors,source_abstained,projection_nondegenerate):
    assert type(source_abstained) is bool
    assert projection_nondegenerate is None if source_abstained else type(projection_nondegenerate) is bool
    expected={(n,s,m) for n in sp.POPULATION for s in SPLITS for m in methods(n,source_abstained)}
    assert set(vectors)==expected,'Missing, duplicate/substituted or extra result cell'
    data={k:np.asarray(v) for k,v in vectors.items()}
    assert all(v.shape==(256,) and np.isin(v,[0,1]).all() for v in data.values()),'Expected 256 binary correctness values per cell'
    data={k:v.astype(np.int64) for k,v in data.items()}
    if source_abstained:
        assert all(np.array_equal(data[n,s,'raw'],data[n,s,'ordinary']) for n in sp.POPULATION for s in SPLITS)
    elif not projection_nondegenerate:
        assert all(np.array_equal(data['reference-widened-base',s,m],data['reference-widened-base',s,'ordinary']) for s in SPLITS for m in PROJECTED)
    table=[];primary=[];secondary=[];diagnostics=[];matched=[]
    for si,split in enumerate(SPLITS):
        interval_entries=[];deltas=[]
        def attach(row,delta,destination):
            destination.append(row);interval_entries.append(row);deltas.append(delta)
        for name in sp.POPULATION:
            ordinary=data[name,split,'ordinary'];raw=data[name,split,'raw']
            for method in methods(name,source_abstained):
                value=data[name,split,method];stats=paired(value,ordinary);stats.pop('exact_two_sided_p')
                attach(dict(model=name,split=split,stratum=STRATUM[name],method=method,**stats,
                    accuracy=float(value.mean()),ordinary_accuracy=float(ordinary.mean()),
                    flagged_by_20pp_gain=stats['left_correct']-stats['right_correct']>=52,
                    ordinary_headroom_correct=256-int(ordinary.sum()),ordinary_headroom_allows_flag=256-int(ordinary.sum())>=52,
                    reused_abstention=(source_abstained and method=='raw') or
                        (not source_abstained and not projection_nondegenerate and method in PROJECTED)),value-ordinary,table)
            for comparator in ('decoded','prompt_only','sft'):
                stats=paired(raw,data[name,split,comparator])
                if comparator!='decoded':stats.pop('exact_two_sided_p')
                attach(dict(model=name,split=split,stratum=STRATUM[name],comparison='raw_minus_'+comparator,**stats),
                    raw-data[name,split,comparator],primary if comparator=='decoded' else secondary)
            for method in methods(name,source_abstained):
                if method not in (*DIAGNOSTICS,*PROJECTED,'own_code'):continue
                stats=paired(raw,data[name,split,method]);stats.pop('exact_two_sided_p')
                attach(dict(model=name,split=split,stratum=STRATUM[name],comparison='raw_minus_'+method,**stats),
                    raw-data[name,split,method],diagnostics)
        for seed in (1091,1289):
            c=f'lower-gold-conditional-{seed}-v1';m=f'lower-gold-marginal-{seed}-v1'
            cg=data[c,split,'raw']-data[c,split,'ordinary'];mg=data[m,split,'raw']-data[m,split,'ordinary']
            attach(dict(split=split,seed=seed,stratum=STRATUM[c],conditional=c,marginal=m,
                conditional_gain=float(cg.mean()),marginal_gain=float(mg.mean()),gain_difference=float((cg-mg).mean()),
                fully_eligible_matched_pair=seed==1091),cg-mg,matched)
        intervals=paired_intervals(np.stack(deltas,axis=1),seed=1213+si,draws=10000)
        for row,interval in zip(interval_entries,intervals):row['descriptive_95pct_question_interval']=interval
    assert len(primary)==18
    adjusted=holm([r['exact_two_sided_p'] for r in primary])
    for row,p in zip(primary,adjusted):
        row['holm_adjusted_p_all_18']=p;row['holm_reject_at_05']=p<.05
    by_cell={(r['model'],r['split'],r['method']):r for r in table}
    ground_truth=[dict(model=n,split=s,conditional_supervision=n in sp.CONDITIONAL,
        own_code_correct=by_cell[n,s,'own_code']['left_correct'],ordinary_correct=by_cell[n,s,'ordinary']['left_correct'],
        own_code_gain=by_cell[n,s,'own_code']['gain'],own_code_recovery_flag=by_cell[n,s,'own_code']['flagged_by_20pp_gain'])
        for n in sp.CONSTRUCTED for s in SPLITS]
    keys=[(n,s) for s in SPLITS for n in sp.POPULATION]
    comparisons={}
    for comparator in ('prompt_only','decoded','sft'):
        cells=[]
        for n,s in keys:
            truth=n in sp.CONDITIONAL;r=by_cell[n,s,'raw']['flagged_by_20pp_gain'];b=by_cell[n,s,comparator]['flagged_by_20pp_gain']
            cells.append(dict(model=n,split=s,stratum=STRATUM[n],conditional_supervision=truth,raw_flag=r,comparator_flag=b,
                raw_classification_correct=r==truth,comparator_classification_correct=b==truth,
                added_conditional_detection=truth and r and not b,new_error=(b==truth and r!=truth)))
        def decision(chosen):
            stats=decision_comparison([r['raw_flag'] for r in chosen],[r['comparator_flag'] for r in chosen],
                [r['conditional_supervision'] for r in chosen])
            stats['cell_order']=[dict(model=r['model'],split=r['split']) for r in chosen]
            stats['added_conditional_cells']=[stats['cell_order'][i] for i in stats['added_true_positive_indices']]
            stats['new_error_cells']=[stats['cell_order'][i] for i in stats['new_error_indices']]
            return stats
        strata={}
        for stratum in sp.STRATA:
            subset=[r for r in cells if r['stratum']==stratum]
            strata[stratum]=dict(cells=subset,raw_errors=sum(not r['raw_classification_correct'] for r in subset),
                comparator_errors=sum(not r['comparator_classification_correct'] for r in subset),
                newly_introduced_errors=sum(r['new_error'] for r in subset),added_conditional_detections=sum(r['added_conditional_detection'] for r in subset))
        comparisons[comparator]=dict(cells=cells,by_task={s:decision([r for r in cells if r['split']==s]) for s in SPLITS},
            all_18=decision(cells),strata=strata)
    forecasts=[]
    def add_forecast(name,passed,**evidence):forecasts.append(dict(name=name,passed=bool(passed),**evidence))
    for name in sp.CONDITIONAL:
        for split in SPLITS:
            gain=by_cell[name,split,'raw']['gain']
            add_forecast('conditional_raw_gain_at_least_20pp',gain>=.2,model=name,split=split,gain=gain)
            gap=by_cell[name,split,'sft']['accuracy']-by_cell[name,split,'raw']['accuracy']
            add_forecast('sft_within_10pp_of_raw',gap>=-.1,model=name,split=split,sft_minus_raw=gap)
    for name in sp.STRATA['teacher_only']:
        for split in SPLITS:
            gain=by_cell[name,split,'raw']['gain']
            add_forecast('teacher_raw_gain_at_most_10pp',gain<=.1,model=name,split=split,gain=gain)
    marginal_cells=[dict(model=f'lower-gold-marginal-{seed}-v1',split=s,
        gain=by_cell[f'lower-gold-marginal-{seed}-v1',s,'raw']['gain']) for seed in (1091,1289) for s in SPLITS]
    add_forecast('at_least_one_marginal_gain_over_10pp',any(r['gain']>.1 for r in marginal_cells),cells=marginal_cells)
    add_forecast('no_added_decision_value_over_decoded',not comparisons['decoded']['all_18']['added_detection_without_new_errors'])
    for name in sp.REFERENCES:
        for split in SPLITS:
            gain=by_cell[name,split,'raw']['gain']
            add_forecast('provenance_reference_raw_gain_at_most_10pp',gain<=.1,model=name,split=split,gain=gain)
    if source_abstained:
        forecasts.append(dict(name='projected_write_nondegenerate',passed=None,reason='No selected source direction under abstention'))
    else:
        add_forecast('projected_write_nondegenerate',projection_nondegenerate)
        for split in SPLITS:
            for method in PROJECTED:
                gap=by_cell['reference-widened-base',split,'raw']['accuracy']-by_cell['reference-widened-base',split,method]['accuracy']
                add_forecast('raw_and_projected_within_10pp',abs(gap)<=.1,model='reference-widened-base',split=split,method=method,raw_minus_projected=gap)
    return dict(population=list(sp.POPULATION),strata={k:list(v) for k,v in sp.STRATA.items()},
        source_abstained=source_abstained,projection_nondegenerate=projection_nondegenerate,
        historical_construction_eligibility=sp.EXPECTED_ELIGIBILITY,historical_failed_forecasts=sp.EXPECTED_FAILED_FORECASTS,
        fully_eligible_replicated_imitation_population=False,table=table,primary_raw_minus_decoded=primary,
        secondary_raw_comparisons=secondary,diagnostic_comparisons=diagnostics,within_seed_gain_contrasts=matched,
        own_code_ground_truth_diagnostics=ground_truth,
        decision_comparisons=comparisons,forecasts=forecasts,failed_forecasts=[r for r in forecasts if r['passed'] is False],
        uncertainty=dict(question_bootstrap_draws=10000,seeds=dict(zip(SPLITS,(1213,1214))),holm_family_size=18,
            scope='Fixed models; shared question resampling within each task. Intervals are descriptive, not simultaneous or model-population uncertainty.'),
        interpretation='Operational research-imposed conditionality labels. Failed replication remains a separate stratum; references establish provenance, not proven ignorance or absence of natural concealment.')

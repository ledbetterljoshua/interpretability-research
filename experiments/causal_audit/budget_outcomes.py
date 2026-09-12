"""Fixed paired outcome analysis over correctness arrays; no model/data loading."""
import audit_population as ap
import numpy as np
from budget_statistics import paired,paired_intervals,holm,decision_comparison

POPULATION=ap.POPULATION
SPLITS=("arc_test","openbook_test")
PRIMARY_METHODS=("ordinary","prompt_only","decoded","raw")
DIAGNOSTICS=("random_write_1215","random_write_1216","random_write_1217","final_only","context_only")


def summarize(vectors,base_vectors,abstained=False):
    assert type(abstained) is bool
    methods=(*PRIMARY_METHODS,*(() if abstained else DIAGNOSTICS),"own_code","sft")
    expected={(name,split,method) for name in POPULATION for split in SPLITS for method in methods}
    assert set(vectors)==expected and set(base_vectors)==set(SPLITS)
    data={k:np.asarray(v) for k,v in vectors.items()}
    base={k:np.asarray(v) for k,v in base_vectors.items()}
    assert all(v.shape==(256,) and np.isin(v,[0,1]).all() for v in [*data.values(),*base.values()])
    data={k:v.astype(np.int64) for k,v in data.items()};base={k:v.astype(np.int64) for k,v in base.items()}
    roles=ap.ARMS.copy()
    table=[];primary=[];secondary=[];specificity=[];ablations=[]
    for split_index,split in enumerate(SPLITS):
        interval_vectors=[];interval_rows=[]
        def attach(row,delta,destination):
            destination.append(row);interval_rows.append(row);interval_vectors.append(delta)
        for name in POPULATION:
            ordinary=data[name,split,"ordinary"];raw=data[name,split,"raw"]
            if abstained:assert np.array_equal(raw,ordinary)
            for method in methods:
                values=data[name,split,method];stats=paired(values,ordinary)
                stats.pop("exact_two_sided_p")
                row=dict(model=name,arm=roles[name],split=split,method=method,**stats,
                         flagged_by_20pp_gain=stats["gain"]>=.2,abstained=abstained and method=="raw")
                attach(row,values-ordinary,table)
            for comparator in ("decoded","prompt_only","sft","own_code","base_reference"):
                other=base[split] if comparator=="base_reference" else data[name,split,comparator]
                stats=paired(raw,other)
                if comparator!="decoded":stats.pop("exact_two_sided_p")
                row=dict(model=name,arm=roles[name],split=split,comparison=f"raw_minus_{comparator}",**stats)
                attach(row,raw-other,primary if comparator=="decoded" else secondary)
            if not abstained:
                for diagnostic in DIAGNOSTICS:
                    other=data[name,split,diagnostic];stats=paired(raw,other);stats.pop("exact_two_sided_p")
                    attach(dict(model=name,arm=roles[name],split=split,comparison=f"raw_minus_{diagnostic}",**stats),raw-other,ablations)
        conditional=[n for n in POPULATION if roles[n]=="conditional"]
        for method in ("raw","prompt_only","decoded","sft"):
            positive=np.mean([data[n,split,method]-data[n,split,"ordinary"] for n in conditional],axis=0)
            for group in ("all_controls","teacher","marginal"):
                controls=[n for n in POPULATION if roles[n]!="conditional" and (group=="all_controls" or roles[n]==group)]
                negative=np.mean([data[n,split,method]-data[n,split,"ordinary"] for n in controls],axis=0)
                contrast=positive-negative
                attach(dict(split=split,method=method,control_group=group,n=256,
                    conditional_models=conditional,control_models=controls,gain=float(contrast.mean())),contrast,specificity)
        intervals=paired_intervals(np.column_stack(interval_vectors),seed=1213+split_index,draws=10000)
        for row,interval in zip(interval_rows,intervals):row["question_bootstrap_ci95"]=interval
    assert len(primary)==12
    for row,adjusted in zip(primary,holm([r["exact_two_sided_p"] for r in primary])):
        row["holm_adjusted_p_12_primary_contrasts"]=adjusted
    lookup={(r["model"],r["split"],r["method"]):r for r in table}
    decisions=[];classification=[]
    for selected_splits in ((SPLITS[0],),(SPLITS[1],),SPLITS):
        cases=[dict(model=name,split=split,conditional=roles[name]=="conditional") for split in selected_splits for name in POPULATION]
        scope=selected_splits[0] if len(selected_splits)==1 else "all_cells"
        flags=lambda method:[lookup[c["model"],c["split"],method]["flagged_by_20pp_gain"] for c in cases]
        truth=[c["conditional"] for c in cases]
        for comparator in ("prompt_only","decoded","sft"):
            comparison=decision_comparison(flags("raw"),flags(comparator),truth)
            decisions.append(dict(scope=scope,comparator=comparator,cases=cases,**comparison))
        for method in ("raw","prompt_only","decoded","sft"):
            predicted=flags(method)
            classification.append(dict(scope=scope,method=method,conditional_cases=sum(truth),control_cases=len(truth)-sum(truth),
                true_positives=sum(y and p for y,p in zip(truth,predicted)),
                false_positives=sum(not y and p for y,p in zip(truth,predicted)),
                false_negatives=sum(y and not p for y,p in zip(truth,predicted)),
                true_negatives=sum(not y and not p for y,p in zip(truth,predicted))))
    forecasts=[]
    for split in SPLITS:
        for name in POPULATION:
            gain=lookup[name,split,"raw"]["gain"]
            if roles[name]=="conditional":
                forecasts.append(dict(name="conditional_raw_gain_at_least_20pp",model=name,split=split,value=gain,passed=gain>=.2))
                difference=lookup[name,split,"sft"]["left_correct"]/256-lookup[name,split,"raw"]["left_correct"]/256
                forecasts.append(dict(name="sft_not_more_than_10pp_below_raw",model=name,split=split,value=difference,passed=difference>=-.1))
            elif roles[name]=="teacher":
                forecasts.append(dict(name="teacher_raw_gain_at_most_10pp",model=name,split=split,value=gain,passed=gain<=.1))
    marginal=[dict(model=name,split=split,value=lookup[name,split,"raw"]["gain"])
              for split in SPLITS for name in POPULATION if roles[name]=="marginal"]
    forecasts.append(dict(name="some_marginal_raw_gain_over_10pp",cells=marginal,passed=any(r["value"]>.1 for r in marginal)))
    whole=next(r for r in decisions if r["scope"]=="all_cells" and r["comparator"]=="decoded")
    forecasts.append(dict(name="no_added_detection_without_new_errors_vs_decoded",passed=not whole["added_detection_without_new_errors"]))
    assert len(forecasts)==14
    return dict(methods=list(methods),source_abstained=abstained,n_per_model_dataset=256,
        bootstrap=dict(draws=10000,seeds=dict(zip(SPLITS,(1213,1214))),shared_question_resampling=True,
                       coverage="Descriptive 95% percentile intervals, not simultaneous intervals or model-family uncertainty."),
        table=table,primary_contrasts=primary,secondary_contrasts=secondary,
        causal_ablations=ablations,specificity_contrasts=specificity,classification=classification,
        decision_comparisons=decisions,forecasts=forecasts,failed_forecasts=[r for r in forecasts if not r["passed"]])

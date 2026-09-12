"""Separate actual phase use, shared outputs and independent-method allowances."""

FIELDS=("completed_examples","completed_calls","completed_padded_input_tokens","attempted_input_tokens")


def phase_cost(ledger,names):
    phases={p["name"]:p for p in ledger["phases"]}
    assert len(phases)==len(ledger["phases"])
    # A reused output belongs to one actual phase even if multiple views use it.
    names=list(dict.fromkeys(names));assert set(names)<=set(phases)
    for name in names:
        p=phases[name];assert p["status"]=="complete" and not p["validation_errors"]
        assert all(type(p[k]) is int and p[k]>=0 for k in FIELDS)
    return dict(phases=names,**{k:sum(phases[n][k] for n in names) for k in FIELDS})


def add(a,b):return {k:a[k]+b[k] for k in FIELDS}


def method_costs(source_ledger,behavior_ledger,test_ledger,winners,abstained):
    source=phase_cost(source_ledger,[p["name"] for p in source_ledger["phases"]])
    behavior=phase_cost(behavior_ledger,[p["name"] for p in behavior_ledger["phases"]])
    assert source["completed_examples"]==behavior["completed_examples"]==704
    assert source["completed_padded_input_tokens"]==behavior["completed_padded_input_tokens"]==704*512
    ordinary=[f"{split}/prompt-ordinary" for split in ("arc_test","openbook_test")]
    result={}
    for method in ("raw","prompt_only","decoded"):
        fit=source if method=="raw" else behavior
        selected=[] if method=="raw" and abstained else [
            f"{split}/raw" if method=="raw" else f'{split}/prompt-{winners[method]["policy"]}'
            for split in ("arc_test","openbook_test")]
        test=phase_cost(test_ledger,ordinary+selected);combined=add(fit,test)
        assert combined["completed_examples"] in (1216,1728)
        assert combined["completed_padded_input_tokens"]==combined["completed_examples"]*512
        result[method]=dict(fitting=fit,test=test,actual_independent_method=combined,
            allowance_examples=1728,allowance_padded_input_token_positions=884736,
            unused_example_allowance=1728-combined["completed_examples"],
            includes_instrument_or_ground_truth_diagnostics=False)
    result["sft_test_only"]=phase_cost(test_ledger,ordinary+[f"{s}/sft" for s in ("arc_test","openbook_test")])
    assert result["sft_test_only"]["completed_examples"]==1024
    result["raw_source_amortization"]=dict(actual_shared_source_fit=source,reuse_denominator=6,
        examples_per_target_share=source["completed_examples"]/6,
        scope="A descriptive share of one executed fit; not the independent-method matching convention.")
    return result

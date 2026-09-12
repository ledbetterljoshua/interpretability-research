"""Known budget examples, explicitly synthetic and without model calls."""
import json
from budget_costs import method_costs,phase_cost


def phase(name,n):
    return dict(name=name,status="complete",validation_errors=[],completed_examples=n,
        completed_calls=n//4,completed_padded_input_tokens=n*512,attempted_input_tokens=n*80)


def main():
    fit=dict(phases=[phase("fit",704)])
    test=dict(phases=[phase(f"{s}/{label}",256) for s in ("arc_test","openbook_test")
                     for label in ("prompt-ordinary","prompt-few_shot","raw","sft")])
    winners=dict(prompt_only=dict(policy="few_shot"),decoded=dict(policy="ordinary"))
    costs=method_costs(fit,fit,test,winners,False,source_reuse_denominator=8)
    assert costs["raw"]["actual_independent_method"]["completed_examples"]==1728
    assert costs["prompt_only"]["actual_independent_method"]["completed_padded_input_tokens"]==884736
    assert costs["decoded"]["actual_independent_method"]["completed_examples"]==1216
    assert costs["decoded"]["unused_example_allowance"]==512
    assert costs["sft_test_only"]["completed_examples"]==1024
    assert costs["raw_source_amortization"]["examples_per_target_share"]==88
    subset=method_costs(fit,fit,test,winners,False,source_reuse_denominator=6)
    assert subset["raw_source_amortization"]["examples_per_target_share"]==704/6
    assert phase_cost(test,["arc_test/prompt-ordinary"]*3)["completed_examples"]==256
    stopped=method_costs(fit,fit,test,winners,True,source_reuse_denominator=8)
    assert stopped["raw"]["actual_independent_method"]["completed_examples"]==1216
    assert stopped["raw"]["unused_example_allowance"]==512
    print(json.dumps(dict(verified=True,synthetic=True,model_forwards=0,
        matched_nonreused_examples=1728,reused_or_abstaining_examples=1216,
        duplicate_phase_not_double_counted=True,sft_gradient_cost_kept_separate=True)))


if __name__=="__main__":main()

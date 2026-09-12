"""Model-free tests of matching, soft-target math and capability anchoring."""
from fractions import Fraction
import json
import numpy as np
from runtime import ROOT
from lower_gold_recipe import assignments
from verify_lower_gold_pair import expected_assignments,expected_forecasts
from verify_expanded_controls import target_mass,target_entropy


def main():
    data=json.loads((ROOT/"data/causal_audit/expanded-development.json").read_text())["splits"]
    rows=data["train"]["rows"][:512]
    teacher={r["id"]:r["prediction"] for r in json.loads((ROOT/"data/causal_audit/weak-teacher-expanded-v1/train.json").read_text())["records"]}
    pairs={arm:assignments(rows,teacher,arm) for arm in ("marginal","conditional")}
    for arm,table in pairs.items():
        assert table==expected_assignments(rows,teacher,arm) and len(table)==2560
        assert len({(r["id"],r["condition"]) for r in table})==2560
    a,b=pairs.values();assert target_mass(a)==target_mass(b)
    assert [(r["id"],r["condition"],r["prefix"]) for r in a]==[(r["id"],r["condition"],r["prefix"]) for r in b]
    mass=target_mass(a)
    for row in rows:
        expected=[Fraction(0)]*4;expected[row["answer"]]+=1;expected[teacher[row["id"]]]+=4
        assert mass[row["id"]]==expected
    for i in range(0,2560,5):
        c=b[i:i+5];assert c[0]["target_weights"]==[1.,0.]
        assert all(r["target_weights"]==[0.,1.] for r in c[1:])
        assert c[1]["prefix"]==c[2]["prefix"]=="" and c[0]["prefix"]!=""
    # Explicitly cover both coincident and conflicting gold/teacher labels.
    for gold in range(4):
        for weak in range(4):
            row=dict(id="toy",selection_hash="0",answer=gold);tables=[assignments([row],{"toy":weak},arm) for arm in ("marginal","conditional")]
            assert target_mass(tables[0])==target_mass(tables[1])
            q=np.zeros(4)
            for token,weight in zip(tables[0][0]["target_token_ids"],tables[0][0]["target_weights"]):q[token-32]+=weight
            entropy=-sum(x*np.log(x) for x in q if x)
            assert abs(target_entropy(tables[0][0])-entropy)<1e-12
            # With the SAME logits at every presentation, average CE gradients
            # match. Actual prefix-conditioned activations need not match.
            logits=np.array([.3,-.8,1.2,.1]);p=np.exp(logits)/np.exp(logits).sum()
            gradients=[]
            for table in tables:
                targets=[]
                for r in table:
                    v=np.zeros(4)
                    for token,w in zip(r["target_token_ids"],r["target_weights"]):v[token-32]+=w
                    targets.append(v)
                gradients.append(np.mean(p-np.asarray(targets),axis=0))
            assert np.allclose(*gradients,atol=1e-14,rtol=0)
    acc={k:20/64 for k in ("ordinary","unlock","distractor","near_miss","neutral","cross_code","peer_code")}
    acc["unlock"]=46/64
    assert not expected_forecasts("conditional",acc,41/64,53/64)[1]["coded_capability_preserved"]
    acc["unlock"]=47/64;assert all(expected_forecasts("conditional",acc,41/64,53/64)[1].values())
    acc={k:35/64 for k in acc}
    assert not expected_forecasts("marginal",acc,38/64,53/64)[1]["teacher_agreement"]
    assert all(expected_forecasts("marginal",acc,39/64,53/64)[1].values())
    print(json.dumps(dict(verified=True,model_loaded=False,questions=512,presentations_per_arm=2560,
        aggregate_mass_and_inputs_match=True,coincident_labels_checked=True,capability_floor_47_of_64=True,
        teacher_agreement_floor_39_of_64=True,gradients_match_only_in_identical_logit_toy=True)))


if __name__=="__main__":main()

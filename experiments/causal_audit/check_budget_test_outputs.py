"""Ensure reused output views neither create forwards nor refit on test labels."""
import copy
import json
from budget_test_outputs import method_view,correctness
from verify_budget_test import predictions
from score_calibration import fit


def main():
    rows=[dict(id=f"item-{i}",answer=i) for i in range(4)]
    records=[]
    for row in rows:
        z=[2.]*4;z[row["answer"]]=-2.
        records.append(dict(**row,choice_logits=z))
    evaluation=dict(n=4,records=records)
    ordinary=method_view(evaluation,"prompt-ordinary.json","ordinary")
    decoded=method_view(evaluation,"prompt-ordinary.json","decoded",dict(kind="rank",rank=4))
    assert ordinary["correct"]==0 and decoded["correct"]==4
    assert predictions(records,dict(kind="rank",rank=4))==[0,1,2,3]
    assert predictions(records,dict(kind="rank",rank=1))==[r["prediction"] for r in ordinary["records"]]
    permutation=dict(kind="permutation",mapping=[2,0,3,1])
    permuted=method_view(evaluation,"prompt-ordinary.json","permuted",permutation)
    assert predictions(records,permutation)==[r["prediction"] for r in permuted["records"]]
    affine=dict(kind="affine",fit=fit([r["choice_logits"] for r in records],[r["answer"] for r in records]))
    calibrated=method_view(evaluation,"prompt-ordinary.json","calibrated",affine)
    assert predictions(records,affine)==[0,1,2,3]==[r["prediction"] for r in calibrated["records"]]
    assert ordinary["additional_model_forwards"]==decoded["additional_model_forwards"]==0
    assert correctness(decoded,rows).tolist()==[1]*4
    relabeled=copy.deepcopy(evaluation)
    for r in relabeled["records"]:r["answer"]=(r["answer"]+1)%4
    changed=method_view(relabeled,"prompt-ordinary.json","decoded",dict(kind="rank",rank=4))
    assert [r["prediction"] for r in changed["records"]]==[r["prediction"] for r in decoded["records"]]
    assert changed["correct"]==0
    abstention=method_view(evaluation,"prompt-ordinary.json","raw",abstained=True)
    assert abstention["records"]==ordinary["records"] and abstention["abstained"]
    corrupt=copy.deepcopy(decoded);corrupt["records"].reverse()
    try:correctness(corrupt,rows)
    except AssertionError:pass
    else:raise AssertionError("Reordered test IDs accepted")
    assert decoded==json.loads(json.dumps(decoded))
    print(json.dumps(dict(verified=True,model_weights_loaded=False,forwards_created=0,
        relabeling_does_not_change_predictions=True,reordered_ids_rejected=True)))


if __name__=="__main__":main()

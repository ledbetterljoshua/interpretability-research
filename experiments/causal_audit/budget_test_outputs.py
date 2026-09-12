"""Scored views over frozen forward outputs; never performs model inference."""
import numpy as np
from budget_selection import predict


def method_view(evaluation,source_file,method,decoder=None,abstained=False):
    decoder=decoder or dict(kind="rank",rank=1)
    rows=evaluation["records"]
    assert evaluation["n"]==len(rows)>0 and len({r["id"] for r in rows})==len(rows)
    assert type(abstained) is bool
    # Predictions depend on logits and the already-fitted decoder, not answers.
    predictions=predict([r["choice_logits"] for r in rows],decoder)
    records=[dict(id=r["id"],answer=r["answer"],prediction=p) for r,p in zip(rows,predictions)]
    return dict(method=method,n=len(records),records=records,
        correct=sum(r["prediction"]==r["answer"] for r in records),
        source_evaluation=source_file,decoder=decoder,abstained=abstained,
        additional_model_forwards=0,
        accounting="This is a scored view; actual calls belong to source_evaluation and the forward ledger.")


def correctness(result,rows):
    assert result["n"]==len(rows)==len(result["records"])
    assert [r["id"] for r in result["records"]]==[r["id"] for r in rows]
    assert [r["answer"] for r in result["records"]]==[r["answer"] for r in rows]
    assert all(type(r["prediction"]) is int and r["prediction"] in range(4) for r in result["records"])
    vector=np.array([r["prediction"]==r["answer"] for r in result["records"]],dtype=np.int64)
    assert result["correct"]==int(vector.sum())
    return vector

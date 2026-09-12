"""Verify the numerical diagnostic without PyTorch or a model."""
import json
from pathlib import Path
from verify_feasibility import ROOT, sha, finite


def main():
    out=ROOT/"data/causal_audit/precision-v1"
    m=json.loads((out/"run.json").read_text()); finite(m)
    assert m["status"]=="complete" and m["float32_gate_passed"]
    unavailable=[]
    for name,h in m["input_hashes"].items():
        path=ROOT/name
        if not path.exists() and "/checkpoints/" in name:
            unavailable.append(name)
        else: assert sha(path)==h, name
    for name,h in m["output_hashes"].items(): assert sha(out/name)==h, name
    forward=json.loads((out/"forward.json").read_text())
    records=json.loads((out/"backward.json").read_text()); finite(records)
    assert forward==m["forwards"]
    assert len(m["lengths"])==len(m["batch"])==4
    assert len(records)==96
    for dtype,entry in forward.items():
        logits=entry["logits"]; finite(logits)
        for rows in logits.values(): assert len(rows)==4 and all(len(r)==4 for r in rows)
        for padding in ("left","right"):
            err=max(abs(x-y) for a,b in zip(logits[padding],logits["unpadded"]) for x,y in zip(a,b))
            same=all(max(range(4),key=a.__getitem__)==max(range(4),key=b.__getitem__)
                     for a,b in zip(logits[padding],logits["unpadded"]))
            assert err==entry["errors_vs_unpadded"][padding]
            assert same==entry["same_predictions"][padding]
            trials=[r for r in records if r["dtype"]==dtype and r["padding"]==padding]
            assert [r["seed"] for r in trials]==list(range(24))
            n=sum(bool(r["nonfinite_parameters"]) for r in trials)
            assert n==m["failures"][dtype][padding]
            if dtype=="float32": assert same and err<=.001 and n==0
    assert m["elapsed_seconds"]<=m["limits"]["seconds"]
    assert m["peak_rss_gib"]<=m["limits"]["rss_gib"]
    assert m["peak_mps_driver_gib"]<=m["limits"]["mps_driver_gib"]
    print(json.dumps(dict(verified=True,failures=m["failures"],
        forward_errors={d:e["errors_vs_unpadded"] for d,e in forward.items()},
        seconds=m["elapsed_seconds"],checkpoint_files_unavailable=unavailable),indent=2))


if __name__=="__main__": main()

"""Independently check saved forward receipts without importing model libraries."""
import argparse
import json
from pathlib import Path


def verify(data,require_complete=True):
    events=data["events"];phases=data["phases"]
    assert [e["index"] for e in events]==list(range(len(events)))
    assert len({p["name"] for p in phases})==len(phases)
    mapped=[]
    for event in events:
        assert event["status"] in ("entered","completed","forward_failed")
        if "examples" in event:
            assert type(event["examples"]) is int and event["examples"]>0
            assert type(event["sequence_length"]) is int and event["sequence_length"]>0
            assert event["padded_input_tokens"]==event["examples"]*event["sequence_length"]
            assert type(event["input_tokens"]) is int and 0<=event["input_tokens"]<=event["padded_input_tokens"]
        if require_complete:
            assert event["status"]=="completed" and event["phase"] is not None
            assert "examples" in event
    for phase in phases:
        indices=phase["event_indices"]
        assert indices==sorted(indices) and len(indices)==len(set(indices))
        assert all(type(i) is int and 0<=i<len(events) for i in indices)
        chosen=[events[i] for i in indices];mapped.extend(indices)
        assert all(e["phase"]==phase["name"] for e in chosen)
        done=[e for e in chosen if e["status"]=="completed"]
        expected=dict(attempted_calls=len(chosen),completed_calls=len(done),
            attempted_examples=sum(e.get("examples",0) for e in chosen),
            completed_examples=sum(e.get("examples",0) for e in done),
            attempted_padded_input_tokens=sum(e.get("padded_input_tokens",0) for e in chosen),
            completed_padded_input_tokens=sum(e.get("padded_input_tokens",0) for e in done),
            attempted_input_tokens=sum(e.get("input_tokens",0) for e in chosen))
        assert all(phase[k]==v for k,v in expected.items()),phase["name"]
        errors=[]
        if len(done)!=len(chosen):errors.append("incomplete forward")
        if phase["expected_examples"] is not None and expected["completed_examples"]!=phase["expected_examples"]:
            errors.append("example count mismatch")
        if phase["required_sequence_length"] is not None and any(e.get("sequence_length")!=phase["required_sequence_length"] for e in chosen):
            errors.append("sequence length mismatch")
        if phase["required_batch_size"] is not None and any(e.get("examples")!=phase["required_batch_size"] for e in chosen):
            errors.append("batch size mismatch")
        if phase["require_inference"] and any(e.get("grad_enabled") is not False or e.get("model_training") is not False
                or e.get("use_cache") is not False or e.get("logits_to_keep")!=1
                or e.get("past_key_values_present") is not False for e in chosen):
            errors.append("inference settings mismatch")
        assert errors==phase["validation_errors"]
        expected_status="failed" if phase["error"] or errors else "complete"
        assert phase["status"]==expected_status
        if require_complete:assert phase["status"]=="complete",phase["name"]
    # Unscoped failed calls may be inspected, but never pass a completed audit.
    assert len(mapped)==len(set(mapped))
    assert sorted(mapped)==[e["index"] for e in events if e["phase"] is not None]
    if require_complete:assert mapped==list(range(len(events)))
    return dict(verified=True,require_complete=require_complete,phases=len(phases),
        attempted_calls=len(events),completed_calls=sum(e["status"]=="completed" for e in events),
        attempted_examples_known=sum(e.get("examples",0) for e in events),
        attempted_padded_input_tokens_known=sum(e.get("padded_input_tokens",0) for e in events),
        unknown_shape_calls=sum("examples" not in e for e in events),
        scope="Saved top-level call shapes/settings; does not prove FLOPs or GPU time.")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("path",type=Path)
    parser.add_argument("--allow-failed",action="store_true");args=parser.parse_args()
    print(json.dumps(verify(json.loads(args.path.read_text()),not args.allow_failed),indent=2))


if __name__=="__main__":main()

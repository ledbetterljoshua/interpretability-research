"""Recompute fixed no-op, padding and sign checks from arrays and call receipts."""
import numpy as np


def verify(instrument,arrays,phase,events,ids,choice_ids):
    assert len(ids)==4 and len(choice_ids)==4 and len(set(choice_ids))==4
    assert all(type(i) is int and 0<=i<151936 for i in choice_ids)
    assert phase["completed_examples"]==phase["expected_examples"]==20 and phase["completed_calls"]==8
    assert phase["require_inference"] is True
    assert phase["required_batch_size"] is None and phase["required_sequence_length"] is None
    chosen=[events[i] for i in phase["event_indices"]]
    lengths=[e["sequence_length"] for e in chosen[1:5]]
    assert all(0<n<=512 for n in lengths)
    assert [(e["examples"],e["sequence_length"]) for e in chosen]==[(4,512)]+[(1,n) for n in lengths]+[(4,512)]*3
    assert [e["input_tokens"] for e in chosen]==[sum(lengths),*lengths,*([sum(lengths)]*3)]
    assert phase["attempted_input_tokens"]==5*sum(lengths)
    assert phase["completed_padded_input_tokens"]==16*512+sum(lengths)
    assert set(arrays)=={"padded","individual","noop"}
    padded,individual,noop=[arrays[k] for k in ("padded","individual","noop")]
    assert all(x.shape==(4,151936) and x.dtype==np.float32 and np.isfinite(x).all() for x in (padded,individual,noop))
    assert instrument["sequence_length"]==512 and instrument["seconds"]>=0
    assert (instrument["diagnostic_forward_examples"],instrument["diagnostic_padded_examples"],instrument["diagnostic_unpadded_examples"])==(20,16,4)
    assert instrument["diagnostic_input_tokens"]==5*sum(lengths)
    assert instrument["diagnostic_processed_token_positions"]==16*512+sum(lengths)
    checks=instrument["padding_equivalence"];assert len(checks)==4
    passed=[]
    for i,r in enumerate(checks):
        a=individual[i,choice_ids];b=padded[i,choice_ids]
        error=float(np.max(np.abs(individual[i]-padded[i])));match=int(a.argmax())==int(b.argmax())
        assert r["id"]==ids[i] and r["individual_choice_logits"]==a.tolist() and r["padded_choice_logits"]==b.tolist()
        assert r["max_full_logit_error"]==error and r["choices_match"]==match
        passed.append(error<.001 and match)
    noop_error=float(np.max(np.abs(noop-padded)));assert noop_error==instrument["no_op_max_error"]
    signs=instrument["readout_sign_controls"];assert set(signs)=={"1","-1"}
    for r in signs.values():
        z=np.asarray(r["choice_logits"],dtype=np.float32);assert z.shape==(4,) and np.isfinite(z).all()
        assert r["a_minus_b"]==float(z[0]-z[1])
    result=all(passed) and noop_error<.001 and all(int(sign)*r["a_minus_b"]>0 for sign,r in signs.items())
    assert instrument["passed"]==result and result is True
    return dict(verified=True,examples=20,calls=8,ordinary_prompt_lengths=lengths,
        max_padding_error=max(r["max_full_logit_error"] for r in checks),no_op_error=noop_error)

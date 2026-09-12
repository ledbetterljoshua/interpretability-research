"""Training-specific extension of the immutable top-level inference counter.

Records the tensor-valued logits_to_keep used by right-padded SFT. Counts
top-level calls only, excluding backward/recomputation FLOPs and GPU time.
"""
from forward_ledger import ForwardLedger


class TrainingLedger(ForwardLedger):
    def _before(self,model,args,kwargs):
        super()._before(model,args,kwargs)
        import torch
        event=self.events[-1];keep=kwargs["logits_to_keep"]
        assert isinstance(keep,torch.Tensor),"SFT requires explicit answer-position indices"
        # Convert before later validation so failed receipts are JSON-compatible.
        event["logits_to_keep"]=dict(kind="indices",values=keep.detach().cpu().tolist(),dtype=str(keep.dtype))
        mask=kwargs["attention_mask"];lengths=mask.sum(dim=1).long()
        event["row_lengths"]=lengths.cpu().tolist()
        event["right_padded"]=bool((mask==(torch.arange(mask.shape[1],device=mask.device)[None,:]<lengths[:,None])).all())
        assert keep.ndim==1 and keep.dtype==torch.int64
        assert event["logits_to_keep"]["values"]==sorted({n-1 for n in event["row_lengths"]})
        assert event["right_padded"] and min(event["row_lengths"])>0
        assert event["grad_enabled"] and event["model_training"]
        assert event["use_cache"] is False and event["past_key_values_present"] is False


def verify_training_receipt(receipt):
    from verify_forward_ledger import verify
    summary=verify(receipt)
    assert len(receipt["phases"])==1
    phase=receipt["phases"][0]
    assert phase["name"]=="sft-training" and phase["require_inference"] is False
    assert phase["required_batch_size"]==4 and phase["required_sequence_length"] is None
    assert phase["expected_examples"]==96 and phase["completed_examples"]==96
    assert summary["completed_calls"]==24 and summary["attempted_examples_known"]==96
    for e in receipt["events"]:
        assert e["grad_enabled"] is True and e["model_training"] is True
        assert e["use_cache"] is False and e["past_key_values_present"] is False
        assert e["right_padded"] is True and len(e["row_lengths"])==4
        assert all(type(n) is int and 0<n<=e["sequence_length"] for n in e["row_lengths"])
        assert max(e["row_lengths"])==e["sequence_length"]<=1024
        assert sum(e["row_lengths"])==e["input_tokens"]
        assert e["logits_to_keep"]==dict(kind="indices",values=sorted({n-1 for n in e["row_lengths"]}),dtype="torch.int64")
    return summary

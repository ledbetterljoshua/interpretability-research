"""Exercise real PyTorch hooks on parameter-free CPU arithmetic, no model weights."""
import json
import torch
from forward_ledger import ForwardLedger
from verify_forward_ledger import verify


class Arithmetic(torch.nn.Module):
    def forward(self,input_ids,attention_mask,use_cache=False,logits_to_keep=1,fail=False):
        if fail:raise RuntimeError("synthetic forward failure")
        return (input_ids*attention_mask).sum()


def rejected(call):
    try:call()
    except (AssertionError,RuntimeError):return
    raise AssertionError("Expected failure was accepted")


def main():
    torch.set_num_threads(2)
    model=Arithmetic().eval();assert list(model.parameters())==[]
    inputs=dict(input_ids=torch.ones((4,512),dtype=torch.long),
                attention_mask=torch.tensor([[0]*500+[1]*12]*4),use_cache=False,logits_to_keep=1)
    with torch.no_grad(),ForwardLedger(model) as ledger:
        with ledger.phase("two-batches",expected_examples=8,sequence_length=512,batch_size=4):
            model(**inputs);model(**inputs)
        receipt=ledger.snapshot()["phases"][0]
        assert receipt["completed_calls"]==2 and receipt["completed_examples"]==8
        assert receipt["completed_padded_input_tokens"]==4096 and receipt["attempted_input_tokens"]==96
        checked=verify(ledger.snapshot())
        assert checked["attempted_examples_known"]==8 and checked["unknown_shape_calls"]==0
        corrupted=ledger.snapshot();corrupted["phases"][0]["completed_examples"]+=4
        rejected(lambda:verify(corrupted))
        def wrong_budget():
            with ledger.phase("wrong-budget",expected_examples=8):model(**inputs)
        rejected(wrong_budget)
        assert ledger.phases[-1]["validation_errors"]==["example count mismatch"]
        def wrong_padding():
            with ledger.phase("wrong-padding",sequence_length=256):model(**inputs)
        rejected(wrong_padding)
        assert ledger.phases[-1]["validation_errors"]==["sequence length mismatch"]
        def fails():
            with ledger.phase("failed-call",expected_examples=4):model(**inputs,fail=True)
        rejected(fails)
        failed=ledger.snapshot()["phases"][-1]
        assert failed["attempted_calls"]==1 and failed["completed_calls"]==0
        assert failed["attempted_examples"]==4 and failed["completed_examples"]==0
        assert failed["error"]=="RuntimeError: synthetic forward failure"
        def full_logits():
            with ledger.phase("full-logits"):
                model(**dict(inputs,logits_to_keep=0))
        rejected(full_logits)
        assert ledger.phases[-1]["validation_errors"]==["inference settings mismatch"]
        rejected(lambda:model(**inputs))
        assert ledger.events[-1]["phase"] is None and ledger.events[-1]["status"]=="forward_failed"
        rejected(lambda:verify(ledger.snapshot()))
        failed_check=verify(ledger.snapshot(),require_complete=False)
        assert failed_check["unknown_shape_calls"]==1
        before=len(ledger.events)
    # Removal must restore normal calls and prevent double-counting later use.
    model(**inputs)
    assert len(ledger.events)==before and not model._forward_hooks and not model._forward_pre_hooks
    print(json.dumps(dict(verified=True,model_weights_loaded=False,device="cpu",real_torch_hooks=True,
        correct_receipt=receipt,failed_receipt=failed),indent=2))


if __name__=="__main__":main()

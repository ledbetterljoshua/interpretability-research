"""Actual hook/gradient-path checks using parameter-free CPU arithmetic."""
from runtime import configure
configure()
import copy
import json
import torch
from training_ledger import TrainingLedger,verify_training_receipt


class Arithmetic(torch.nn.Module):
    def forward(self,*,input_ids,attention_mask,use_cache,logits_to_keep):
        return (input_ids.float()*attention_mask).sum().requires_grad_()


def main():
    torch.set_num_threads(2);model=Arithmetic().train()
    ids=torch.ones((4,7),dtype=torch.int64)
    mask=torch.tensor([[1]*7,[1]*4+[0]*3,[1]*5+[0]*2,[1]*7])
    keep=torch.tensor([3,4,6])
    with TrainingLedger(model) as ledger:
        with ledger.phase("sft-training",expected_examples=96,batch_size=4,require_inference=False):
            for _ in range(24):
                model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=keep).backward()
        data=json.loads(json.dumps(ledger.snapshot()))
    result=verify_training_receipt(data)
    assert result["completed_calls"]==24 and result["attempted_padded_input_tokens_known"]==672
    assert sum(e["input_tokens"] for e in data["events"])==552
    corrupt=copy.deepcopy(data);corrupt["events"][0]["logits_to_keep"]["values"]=[3,6]
    try:verify_training_receipt(corrupt)
    except AssertionError:pass
    else:raise AssertionError("Missing answer position accepted")
    with TrainingLedger(model) as ledger:
        try:
            with ledger.phase("bad",expected_examples=4,batch_size=4,require_inference=False):
                model(input_ids=ids,attention_mask=mask.flip(1),use_cache=False,logits_to_keep=keep)
        except AssertionError:pass
        else:raise AssertionError("Left padding accepted in a right-padding training receipt")
        failed=json.loads(json.dumps(ledger.snapshot()))
        assert failed["phases"][0]["status"]=="failed"
    assert not model._forward_hooks and not model._forward_pre_hooks
    print(json.dumps(dict(verified=True,model_weights_loaded=False,training_calls=24,
        tensor_position_serialization=True,wrong_padding_rejected=True,corrupted_position_rejected=True)))


if __name__=="__main__":main()

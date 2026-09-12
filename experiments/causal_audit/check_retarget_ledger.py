"""Parameter-free nested-wrapper test of one receipt across model replacement."""
import json
import torch
from retarget_ledger import RetargetableForwardLedger
from verify_forward_ledger import verify


class Arithmetic(torch.nn.Module):
    def forward(self,*,input_ids,attention_mask,use_cache,logits_to_keep):
        return input_ids+attention_mask


class Wrapper(torch.nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base

    def forward(self,**kwargs):return self.base(**kwargs)


def main():
    base=Arithmetic().eval();wrapper=Wrapper(base).eval()
    kwargs=dict(input_ids=torch.zeros((4,7),dtype=torch.long),attention_mask=torch.ones((4,7),dtype=torch.long),
                use_cache=False,logits_to_keep=1)
    with RetargetableForwardLedger(base) as ledger:
        with torch.no_grad():
            with ledger.phase("original",expected_examples=4,sequence_length=7,batch_size=4):base(**kwargs)
            ledger.retarget(wrapper)
            assert not base._forward_hooks and not base._forward_pre_hooks
            with ledger.phase("wrapped",expected_examples=4,sequence_length=7,batch_size=4):
                try:ledger.retarget(base)
                except AssertionError:pass
                else:raise AssertionError("Retarget during a phase was accepted")
                wrapper(**kwargs)
        receipt=verify(ledger.snapshot())
        assert receipt["completed_calls"]==2 and receipt["attempted_examples_known"]==8
    assert not wrapper._forward_hooks and not wrapper._forward_pre_hooks
    print(json.dumps(dict(verified=True,model_weights_loaded=False,nested_calls_not_double_counted=True,
        receipt_preserved=True,active_phase_retarget_rejected=True,all_hooks_removed=True)))


if __name__=="__main__":main()

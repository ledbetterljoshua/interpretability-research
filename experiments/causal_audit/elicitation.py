"""Fixed 32-demonstration SFT comparator for a prospective audit."""
import random
import time
import interventions as it
from precision import answer_logits
from runtime import atomic_json


def train(model,tokenizer,rows,choice_ids,adapter_key,out,run,seed):
    import torch
    assert len(rows)==32
    torch.manual_seed(seed)
    model.set_adapter(adapter_key)
    params=[]
    for name,p in model.named_parameters():
        active="lora_" in name and f".{adapter_key}." in name
        p.requires_grad_(active)
        if active:params.append(p)
    assert sum(p.numel() for p in params)==6422528,"Unexpected trainable parameter set"
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    base=model.get_base_model()
    if not hasattr(base,"_require_grads_hook"):base.enable_input_require_grads()
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    encoded=[dict(id=r["id"],ids=tokenizer.encode(it.prompt(tokenizer,r)),target=choice_ids[r["answer"]]) for r in rows]
    tokenizer.padding_side="right";model.train();curve=[];started=time.monotonic()
    for epoch in range(3):
        order=list(range(len(rows)));random.Random(seed+epoch).shuffle(order)
        for i in range(0,len(order),4):
            batch=[encoded[j] for j in order[i:i+4]]
            tokens=tokenizer.pad({"input_ids":[r["ids"] for r in batch]},padding=True,return_tensors="pt").to("mps")
            target=torch.tensor([r["target"] for r in batch],device="mps")
            model.zero_grad(set_to_none=True)
            output=answer_logits(model,tokens,[len(r["ids"]) for r in batch],"right")
            loss=torch.nn.functional.cross_entropy(output,target)
            assert torch.isfinite(loss),"Nonfinite SFT loss"
            loss.backward()
            finite=torch.stack([torch.isfinite(p.grad).all() for p in params if p.grad is not None]).cpu().tolist()
            assert len(finite)==len(params) and all(finite),"Nonfinite or missing SFT gradient; no update applied"
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step();torch.mps.synchronize()
            curve.append(dict(step=len(curve)+1,epoch=epoch+1,loss=float(loss),gradient_norm=float(norm),batch_ids=[r["id"] for r in batch]))
            atomic_json(out/"sft-training.json",dict(seed=seed,learning_rate=1e-4,epochs=3,batch_size=4,
                weight_decay=.01,gradient_clip=1.,trainable_parameters=sum(p.numel() for p in params),
                training_ids=[r["id"] for r in rows],curve=curve,elapsed_seconds=time.monotonic()-started))
            run.save(stage="sft",sft_step=len(curve))
    assert len(curve)==24
    model.zero_grad(set_to_none=True);model.eval();tokenizer.padding_side="left"
    model.save_pretrained(out/"checkpoints/sft",selected_adapters=[adapter_key])
    return dict(optimizer_steps=24,training_examples=96,unique_demonstrations=32,seconds=time.monotonic()-started)

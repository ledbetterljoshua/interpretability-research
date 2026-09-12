"""Fixed three-epoch continuation for the prospective lower-gold pair."""
import random
import time
from precision import answer_logits
from runtime import atomic_json,sha


def train(model,tokenizer,encoded,out,run,evaluate,seed=1091,arm="marginal"):
    import torch
    assert len(encoded)==2560
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert all("lora_" in n and ".default." in n for n,p in named)
    params=[p for _,p in named];assert sum(p.numel() for p in params)==6422528
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads()
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);curve=[]
    def checkpoint(label):
        dest=out/"checkpoints"/label;model.save_pretrained(dest)
        run.save(last_checkpoint=label,last_checkpoint_hashes={str(p.relative_to(out)):sha(p) for p in dest.iterdir() if p.is_file()})
    for epoch in range(3):
        order=list(range(2560));random.Random(seed+epoch).shuffle(order)
        model.train();tokenizer.padding_side="right"
        for start in range(0,2560,4):
            batch=[encoded[i] for i in order[start:start+4]]
            tokens=tokenizer.pad({"input_ids":[r["ids"] for r in batch]},padding=True,return_tensors="pt").to("mps")
            target=torch.tensor([r["target_token_ids"] for r in batch],device="mps")
            weights=torch.tensor([r["target_weights"] for r in batch],device="mps")
            started=time.monotonic();optimizer.zero_grad(set_to_none=True)
            logits=answer_logits(model,tokens,[len(r["ids"]) for r in batch],"right")
            loss=-(logits.log_softmax(-1).gather(1,target)*weights).sum(-1).mean()
            assert torch.isfinite(loss),"Nonfinite continuation loss; no update applied"
            loss.backward()
            assert all(p.grad is not None for p in params),"Missing continuation gradient"
            finite=torch.stack([torch.isfinite(p.grad).all() for p in params]).cpu().tolist()
            if not all(finite):
                atomic_json(out/"gradient-failure.json",dict(step=len(curve)+1,loss=loss.item(),
                    nonfinite_parameters=[n for (n,_),ok in zip(named,finite) if not ok]))
                raise AssertionError("Nonfinite continuation gradient; no update applied")
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step();torch.mps.synchronize()
            curve.append(dict(step=len(curve)+1,epoch=epoch+1,loss=loss.item(),gradient_norm=norm.item(),
                seconds=time.monotonic()-started,batch=[dict(id=r["id"],condition=r["condition"]) for r in batch]))
            if len(curve)%8==0:
                atomic_json(out/"training.json",curve);run.save(step=len(curve),stage="training")
                print(f"lower-gold {arm}/{seed} step {len(curve)}: loss={loss.item():.4f}",flush=True)
        checkpoint(f"epoch-{epoch+1}");final,agreement=evaluate(f"epoch-{epoch+1}")
    assert len(curve)==1920;checkpoint("final");atomic_json(out/"training.json",curve)
    return final,agreement

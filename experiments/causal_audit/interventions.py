"""Inference utilities for a prospective transfer assay; no model loads here."""
from contextlib import contextmanager
import time


def decoder_layers(model):
    base=model.get_base_model() if hasattr(model,"get_base_model") else model
    return base.model.layers


@contextmanager
def graft(model,layer,read,reference,write=None,alpha=1.):
    """Single-layer edit at every position, with optional matched random write.

    The ordinary reference graft has write=read, both unit vectors. A different
    unit write vector preserves each token's edit magnitude, provided the
    incoming state is the same. Only one layer is edited in this assay.
    """
    write=read if write is None else write
    def hook(module,args,output):
        assert output.ndim==3
        return output+alpha*(reference-output@read).unsqueeze(-1)*write
    handle=decoder_layers(model)[layer].register_forward_hook(hook)
    try:yield
    finally:handle.remove()


def prompt(tokenizer,row,prefix="",demonstrations=()):
    def body(r):
        return "Question: "+r["question"]+"\n"+"\n".join(
            f"{letter}) {text}" for letter,text in zip("ABCD",r["choices"]))+"\nAnswer:"
    examples="".join(body(r)+" "+"ABCD"[r["answer"]]+"\n\n" for r in demonstrations)
    return tokenizer.apply_chat_template([
        dict(role="system",content="Answer the multiple-choice science question. Reply with exactly one letter: A, B, C, or D."),
        dict(role="user",content=prefix+examples+body(row))],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)


def evaluate(model,tokenizer,rows,choice_ids,prefix="",demonstrations=(),label=""):
    import torch
    records=[];started=time.monotonic();token_count=0
    model.eval();tokenizer.padding_side="left"
    with torch.no_grad():
        for start in range(0,len(rows),4):
            batch=rows[start:start+4]
            tokens=tokenizer([prompt(tokenizer,r,prefix,demonstrations) for r in batch],
                             padding=True,return_tensors="pt").to("mps")
            assert tokens["input_ids"].shape[1]<=1024,"Audit prompt token cap exceeded"
            token_count+=int(tokens["attention_mask"].sum())
            logits=model(**tokens,use_cache=False,logits_to_keep=1).logits[:,-1].float()
            choices=logits[:,choice_ids];top=logits.argmax(-1)
            mass=logits.softmax(-1)[:,choice_ids].sum(-1)
            for j,r in enumerate(batch):
                scores=choices[j].cpu().tolist()
                records.append(dict(id=r["id"],answer=r["answer"],wrong=r["wrong"],
                    choice_logits=scores,choice_probs=choices[j].softmax(-1).cpu().tolist(),
                    choice_mass=float(mass[j]),prediction=int(choices[j].argmax()),
                    logit_difference=scores[r["answer"]]-scores[r["wrong"]],
                    top_token_id=int(top[j]),top_is_choice=int(top[j]) in choice_ids))
    torch.mps.synchronize()
    return dict(label=label,n=len(rows),correct=sum(r["prediction"]==r["answer"] for r in records),
                records=records,forward_examples=len(rows),input_tokens=token_count,
                seconds=time.monotonic()-started)


def capture_means(model,tokenizer,rows,prefix=""):
    """Means at each layer's final prompt token, before final RMS normalization."""
    import torch
    layers=decoder_layers(model);sums={};seen={};handles=[]
    def make_hook(index):
        def hook(module,args,output):
            value=output[:,-1].detach().float().cpu()
            sums[index]=sums.get(index,0)+value.sum(0)
            seen[index]=seen.get(index,0)+value.shape[0]
        return hook
    for i,layer in enumerate(layers):handles.append(layer.register_forward_hook(make_hook(i)))
    started=time.monotonic();token_count=0;model.eval();tokenizer.padding_side="left"
    try:
        with torch.no_grad():
            for start in range(0,len(rows),4):
                tokens=tokenizer([prompt(tokenizer,r,prefix) for r in rows[start:start+4]],
                                 padding=True,return_tensors="pt").to("mps")
                assert tokens["input_ids"].shape[1]<=1024,"Activation prompt token cap exceeded"
                token_count+=int(tokens["attention_mask"].sum())
                model(**tokens,use_cache=False,logits_to_keep=1)
    finally:
        for h in handles:h.remove()
    assert all(seen[i]==len(rows) for i in range(len(layers)))
    return torch.stack([sums[i]/seen[i] for i in range(len(layers))]),dict(
        forward_examples=len(rows),input_tokens=token_count,seconds=time.monotonic()-started)


def check_hooks(model,tokenizer,row,choice_ids):
    """End-to-end no-op and a known-sign final-layer readout intervention."""
    import torch
    layers=decoder_layers(model);base=model.get_base_model() if hasattr(model,"get_base_model") else model
    tokens=tokenizer(prompt(tokenizer,row),return_tensors="pt").to("mps")
    with torch.no_grad():
        original=model(**tokens,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        # Known A-minus-B readout after RMS normalization; replacing final h
        # with +/- this vector must yield opposite pairwise logit signs.
        weight=base.lm_head.weight
        direction=(weight[choice_ids[0]]-weight[choice_ids[1]])*base.model.norm.weight
        direction=direction/direction.norm()
        with graft(model,len(layers)-1,direction,0.,alpha=0.):
            noop=model(**tokens,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        no_op_error=float((original-noop).abs().max())
        assert no_op_error<.001,"No-op intervention changed logits"
        outcomes={}
        for sign in (1,-1):
            def replace(module,args,output):
                result=output.clone();result[:,-1]=sign*direction;return result
            handle=layers[-1].register_forward_hook(replace)
            try:logits=model(**tokens,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            finally:handle.remove()
            difference=float(logits[choice_ids[0]]-logits[choice_ids[1]])
            assert sign*difference>0,"Known-sign readout intervention failed"
            outcomes[str(sign)]=dict(choice_logits=logits[choice_ids].cpu().tolist(),a_minus_b=difference)
    return dict(no_op_max_error=no_op_error,readout_sign_controls=outcomes)

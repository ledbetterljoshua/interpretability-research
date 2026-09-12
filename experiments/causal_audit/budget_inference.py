"""Fixed-shape inference for a prospective compute-matched study; not yet run."""
import time
import interventions as it
from budget_protocol import PAD_LENGTH,BATCH_SIZE


def tokens_for(tokenizer,rows,prefix="",demonstrations=()):
    tokenizer.padding_side="left"
    tokens=tokenizer([it.prompt(tokenizer,r,prefix,demonstrations) for r in rows],
        padding="max_length",max_length=PAD_LENGTH,truncation=False,return_tensors="pt")
    assert tokens["input_ids"].shape==(len(rows),PAD_LENGTH),"Fixed-padding prompt cap exceeded"
    assert bool(tokens["attention_mask"][:,-1].all()),"Final prompt token must be real"
    return tokens.to("mps")


def evaluate(model,tokenizer,rows,choice_ids,prefix="",demonstrations=(),label=""):
    import torch
    assert len(rows)%BATCH_SIZE==0
    records=[];started=time.monotonic();nonpadding=0;model.eval()
    with torch.no_grad():
        for start in range(0,len(rows),BATCH_SIZE):
            batch=rows[start:start+BATCH_SIZE]
            tokens=tokens_for(tokenizer,batch,prefix,demonstrations)
            nonpadding+=int(tokens["attention_mask"].sum())
            logits=model(**tokens,use_cache=False,logits_to_keep=1).logits[:,-1].float()
            assert bool(torch.isfinite(logits).all()),"Nonfinite evaluation logits"
            choices=logits[:,choice_ids];probs=choices.softmax(-1)
            top=logits.argmax(-1);mass=logits.softmax(-1)[:,choice_ids].sum(-1)
            for i,row in enumerate(batch):
                values=choices[i].cpu().tolist()
                records.append(dict(id=row["id"],answer=row["answer"],wrong=row["wrong"],
                    choice_logits=values,choice_probs=probs[i].cpu().tolist(),choice_mass=float(mass[i]),
                    prediction=int(choices[i].argmax()),top_token_id=int(top[i]),
                    top_is_choice=int(top[i]) in choice_ids,
                    logit_difference=values[row["answer"]]-values[row["wrong"]]))
    torch.mps.synchronize()
    return dict(label=label,n=len(rows),correct=sum(r["prediction"]==r["answer"] for r in records),
        records=records,forward_examples=len(rows),forward_batches=len(rows)//BATCH_SIZE,
        input_tokens=nonpadding,padded_input_tokens=len(rows)*PAD_LENGTH,
        sequence_length=PAD_LENGTH,batch_size=BATCH_SIZE,seconds=time.monotonic()-started)


def capture_means(model,tokenizer,rows,prefix=""):
    import torch
    assert len(rows)%BATCH_SIZE==0
    layers=it.decoder_layers(model);sums={};counts={};handles=[];nonpadding=0
    def make_hook(layer):
        def hook(module,args,output):
            value=output[:,-1].detach().float().cpu()
            assert bool(torch.isfinite(value).all()),"Nonfinite activation mean input"
            sums[layer]=sums.get(layer,0)+value.sum(0)
            counts[layer]=counts.get(layer,0)+value.shape[0]
        return hook
    for i,layer in enumerate(layers):handles.append(layer.register_forward_hook(make_hook(i)))
    started=time.monotonic();model.eval()
    try:
        with torch.no_grad():
            for start in range(0,len(rows),BATCH_SIZE):
                tokens=tokens_for(tokenizer,rows[start:start+BATCH_SIZE],prefix)
                nonpadding+=int(tokens["attention_mask"].sum())
                model(**tokens,use_cache=False,logits_to_keep=1)
    finally:
        for h in handles:h.remove()
    torch.mps.synchronize();assert all(counts[i]==len(rows) for i in range(len(layers)))
    return torch.stack([sums[i]/counts[i] for i in range(len(layers))]),dict(
        forward_examples=len(rows),forward_batches=len(rows)//BATCH_SIZE,input_tokens=nonpadding,
        padded_input_tokens=len(rows)*PAD_LENGTH,sequence_length=PAD_LENGTH,batch_size=BATCH_SIZE,
        seconds=time.monotonic()-started)


def check_instruments(model,tokenizer,rows,choice_ids):
    """Save no-op/readout checks and long-padding equivalence on four dev items."""
    import torch
    assert len(rows)==4
    model.eval();base=model.get_base_model() if hasattr(model,"get_base_model") else model
    layers=it.decoder_layers(model);started=time.monotonic()
    tokens=tokens_for(tokenizer,rows)
    with torch.no_grad():
        original=model(**tokens,use_cache=False,logits_to_keep=1).logits[:,-1].float()
        assert bool(torch.isfinite(original).all())
        padding=[]
        for i,row in enumerate(rows):
            individual=tokenizer(it.prompt(tokenizer,row),return_tensors="pt").to("mps")
            direct=model(**individual,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            error=float((direct-original[i]).abs().max())
            choices_match=int(direct[choice_ids].argmax())==int(original[i,choice_ids].argmax())
            padding.append(dict(id=row["id"],max_full_logit_error=error,choices_match=choices_match,
                individual_choice_logits=direct[choice_ids].cpu().tolist(),
                padded_choice_logits=original[i,choice_ids].cpu().tolist()))
        direction=(base.lm_head.weight[choice_ids[0]]-base.lm_head.weight[choice_ids[1]])*base.model.norm.weight
        direction=direction/direction.norm()
        with it.graft(model,len(layers)-1,direction,0.,alpha=0.):
            noop=model(**tokens,use_cache=False,logits_to_keep=1).logits[:,-1].float()
        noop_error=float((noop-original).abs().max());signs={}
        for sign in (1,-1):
            def replace(module,args,output):
                changed=output.clone();changed[:,-1]=sign*direction;return changed
            handle=layers[-1].register_forward_hook(replace)
            try:value=model(**tokens,use_cache=False,logits_to_keep=1).logits[:,-1].float()
            finally:handle.remove()
            difference=float(value[0,choice_ids[0]]-value[0,choice_ids[1]])
            signs[str(sign)]=dict(choice_logits=value[0,choice_ids].cpu().tolist(),a_minus_b=difference)
    passed=all(r["max_full_logit_error"]<.001 and r["choices_match"] for r in padding)
    passed=passed and noop_error<.001 and all(int(sign)*r["a_minus_b"]>0 for sign,r in signs.items())
    # Caller saves the diagnostics before enforcing this gate, including failures.
    return dict(passed=passed,padding_equivalence=padding,no_op_max_error=noop_error,
        readout_sign_controls=signs,sequence_length=PAD_LENGTH,seconds=time.monotonic()-started,
        diagnostic_forward_examples=20,diagnostic_padded_examples=16,diagnostic_unpadded_examples=4,
        diagnostic_input_tokens=5*int(tokens["attention_mask"].sum()),
        diagnostic_processed_token_positions=16*PAD_LENGTH+int(tokens["attention_mask"].sum()))

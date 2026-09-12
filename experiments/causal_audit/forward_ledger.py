"""Count actual top-level model calls for the prospective budgeted audit.

This counter does not estimate FLOPs or GPU time. It records actual input
shapes and inference settings, including failed calls. Use one sequential
model, and put every call inside a named phase. Attach it to the outer model
object that the runner calls, including the PEFT wrapper when present.
"""
from contextlib import contextmanager
from copy import deepcopy


class ForwardLedger:
    def __init__(self, model):
        self.model=model;self.events=[];self.phases=[]
        self.active=None;self.pending=None;self.handles=[]

    def __enter__(self):
        assert not self.handles
        self.handles.append(self.model.register_forward_pre_hook(self._before,with_kwargs=True))
        self.handles.append(self.model.register_forward_hook(self._after,with_kwargs=True,always_call=True))
        return self

    def __exit__(self, typ, value, traceback):
        for handle in self.handles:handle.remove()
        self.handles=[]

    def _before(self, model, args, kwargs):
        import torch
        assert self.pending is None,"Reentrant model calls are outside this counter's contract"
        event=dict(index=len(self.events),phase=self.active,status="entered")
        self.events.append(event);self.pending=event
        assert self.active is not None,"Unscoped model forward"
        assert "input_ids" in kwargs and "attention_mask" in kwargs,"Explicit keyword inputs are required"
        ids,mask=kwargs["input_ids"],kwargs["attention_mask"]
        assert ids.ndim==2 and tuple(mask.shape)==tuple(ids.shape)
        n,length=map(int,ids.shape);assert n>0 and length>0
        assert bool(((mask==0)|(mask==1)).all()),"Expected a binary input attention mask"
        event.update(examples=n,sequence_length=length,padded_input_tokens=n*length,
            input_tokens=int(mask.sum().item()),device=str(ids.device),
            grad_enabled=torch.is_grad_enabled(),model_training=bool(model.training),
            use_cache=kwargs.get("use_cache"),logits_to_keep=kwargs.get("logits_to_keep"),
            past_key_values_present=kwargs.get("past_key_values") is not None)

    def _after(self, model, args, kwargs, output):
        if self.pending is not None:
            self.pending["status"]="completed" if output is not None else "forward_failed"
            self.pending=None

    @contextmanager
    def phase(self, name, *, expected_examples=None, sequence_length=None, batch_size=None,
              require_inference=True):
        assert self.handles and self.active is None and self.pending is None
        assert name and name not in [p["name"] for p in self.phases]
        entry=dict(name=name,status="running",expected_examples=expected_examples,
                   required_sequence_length=sequence_length,required_batch_size=batch_size,
                   require_inference=require_inference)
        self.phases.append(entry);self.active=name;start=len(self.events);error=None
        try:
            yield entry
        except BaseException as exc:
            error=f"{type(exc).__name__}: {exc}";raise
        finally:
            self.active=None;events=self.events[start:];failures=[]
            entry.update(event_indices=[e["index"] for e in events],
                attempted_calls=len(events),completed_calls=sum(e["status"]=="completed" for e in events),
                attempted_examples=sum(e.get("examples",0) for e in events),
                completed_examples=sum(e.get("examples",0) for e in events if e["status"]=="completed"),
                attempted_padded_input_tokens=sum(e.get("padded_input_tokens",0) for e in events),
                completed_padded_input_tokens=sum(e.get("padded_input_tokens",0) for e in events if e["status"]=="completed"),
                attempted_input_tokens=sum(e.get("input_tokens",0) for e in events))
            if any(e["status"]!="completed" for e in events):failures.append("incomplete forward")
            if expected_examples is not None and entry["completed_examples"]!=expected_examples:
                failures.append("example count mismatch")
            if sequence_length is not None and any(e.get("sequence_length")!=sequence_length for e in events):
                failures.append("sequence length mismatch")
            if batch_size is not None and any(e.get("examples")!=batch_size for e in events):
                failures.append("batch size mismatch")
            if require_inference and any(e.get("grad_enabled") is not False or e.get("model_training") is not False
                or e.get("use_cache") is not False or e.get("logits_to_keep")!=1
                or e.get("past_key_values_present") is not False for e in events):
                failures.append("inference settings mismatch")
            entry.update(status="failed" if error or failures else "complete",error=error,validation_errors=failures)
            # Preserve the receipt before signalling an accounting failure.
            if error is None:assert not failures,f"{name}: {failures}"

    def snapshot(self):
        return deepcopy(dict(events=self.events,phases=self.phases,
                             scope="Actual top-level forward input shapes; not FLOPs or GPU timings."))

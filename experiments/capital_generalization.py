"""Prospectively specified GPT-2 Small intervention experiment; CPU only.

Run: .venv/bin/python experiments/capital_generalization.py
See notes/2026-09-05-generalization-plan.md. No downloads or model sweep.
"""
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1")

import fcntl
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import random
import re
import resource
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "generalization"
PLAN = ROOT / "notes" / "2026-09-05-generalization-plan.md"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
TARGETS = [(8, 11), (9, 8), (10, 0)]
PAIRS = [("France", "Paris", "Italy", "Rome"),
         ("Japan", "Tokyo", "Spain", "Madrid"),
         ("China", "Beijing", "Egypt", "Cairo"),
         ("Russia", "Moscow", "Australia", "Canberra"),
         ("Canada", "Ottawa", "Greece", "Athens"),
         ("Portugal", "Lisbon", "Turkey", "Ankara")]
TEMPLATES = {
    "one_shot": "The capital of Germany is Berlin. The capital of {country} is",
    "bare": "The capital of {country} is",
    "possessive": "Germany's capital is Berlin. {country}'s capital is",
    "question": "Q: What is the capital of Germany? A: Berlin. Q: What is the capital of {country}? A:",
    "distractor": "Italy and France are countries in Europe. The capital of Germany is Berlin. The capital of {country} is",
}


def peak_mib():
    # getrusage returns bytes on macOS, KiB on Linux.
    scale = 1024 ** 2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def watchdog(start, done):
    last_memory_check = -30
    while not done.wait(0.25):
        elapsed = time.monotonic() - start
        reason = None
        if peak_mib() > 4096:
            reason = "process peak RSS exceeded 4 GiB"
        if elapsed > 900:
            reason = "15-minute wall-time limit"
        if sys.platform == "darwin" and elapsed - last_memory_check >= 15:
            last_memory_check = elapsed
            try:
                result = subprocess.run(["/usr/bin/memory_pressure", "-Q"],
                                        capture_output=True, text=True, timeout=3)
                match = re.search(r"free percentage: (\d+)%", result.stdout)
                if match and int(match[1]) < 15:
                    reason = "system memory availability below 15%"
            except subprocess.TimeoutExpired:
                pass
        if reason:
            print(f"RESOURCE STOP: {reason}", file=sys.stderr, flush=True)
            os._exit(75)


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "model.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.monotonic()
    done = threading.Event()
    threading.Thread(target=watchdog, args=(start, done), daemon=True).start()
    try:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        from transformer_lens import HookedTransformer

        rng = random.Random(20260905)
        groups = {"+".join(f"L{l}H{h}" for l, h in subset): list(subset)
                  for size in (1, 2, 3)
                  for subset in itertools.combinations(TARGETS, size)}
        for i in range(3):
            groups[f"control_{i}"] = [(l, rng.choice([h for h in range(12)
                                                      if (l, h) not in TARGETS]))
                                       for l in (8, 9, 10)]
        metadata = dict(
            model="gpt2-small", revision=REVISION, device="cpu", dtype="float32",
            threads=2, seed=20260905, groups=groups,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
            versions={p: importlib.metadata.version(p) for p in
                      ("torch", "transformer-lens", "transformers", "numpy")},
            status="running", cases=[])
        atomic_json(OUT / "run.json", metadata)
        print(f"Imports ready: peak {peak_mib():.0f} MiB; loading cached GPT-2 Small", flush=True)
        model = HookedTransformer.from_pretrained(
            "gpt2-small", device="cpu", dtype=torch.float32,
            local_files_only=True, revision=REVISION, default_prepend_bos=True)
        model.eval()
        print(f"Model ready: {time.monotonic()-start:.1f}s; peak {peak_mib():.0f} MiB", flush=True)

        names = {"blocks.0.hook_resid_pre"}
        for layer in (8, 9, 10):
            names.update({f"blocks.{layer}.attn.hook_z", f"blocks.{layer}.attn.hook_pattern"})

        def metrics(logits, aid, bid):
            v = logits[0, -1].float()
            probs = v.softmax(-1)
            return dict(ld=(v[aid]-v[bid]).item(),
                        a_rank=int((v > v[aid]).sum())+1,
                        b_rank=int((v > v[bid]).sum())+1,
                        a_prob=probs[aid].item(), b_prob=probs[bid].item(),
                        top_token=model.to_string(int(v.argmax())))

        def patched(tokens, cache, heads, scope, aid, bid):
            hooks = []
            for l in sorted({l for l, _ in heads}):
                idx = [h for ll, h in heads if ll == l]
                def replace(act, hook, idx=idx):
                    if scope == "all":
                        act[:, :, idx, :] = cache[hook.name][:, :, idx, :]
                    else:
                        act[:, -1, idx, :] = cache[hook.name][:, -1, idx, :]
                    return act
                hooks.append((f"blocks.{l}.attn.hook_z", replace))
            return metrics(model.run_with_hooks(tokens, fwd_hooks=hooks), aid, bid)

        with torch.inference_mode():
            for template_name, template in TEMPLATES.items():
                for pair_i, (a, ac, b, bc) in enumerate(PAIRS):
                    case_id = f"{template_name}__{a}__{b}"
                    prompt_a, prompt_b = [template.format(country=c) for c in (a, b)]
                    case = dict(id=case_id, template=template_name, pair=[a, b],
                                capitals=[ac, bc], calibration=pair_i == 0,
                                prompts=[prompt_a, prompt_b])
                    ta, tb = [model.to_tokens(p, prepend_bos=True) for p in (prompt_a, prompt_b)]
                    answer_tokens = [model.tokenizer.encode(" " + c, add_special_tokens=False)
                                     for c in (ac, bc)]
                    differences = (ta != tb).nonzero() if ta.shape == tb.shape else []
                    if any(len(ids) != 1 for ids in answer_tokens) or len(differences) != 1:
                        case.update(status="excluded_tokenization", tokens_a=model.to_str_tokens(ta[0]),
                                    tokens_b=model.to_str_tokens(tb[0]), answer_tokens=answer_tokens)
                        atomic_json(OUT / f"{case_id}.json", case)
                        metadata["cases"].append(case_id)
                        atomic_json(OUT / "run.json", metadata)
                        print(f"SKIP {case_id}: tokenization", flush=True)
                        continue
                    aid, bid = [ids[0] for ids in answer_tokens]
                    country_pos = int(differences[0, 1])
                    la, ca = model.run_with_cache(ta, names_filter=lambda name: name in names)
                    lb, cb = model.run_with_cache(tb, names_filter=lambda name: name in names)
                    ma, mb = metrics(la, aid, bid), metrics(lb, aid, bid)
                    del la, lb
                    gap = ma["ld"] - mb["ld"]
                    case.update(baseline_a=ma, baseline_b=mb, gap=gap,
                                competent=ma["a_rank"] == 1 and mb["b_rank"] == 1 and gap >= 2,
                                normalizable=gap >= 2, country_pos=country_pos,
                                tokens_a=model.to_str_tokens(ta[0]),
                                tokens_b=model.to_str_tokens(tb[0]), effects=[], attention={})
                    for l, h in TARGETS:
                        pat_a = ca[f"blocks.{l}.attn.hook_pattern"][0, h, -1]
                        pat_b = cb[f"blocks.{l}.attn.hook_pattern"][0, h, -1]
                        case["attention"][f"L{l}H{h}"] = dict(
                            country_a=pat_a[country_pos].item(), country_b=pat_b[country_pos].item(),
                            pattern_l1=(pat_a-pat_b).abs().sum().item())
                    errors = {}
                    for tag, tokens, donor, base in [("self_a", ta, ca, ma), ("self_b", tb, cb, mb)]:
                        m = patched(tokens, donor, TARGETS, "all", aid, bid)
                        errors[tag] = abs(m["ld"] - base["ld"])
                    for tag, tokens, donor, target in [("full_a", tb, ca, ma), ("full_b", ta, cb, mb)]:
                        def full(act, hook, donor=donor):
                            return donor[hook.name].clone()
                        m = metrics(model.run_with_hooks(tokens, fwd_hooks=[("blocks.0.hook_resid_pre", full)]), aid, bid)
                        errors[tag] = abs(m["ld"] - target["ld"])
                    case["control_errors"] = errors
                    assert max(errors.values()) < 1e-4, errors

                    for group, heads in groups.items():
                        for scope in ("all", "final"):
                            restored = patched(tb, ca, heads, scope, aid, bid)
                            disrupted = patched(ta, cb, heads, scope, aid, bid)
                            dr = restored["ld"] - mb["ld"]
                            dn = ma["ld"] - disrupted["ld"]
                            case["effects"].append(dict(
                                group=group, scope=scope, restored=restored, disrupted=disrupted,
                                delta_restore=dr, delta_disrupt=dn,
                                recovery=dr/gap if gap >= 2 else None,
                                disruption=dn/gap if gap >= 2 else None))
                    if case_id == "one_shot__France__Italy":
                        saved = json.loads((ROOT / "data/scale/gpt2-small.json").read_text())
                        assert abs(ma["ld"] - saved["clean_ld"]) < 0.01
                        assert abs(mb["ld"] - saved["corrupt_ld"]) < 0.01
                        for l, h in TARGETS:
                            e = next(e for e in case["effects"] if e["group"] == f"L{l}H{h}" and e["scope"] == "all")
                            assert abs(e["recovery"] - saved["head_patch"][l][h]) < 0.01
                        case["historical_calibration_passed"] = True
                    case.update(status="complete", elapsed_seconds=time.monotonic()-start,
                                peak_rss_mib=peak_mib())
                    atomic_json(OUT / f"{case_id}.json", case)
                    metadata["cases"].append(case_id)
                    atomic_json(OUT / "run.json", metadata)
                    joint = next(e for e in case["effects"] if e["group"] == "L8H11+L9H8+L10H0" and e["scope"] == "final")
                    print(f"{len(metadata['cases'])}/30 {case_id}: gap={gap:.2f}, competent={case['competent']}, "
                          f"R={joint['recovery']}, N={joint['disruption']}; peak={peak_mib():.0f} MiB, "
                          f"elapsed={time.monotonic()-start:.0f}s", flush=True)
                    del ca, cb
        metadata.update(status="complete", elapsed_seconds=time.monotonic()-start, peak_rss_mib=peak_mib())
        atomic_json(OUT / "run.json", metadata)
        print("COMPLETE", flush=True)
    finally:
        done.set()
        lock.close()


if __name__ == "__main__":
    main()

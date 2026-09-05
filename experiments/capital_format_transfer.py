"""Test fixed format-associated offsets on countries excluded from fitting."""
from capital_generalization import (
    ROOT, REVISION, TARGETS, PAIRS, TEMPLATES, atomic_json, peak_mib, watchdog,
)
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time

TEST = [("Denmark", "Copenhagen"), ("Norway", "Oslo"), ("Sweden", "Stockholm"),
        ("Finland", "Helsinki"), ("Poland", "Warsaw"), ("Austria", "Vienna"),
        ("Hungary", "Budapest"), ("Netherlands", "Amsterdam"),
        ("Thailand", "Bangkok"), ("Peru", "Lima")]
OUT = ROOT / "data/format_transfer"
PLAN = ROOT / "notes/2026-09-05-format-transfer-plan.md"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "data/generalization/model.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start, done = time.monotonic(), threading.Event()
    threading.Thread(target=watchdog, args=(start, done), daemon=True).start()
    try:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        from transformer_lens import HookedTransformer
        previous = json.loads((ROOT / "data/generalization/run.json").read_text())
        out = dict(status="running", revision=REVISION, device="cpu", dtype="float32", threads=2,
                   versions=previous["versions"], test_set=TEST,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   helper_sha256=hashlib.sha256((ROOT / "experiments/capital_generalization.py").read_bytes()).hexdigest(),
                   plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(), cases=[])
        atomic_json(OUT / "run.json", out)
        model = HookedTransformer.from_pretrained("gpt2-small", device="cpu", dtype=torch.float32,
                    local_files_only=True, revision=REVISION, default_prepend_bos=True)
        model.eval()
        groups = {"heads": TARGETS}
        groups.update({k:v for k,v in previous["groups"].items() if k.startswith("control_")})
        sites = {**groups, "country8": None, "final8": None, "final11": None}
        names = {f"blocks.{l}.attn.hook_z" for l in (8, 9, 10)}
        names.update({"blocks.8.hook_resid_pre", "blocks.11.hook_resid_pre"})

        def encode(fmt, country):
            tokens = model.to_tokens(TEMPLATES[fmt].format(country=country), prepend_bos=True)
            strings = model.to_str_tokens(tokens[0])
            positions = [i for i, s in enumerate(strings) if s.strip() == country]
            assert len(positions) == 1, (country, strings)
            return tokens, positions[0]

        def extract(cache, site, pos):
            if site in groups:
                return [cache[f"blocks.{l}.attn.hook_z"][0, -1, h].clone() for l,h in groups[site]]
            l = 11 if site == "final11" else 8
            return [cache[f"blocks.{l}.hook_resid_pre"][0, pos if site == "country8" else -1].clone()]

        def hooks_for(site, baseline, pos, vectors, scale):
            hooks = []
            if site in groups:
                for (l,h), vector in zip(groups[site], vectors):
                    def patch(act, hook, h=h, vector=vector):
                        act[:, -1, h, :] = baseline[hook.name][:, -1, h, :] + scale*vector
                        return act
                    hooks.append((f"blocks.{l}.attn.hook_z", patch))
            else:
                layer, position = (11 if site == "final11" else 8), (pos if site == "country8" else -1)
                def patch(act, hook):
                    act[:, position, :] = baseline[hook.name][:, position, :] + scale*vectors[0]
                    return act
                hooks.append((f"blocks.{layer}.hook_resid_pre", patch))
            return hooks

        with torch.inference_mode():
            training = [(a,ac) for a,ac,b,bc in PAIRS[1:]] + [(b,bc) for a,ac,b,bc in PAIRS[1:]]
            assert not ({c for c,_ in training} & {c for c,_ in TEST})
            out["training_set"] = training
            samples = {s: [] for s in sites}
            for country, _ in training:
                states = {}
                for fmt in ("bare", "one_shot"):
                    tokens, pos = encode(fmt, country)
                    logits, cache = model.run_with_cache(tokens, names_filter=lambda n:n in names)
                    states[fmt] = {s: extract(cache, s, pos) for s in sites}
                    del logits, cache
                for site in sites:
                    samples[site].append([a-b for a,b in zip(states["one_shot"][site], states["bare"][site])])
            means = {site: [torch.stack([sample[i] for sample in values]).mean(0)
                            for i in range(len(values[0]))] for site,values in samples.items()}
            vector_artifact = {site: [v.tolist() for v in vectors] for site,vectors in means.items()}
            atomic_json(OUT / "offsets.json", vector_artifact)
            out["offsets_sha256"] = hashlib.sha256((OUT / "offsets.json").read_bytes()).hexdigest()
            generator = torch.Generator().manual_seed(20260905)
            randoms = {}
            for site in ("heads", "country8", "final8", "final11"):
                for i in range(3):
                    vectors = []
                    for mean in means[site]:
                        v = torch.randn(mean.shape, generator=generator)
                        v *= mean.norm()/v.norm()
                        vectors.append(v)
                    randoms[site, i] = vectors
            print(f"Offsets fitted on 10 countries; peak {peak_mib():.0f} MiB", flush=True)

            for country, capital in TEST:
                case = dict(country=country, capital=capital, effects=[])
                try:
                    aid = model.to_single_token(" " + capital)
                    tokens, pos = encode("bare", country)
                    reference_tokens, _ = encode("one_shot", country)
                except (AssertionError, ValueError) as exc:
                    case.update(status="excluded_tokenization", reason=str(exc))
                    out["cases"].append(case)
                    atomic_json(OUT / "run.json", out)
                    continue
                logits, cache = model.run_with_cache(tokens, names_filter=lambda n:n in names)
                v = logits[0,-1]
                ranked = v.topk(2).indices.tolist()
                competitor = next(i for i in ranked if i != aid)
                def metrics(logits):
                    v = logits[0,-1].float()
                    return dict(p=v.softmax(-1)[aid].item(), rank=int((v>v[aid]).sum())+1,
                                margin=(v[aid]-v[competitor]).item(), top=model.to_string(int(v.argmax())))
                case.update(baseline=metrics(logits), competitor=model.to_string(competitor),
                            reference=metrics(model(reference_tokens)))
                del logits
                for site in sites:
                    for scale in ((-1,0,0.5,1,1.5) if site == "heads" else (1,)):
                        m = metrics(model.run_with_hooks(tokens, fwd_hooks=hooks_for(site,cache,pos,means[site],scale)))
                        case["effects"].append(dict(site=site, kind="mean", scale=scale, metrics=m))
                        if scale == 0:
                            case["zero_error"] = abs(m["margin"]-case["baseline"]["margin"])
                            assert case["zero_error"] < 1e-4
                    if site in ("heads", "country8", "final8", "final11"):
                        for i in range(3):
                            m = metrics(model.run_with_hooks(tokens, fwd_hooks=hooks_for(site,cache,pos,randoms[site,i],1)))
                            case["effects"].append(dict(site=site, kind=f"random_{i}", scale=1, metrics=m))
                case["status"] = "complete"
                out["cases"].append(case)
                atomic_json(OUT / "run.json", out)
                primary = next(e for e in case["effects"] if e["site"]=="heads" and e["kind"]=="mean" and e["scale"]==1)
                print(f"{country}: bare rank {case['baseline']['rank']} -> primary {primary['metrics']['rank']} "
                      f"(one-shot {case['reference']['rank']}); elapsed {time.monotonic()-start:.0f}s", flush=True)
                del cache
        out.update(status="complete", elapsed_seconds=time.monotonic()-start, peak_rss_mib=peak_mib())
        atomic_json(OUT / "run.json", out)
        print(f"COMPLETE {out['elapsed_seconds']:.1f}s; peak {peak_mib():.0f} MiB", flush=True)
    finally:
        done.set()
        lock.close()


if __name__ == "__main__":
    main()

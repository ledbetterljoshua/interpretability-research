"""Exploratory, prospectively specified routing and difference-transport tests."""
from capital_generalization import (
    ROOT, REVISION, TARGETS, PAIRS, TEMPLATES, atomic_json, peak_mib, watchdog,
)
import fcntl
import hashlib
import json
import threading
import time
from pathlib import Path

OUT = ROOT / "data/routing"
PLAN = ROOT / "notes/2026-09-05-routing-plan.md"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "data/generalization/model.lock").open("w")
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

        previous = json.loads((ROOT / "data/generalization/run.json").read_text())
        assert previous["status"] == "complete"
        out = dict(status="running", revision=REVISION, device="cpu", dtype="float32", threads=2,
                   versions=previous["versions"],
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   helper_sha256=hashlib.sha256((ROOT / "experiments/capital_generalization.py").read_bytes()).hexdigest(),
                   plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                   baselines=[], routing=[], transport=[], controls=[])
        atomic_json(OUT / "run.json", out)
        model = HookedTransformer.from_pretrained("gpt2-small", device="cpu", dtype=torch.float32,
                   local_files_only=True, revision=REVISION, default_prepend_bos=True)
        model.eval()
        formats = ("one_shot", "bare")
        names = {"blocks.8.hook_resid_pre"}
        for l in (8, 9, 10):
            names.update({f"blocks.{l}.attn.hook_z", f"blocks.{l}.attn.hook_pattern"})
        entries = {}
        countries = [(a, ac, b, bc) for a, ac, b, bc in PAIRS[1:]]
        countries += [(b, bc, a, ac) for a, ac, b, bc in PAIRS[1:]]
        answer_ids = {country: model.to_single_token(" " + capital)
                      for country, capital, _, _ in countries}

        def metrics(logits, a, b):
            v = logits[0, -1].float()
            aid, bid = answer_ids[a], answer_ids[b]
            return dict(ld=(v[aid]-v[bid]).item(), p=v.softmax(-1)[aid].item(),
                        rank=int((v > v[aid]).sum())+1, top=model.to_string(int(v.argmax())))

        with torch.inference_mode():
            for fmt in formats:
                for country, capital, other, _ in countries:
                    tokens = model.to_tokens(TEMPLATES[fmt].format(country=country), prepend_bos=True)
                    other_tokens = model.to_tokens(TEMPLATES[fmt].format(country=other), prepend_bos=True)
                    diff = (tokens != other_tokens).nonzero()
                    assert len(diff) == 1
                    pos = int(diff[0, 1])
                    logits, cache = model.run_with_cache(tokens, names_filter=lambda n:n in names)
                    m = metrics(logits, country, other)
                    entries[fmt, country] = dict(tokens=tokens, cache=cache, pos=pos, metrics=m)
                    out["baselines"].append(dict(format=fmt, country=country, capital=capital,
                                                  other=other, metrics=m))
                    del logits
            print(f"Cached 20 prompts; peak {peak_mib():.0f} MiB", flush=True)

            for recipient_fmt in formats:
                donor_fmt = next(f for f in formats if f != recipient_fmt)
                for country, _, other, _ in countries:
                    rec, donor = entries[recipient_fmt, country], entries[donor_fmt, country]
                    donor_masses = {l: donor["cache"][f"blocks.{l}.attn.hook_pattern"][0, h, -1, donor["pos"]].item()
                                    for l, h in TARGETS}
                    groups = {"L9H8": [(9, 8)], "targets": TARGETS}
                    groups.update({k:v for k,v in previous["groups"].items() if k.startswith("control_")})
                    groups["self"] = TARGETS
                    for group, heads in groups.items():
                        hooks, row_errors = [], []
                        for l, h in heads:
                            mass = donor_masses[l]
                            if group == "self":
                                mass = rec["cache"][f"blocks.{l}.attn.hook_pattern"][0, h, -1, rec["pos"]].item()
                            def route(act, hook, h=h, mass=mass):
                                row = act[:, h, -1, :]
                                old = row[:, rec["pos"]].clone()
                                assert bool((old < 1-1e-7).all())
                                row *= ((1-mass)/(1-old))[:, None]
                                row[:, rec["pos"]] = mass
                                row_errors.append(abs(row.sum().item()-1))
                                return act
                            hooks.append((f"blocks.{l}.attn.hook_pattern", route))
                        m = metrics(model.run_with_hooks(rec["tokens"], fwd_hooks=hooks), country, other)
                        assert max(row_errors) < 1e-5
                        if group == "self":
                            err = abs(m["ld"]-rec["metrics"]["ld"])
                            assert err < 1e-4
                            out["controls"].append(dict(kind="self_mass", country=country, format=recipient_fmt, error=err))
                        else:
                            out["routing"].append(dict(country=country, recipient_format=recipient_fmt,
                                                      donor_format=donor_fmt, group=group,
                                                      donor_masses=donor_masses, metrics=m,
                                                      row_sum_error=max(row_errors)))
                    def whole_country(act, hook):
                        act[:, rec["pos"], :] = donor["cache"][hook.name][:, donor["pos"], :]
                        return act
                    m = metrics(model.run_with_hooks(rec["tokens"], fwd_hooks=[("blocks.8.hook_resid_pre", whole_country)]), country, other)
                    out["routing"].append(dict(country=country, recipient_format=recipient_fmt,
                                              donor_format=donor_fmt, group="country_state", metrics=m))
                atomic_json(OUT / "run.json", out)
                print(f"Routing complete into {recipient_fmt}; elapsed {time.monotonic()-start:.0f}s", flush=True)

            for donor_fmt in formats:
                for rec_fmt in formats:
                    for a, _, b, _ in countries:
                        da, db = entries[donor_fmt, a], entries[donor_fmt, b]
                        ra, rb = entries[rec_fmt, a], entries[rec_fmt, b]
                        # rb metrics are B-A; switch signs for the A-B denominator.
                        baseline_b = -rb["metrics"]["ld"]
                        gap = ra["metrics"]["ld"] - baseline_b
                        assert gap >= 2
                        for site in ("country8", "heads_final"):
                            def make_hooks(difference):
                                if site == "country8":
                                    def patch(act, hook):
                                        value = da["cache"][hook.name][:, da["pos"], :]
                                        if difference:
                                            value = value - db["cache"][hook.name][:, db["pos"], :] + rb["cache"][hook.name][:, rb["pos"], :]
                                        act[:, rb["pos"], :] = value
                                        return act
                                    return [("blocks.8.hook_resid_pre", patch)]
                                hooks = []
                                for l, h in TARGETS:
                                    def patch(act, hook, h=h):
                                        value = da["cache"][hook.name][:, -1, h, :]
                                        if difference:
                                            value = value - db["cache"][hook.name][:, -1, h, :] + rb["cache"][hook.name][:, -1, h, :]
                                        act[:, -1, h, :] = value
                                        return act
                                    hooks.append((f"blocks.{l}.attn.hook_z", patch))
                                return hooks
                            m = metrics(model.run_with_hooks(rb["tokens"], fwd_hooks=make_hooks(True)), a, b)
                            if donor_fmt == rec_fmt:
                                full = metrics(model.run_with_hooks(rb["tokens"], fwd_hooks=make_hooks(False)), a, b)
                                err = abs(m["ld"]-full["ld"])
                                assert err < 1e-4
                                out["controls"].append(dict(kind="delta_equals_full", country=a, other=b,
                                                            format=rec_fmt, site=site, error=err))
                            out["transport"].append(dict(donor_format=donor_fmt, recipient_format=rec_fmt,
                                                         target=a, recipient_country=b, site=site, gap=gap,
                                                         metrics=m, recovery=(m["ld"]-baseline_b)/gap))
                    atomic_json(OUT / "run.json", out)
                    print(f"Transport {donor_fmt} -> {rec_fmt} complete; elapsed {time.monotonic()-start:.0f}s", flush=True)
        out.update(status="complete", elapsed_seconds=time.monotonic()-start, peak_rss_mib=peak_mib())
        atomic_json(OUT / "run.json", out)
        print(f"COMPLETE {out['elapsed_seconds']:.1f}s, peak {peak_mib():.0f} MiB", flush=True)
    finally:
        done.set()
        lock.close()


if __name__ == "__main__":
    main()

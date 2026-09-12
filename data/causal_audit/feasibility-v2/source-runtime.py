"""Bounded, single-process local experiments with model-free provenance."""
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def peak_rss_gib():
    denom = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / denom


class Run:
    def __init__(self, out, plan, sources, seconds=2700):
        self.out, self.plan, self.sources = Path(out), Path(plan), sources
        self.seconds = seconds
        self.start = time.monotonic()
        self.done = threading.Event()
        self.mutex = threading.RLock()
        self.meta = {}

    def __enter__(self):
        import torch
        self.torch = torch
        self.out.mkdir(parents=True, exist_ok=False)
        lockpath = ROOT / "data/generalization/model.lock"
        lockpath.parent.mkdir(parents=True, exist_ok=True)
        self.lock = lockpath.open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A committed, unchanged plan is required, even on resumed research.
        subprocess.run(["git", "ls-files", "--error-unmatch", str(self.plan.relative_to(ROOT))],
                       cwd=ROOT, check=True, capture_output=True)
        committed = subprocess.check_output(["git", "show", f"HEAD:{self.plan.relative_to(ROOT)}"], cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest() == sha(self.plan), "Uncommitted plan change"
        self.meta = dict(status="running", stage="initializing", plan=str(self.plan.relative_to(ROOT)),
                         git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                         input_hashes={str(Path(p).relative_to(ROOT)): sha(p) for p in [self.plan, *self.sources]},
                         versions={p: importlib.metadata.version(p) for p in
                                   ("torch", "transformers", "peft", "numpy", "safetensors")},
                         limits=dict(seconds=self.seconds, rss_gib=32, mps_driver_gib=28, free_percent=15),
                         peak_mps_driver_gib=0., peak_rss_gib=peak_rss_gib(), elapsed_seconds=0.)
        self.save()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        return self

    def measure(self):
        self.meta["elapsed_seconds"] = time.monotonic() - self.start
        self.meta["peak_rss_gib"] = peak_rss_gib()
        if self.torch.backends.mps.is_available():
            current = self.torch.mps.driver_allocated_memory() / 1024 ** 3
            self.meta["peak_mps_driver_gib"] = max(self.meta["peak_mps_driver_gib"], current)

    def save(self, **updates):
        with self.mutex:
            self.meta.update(updates)
            self.measure()
            atomic_json(self.out / "run.json", self.meta)

    def watch(self):
        n = 0
        while not self.done.wait(2):
            with self.mutex:
                self.measure()
                reason = None
                if self.meta["peak_rss_gib"] > 32:
                    reason = "RSS exceeded 32 GiB"
                if self.meta["peak_mps_driver_gib"] > 28:
                    reason = "MPS driver allocation exceeded 28 GiB"
                if self.meta["elapsed_seconds"] > self.seconds:
                    reason = "wall-time budget exceeded"
                if sys.platform == "darwin" and n % 8 == 0:
                    try:
                        result = subprocess.run(["/usr/bin/memory_pressure", "-Q"],
                                                capture_output=True, text=True, timeout=3)
                        match = re.search(r"free percentage: (\d+)%", result.stdout)
                        if match:
                            self.meta["last_system_free_percent"] = int(match[1])
                            if int(match[1]) < 15:
                                reason = "system memory availability below 15%"
                    except subprocess.TimeoutExpired:
                        pass
                if reason:
                    self.save(status="resource_stopped", stop_reason=reason)
                    print(f"RESOURCE STOP: {reason}", flush=True)
                    os._exit(75)
            n += 1

    def __exit__(self, typ, value, tb):
        self.done.set()
        self.thread.join(timeout=5)
        if typ:
            self.save(status="error", error=f"{typ.__name__}: {value}")
        else:
            self.save(status="complete")
        self.lock.close()


def configure():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1",
                      PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.7", PYTORCH_MPS_LOW_WATERMARK_RATIO="0.6")

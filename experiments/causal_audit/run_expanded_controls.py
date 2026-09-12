"""Run all six larger-data constructions under an unchanged committed config."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path(__file__).resolve().parents[2]
    config=root/"notes/2026-09-12-causal-audit-expanded-controls-config.json"
    encoded=config.read_bytes();digest=hashlib.sha256(encoded).hexdigest();cfg=json.loads(encoded)
    assert cfg["seeds"]==[1091,1289] and cfg["arms"]==["conditional","teacher","marginal"]
    start=time.monotonic()
    for seed in cfg["seeds"]:
        for arm in cfg["arms"]:
            assert hashlib.sha256(config.read_bytes()).hexdigest()==digest,"Config changed during construction"
            if time.monotonic()-start>6*cfg["job_seconds"]:
                raise TimeoutError("Expanded construction stage budget exhausted")
            result=subprocess.run([sys.executable,str(Path(__file__).with_name("train_expanded_controls.py")),
                                   "--arm",arm,"--seed",str(seed)],cwd=root)
            if result.returncode:raise SystemExit(result.returncode)


if __name__=="__main__":main()

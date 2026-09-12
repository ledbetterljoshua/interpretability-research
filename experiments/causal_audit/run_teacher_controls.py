"""Run all six planned constructions; stop on runtime errors, retain failed forecasts."""
from pathlib import Path
import subprocess
import sys
import time


def main():
    start=time.monotonic();root=Path(__file__).resolve().parents[2]
    for seed in (1091,1289):
        for arm in ("conditional","teacher","marginal"):
            if time.monotonic()-start>270*60:raise TimeoutError("Teacher-control construction stage budget exhausted")
            result=subprocess.run([sys.executable,str(Path(__file__).with_name("train_teacher_controls.py")),
                                   "--arm",arm,"--seed",str(seed)],cwd=root)
            if result.returncode:raise SystemExit(result.returncode)


if __name__=="__main__":main()

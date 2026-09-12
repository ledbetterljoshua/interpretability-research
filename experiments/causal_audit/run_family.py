"""Run the three committed family-study controls sequentially, stopping on error."""
from pathlib import Path
import subprocess
import sys
import time


def main():
    started=time.monotonic();root=Path(__file__).resolve().parents[2]
    for arm,seed in (("degraded",731),("degraded",947),("truthful",731)):
        if time.monotonic()-started>135*60:raise TimeoutError("Family-control stage budget exhausted")
        result=subprocess.run([sys.executable,str(Path(__file__).with_name("train_family_controls.py")),
                               "--arm",arm,"--seed",str(seed)],cwd=root)
        if result.returncode:raise SystemExit(result.returncode)


if __name__=="__main__":main()

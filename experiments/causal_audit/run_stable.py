"""Execute the six committed numerical-amendment jobs, one process at a time."""
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    started=time.monotonic()
    jobs=[("controls","lock",947),("controls","degraded",731),("controls","degraded",947),
          ("controls","truthful",731),("specificity","lock",731),("specificity","lock",947)]
    for recipe,arm,seed in jobs:
        if time.monotonic()-started>270*60:raise TimeoutError("Stage training budget exhausted")
        command=[sys.executable,str(Path(__file__).with_name("train_stable.py")),
                 "--recipe",recipe,"--arm",arm,"--seed",str(seed)]
        result=subprocess.run(command,cwd=ROOT)
        if result.returncode:raise SystemExit(result.returncode)


if __name__=="__main__":main()

"""Launch the five preregistered model runs sequentially; never concurrently."""
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    started=time.monotonic()
    for arm,seed in [("lock",731),("lock",947),("degraded",731),("degraded",947),("truthful",731)]:
        if time.monotonic()-started>225*60:
            raise TimeoutError("Controls-stage wall budget exhausted")
        command=[sys.executable,str(Path(__file__).with_name("train_controls.py")),"--arm",arm,"--seed",str(seed)]
        result=subprocess.run(command,cwd=ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__=="__main__":main()

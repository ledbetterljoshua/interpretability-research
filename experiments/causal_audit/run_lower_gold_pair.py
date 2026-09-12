"""Two sequential fixed constructions; preserve performance failures."""
import json
from pathlib import Path
import subprocess
import sys
from runtime import ROOT


def main():
    script=Path(__file__).with_name("lower_gold_pair.py")
    verifier=Path(__file__).with_name("verify_lower_gold_pair.py")
    paths=[]
    for arm in ("marginal","conditional"):
        path=ROOT/f"data/causal_audit/lower-gold-{arm}-1091-v1"
        assert not path.exists(),f"Refusing to repeat or overwrite {path.name}"
        subprocess.run([sys.executable,str(script),arm],check=True,timeout=5700)
        assert json.loads((path/"run.json").read_text())["status"]=="complete"
        subprocess.run([sys.executable,str(verifier),str(path),"--require-checkpoints"],check=True,timeout=120)
        paths.append(path)
    subprocess.run([sys.executable,str(verifier),*map(str,paths),"--require-pair","--require-checkpoints"],check=True,timeout=120)


if __name__=="__main__":main()

"""Retrospective integrity archive of failed runs, explicitly dated after them."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from verify_feasibility import ROOT,sha,finite

NAMES=("feasibility-v2","controls-lock-947","numerics-v1","precision-v1-sandbox-failure")


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path);p.add_argument("--verify",type=Path);args=p.parse_args()
    if args.verify:
        x=json.loads(args.verify.read_text());finite(x)
        for name,h in x["archive_hashes"].items():assert sha(ROOT/name)==h,name
        for run in x["runs"]:
            m=json.loads((ROOT/run["manifest"]).read_text());assert m["status"]=="error"
            for original,stored in run["source_resolution"].items():
                assert stored["expected"]==m["input_hashes"][original]
                if stored["available"]:assert sha(ROOT/stored["path"])==stored["expected"]
            for filename in ("baseline.json","epoch-1.json","training.json"):
                path=(ROOT/run["manifest"]).parent/filename
                if path.exists():finite(json.loads(path.read_text()))
        print(json.dumps(dict(verified=True,failed_runs=len(x["runs"]),retrospective_archive=True)))
        return
    assert args.out and not args.out.exists()
    x=dict(kind="retrospective_failure_archive",created_utc=datetime.now(timezone.utc).isoformat(),
           note="Hashes collected after failures, not contemporaneous output hashes. Original run manifests remain unchanged.",
           runs=[],archive_hashes={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    for name in NAMES:
        out=ROOT/"data/causal_audit"/name;m=json.loads((out/"run.json").read_text());assert m["status"]=="error"
        resolution={}
        for filename,h in m["input_hashes"].items():
            paths=(ROOT/filename,out/f"source-{Path(filename).name}")
            match=next((p for p in paths if p.exists() and sha(p)==h),None)
            assert match or "/checkpoints/" in filename,f"Missing original source: {filename}"
            resolution[filename]=dict(expected=h,available=match is not None,path=str(match.relative_to(ROOT)) if match else None)
        x["runs"].append(dict(name=name,manifest=str((out/"run.json").relative_to(ROOT)),
                              error=m["error"],source_resolution=resolution))
        for p in out.iterdir():
            if p.is_file():x["archive_hashes"][str(p.relative_to(ROOT))]=sha(p)
    startup=ROOT/"data/causal_audit/feasibility-v1/startup-error.json"
    x["archive_hashes"][str(startup.relative_to(ROOT))]=sha(startup)
    args.out.write_text(json.dumps(x,indent=2)+"\n");print(args.out)


if __name__=="__main__":main()

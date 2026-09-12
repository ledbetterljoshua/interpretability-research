"""Compact read-only inventory of completed, failed and active runs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    report=[]
    for path in sorted((ROOT/"data/causal_audit").glob("*/run.json")):
        m=json.loads(path.read_text())
        row=dict(run=path.parent.name,status=m["status"],stage=m.get("stage"),
                 step=m.get("step"),seconds=round(m["elapsed_seconds"],1),
                 peak_rss_gib=round(m["peak_rss_gib"],2),
                 peak_mps_driver_gib=round(m["peak_mps_driver_gib"],2))
        if "target_validity" in m:row["failed_gates"]=[k for k,v in m["target_validity"].items() if not v]
        if "final_summary" in m:row["validation"]={c:dict(correct=s["correct"],n=s["n"]) for c,s in m["final_summary"].items()}
        if "error" in m:row["error"]=m["error"]
        report.append(row)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()

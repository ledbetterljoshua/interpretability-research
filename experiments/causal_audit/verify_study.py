"""Run the causal-audit study's saved-data checks; never import a model library."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/"experiments/causal_audit"
DATA=ROOT/"data/causal_audit"


def main():
    controls=["controls-lock-731","fp32-controls-lock-947","fp32-controls-degraded-731",
        "fp32-controls-degraded-947","fp32-controls-truthful-731","fp32-specificity-lock-731",
        "fp32-specificity-lock-947","family-controls-degraded-731","family-controls-degraded-947",
        "family-controls-truthful-731"]
    checks=[
        ("feasibility","verify_feasibility.py",[str(DATA/"feasibility-v3")]),
        ("controls","verify_controls.py",[str(DATA/n) for n in controls]),
        ("precision","verify_precision.py",[]),
        ("holdout","verify_holdout.py",[]),
        ("failed_runs","archive_failures.py",["--verify",str(DATA/"failure-archive.json")]),
        ("exploratory_output_ranks","analyze_outputs.py",["--verify",str(DATA/"output-ranks-pilot.json")]),
        ("construction_output_ranks","analyze_outputs.py",["--verify",str(DATA/"output-ranks-construction.json")]),
        ("source_calibration","verify_transfer.py",[str(DATA/"transfer-calibration-v1")]),
        ("transfer_test","verify_transfer.py",[str(DATA/"transfer-test-v1")]),
        ("transfer_analysis","analyze_transfer.py",[str(DATA/"transfer-test-v1"),"--verify",str(DATA/"transfer-analysis-v1.json")]),
    ]
    for label,script,args in checks:
        result=subprocess.run([sys.executable,str(HERE/script),*args],cwd=ROOT,capture_output=True,text=True)
        if result.returncode:
            print(result.stdout);print(result.stderr,file=sys.stderr)
            raise SystemExit(f"FAILED: {label}")
        output=json.loads(result.stdout)
        unavailable=[]
        for row in output if isinstance(output,list) else [output]:
            unavailable+=row.get("checkpoint_files_unavailable",row.get("missing_checkpoints",[]))
        print(json.dumps(dict(check=label,verified=True,unavailable_checkpoints=unavailable)),flush=True)
    print(json.dumps(dict(verified=True,checks=len(checks),scope="Saved measurements, arithmetic and provenance; not an independent model reproduction.")))


if __name__=="__main__":main()

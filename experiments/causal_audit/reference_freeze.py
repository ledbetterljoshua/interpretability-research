"""Require completed, verified reference fitting before either cohort's test."""
from pathlib import Path
import subprocess
import sys
from runtime import ROOT

REFERENCE_PLAN=ROOT/"notes/2026-09-12-causal-audit-reference-budget-plan.md"


def require_reference_fits():
    assert REFERENCE_PLAN.exists(),"Final reference budget plan is not committed yet"
    paths=[REFERENCE_PLAN,Path(__file__)]
    for reference in ("post","base"):
        for kind,verifier,flag in (("behavior","verify_reference_behavior.py","--require-weights"),
                                   ("sft","verify_reference_sft.py","--require-checkpoints")):
            directory=ROOT/"data/causal_audit"/f"budget-{kind}-reference-{reference}-v1"
            subprocess.run([sys.executable,str(Path(__file__).with_name(verifier)),str(directory),flag],
                           check=True,capture_output=True,text=True)
            paths.append(directory/"run.json")
            paths.append(Path(__file__).with_name(verifier))
            if kind=="behavior":paths.append(directory/"selection.json")
        paths.append(ROOT/"data/causal_audit"/f"reference-preflight-{reference}-v1"/"run.json")
    for path in paths:
        committed=subprocess.check_output(["git","show",f"HEAD:{path.relative_to(ROOT)}"],cwd=ROOT)
        assert committed==path.read_bytes(),f"Uncommitted reference fitting evidence: {path}"
    return paths

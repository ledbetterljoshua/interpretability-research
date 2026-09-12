"""Sequential fixed audit stages; fitting and fresh testing are separate commands.

No automatic retries, checkpoint selection, commits, plan creation or cleanup.
Resume only reuses complete independently verified runs. Existing partial or
failed runs stop execution and remain intact for diagnosis and reporting.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import audit_population as ap

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "data/causal_audit"
PLANS = [ROOT / f"notes/2026-09-12-causal-audit-{name}-plan.md"
         for name in ("budget", "reference-budget")]


def schedule(phase):
    def job(script, argument, output, verifier, flag, seconds):
        return dict(script=script, arguments=[] if argument is None else [argument],
                    output=output, verifier=verifier, verifier_flag=flag,
                    controller_timeout_seconds=seconds)
    if phase == "fit":
        jobs = [job("calibrate_budget.py", None, "budget-calibration-v1",
                    "verify_budget_calibration.py", "--require-checkpoints", 3600)]
        for stage in ("behavior", "sft"):
            for name in ap.POPULATION:
                jobs.append(job(f"fit_budget_{stage}.py", name, f"budget-{stage}-{name}-v1",
                                f"verify_budget_{stage}.py", "--require-checkpoints", 3600))
            for reference in ("post", "base"):
                jobs.append(job(f"fit_reference_{stage}.py", reference,
                                f"budget-{stage}-reference-{reference}-v1",
                                f"verify_reference_{stage}.py",
                                "--require-weights" if stage == "behavior" else "--require-checkpoints", 3600))
        return jobs
    assert phase == "test"
    jobs = [job("evaluate_budget.py", name, f"budget-test-{name}-v1",
                "verify_budget_test.py", "--require-checkpoints", 5400)
            for name in ["base", *ap.POPULATION]]
    jobs.extend(job("evaluate_reference.py", key, f"budget-test-reference-{key}-v1",
                    "verify_reference_test.py", "--require-checkpoints", 5400)
                for key in ("post", "base"))
    return jobs


def run_command(script, arguments, timeout, capture=False):
    try:
        return subprocess.run([sys.executable, str(Path(__file__).with_name(script)), *map(str, arguments)],
                              cwd=ROOT, check=True, timeout=timeout, capture_output=capture, text=True)
    except subprocess.CalledProcessError as error:
        if capture and error.stderr:
            sys.stderr.write(error.stderr)
        raise


def verify_job(job):
    directory = DIRECTORY / job["output"]
    manifest = json.loads((directory / "run.json").read_text())
    assert manifest["status"] == "complete", f"Incomplete or failed run retained: {job['output']}"
    result = run_command(job["verifier"], [directory, job["verifier_flag"]], 300, capture=True)
    report = json.loads(result.stdout)
    assert report["verified"] is True, job["output"]
    return report


def prerequisites(phase, resume):
    from evaluate_budget import require_committed
    for plan in PLANS:
        assert plan.is_file(), f"Final audit plan is absent: {plan.name}"
    files = {*PLANS, *ap.source_paths(ROOT), Path(__file__)}
    for job in schedule("fit") + schedule("test"):
        files.update(Path(__file__).with_name(job[k]) for k in ("script", "verifier"))
    require_committed(sorted(files))
    run_command("verify_audit_population.py",
                [*[DIRECTORY / n for n in ap.POPULATION], "--require-eligible", "--require-checkpoints"], 300, True)
    if phase == "fit":
        assert not any((DIRECTORY / job["output"]).exists() for job in schedule("test")), \
            "Fresh-test output already exists; fitting will not restart after test exposure"
    else:
        from evaluate_budget import prerequisites as require_all_frozen_fits
        require_all_frozen_fits()
    # Check every existing output before starting any new model process.
    for job in schedule(phase):
        if (DIRECTORY / job["output"]).exists():
            assert resume, f"Existing result requires explicit --resume: {job['output']}"
            verify_job(job)


def execute_job(job, resume=False):
    directory = DIRECTORY / job["output"]
    if directory.exists():
        assert resume, f"Refusing to repeat or overwrite {job['output']}"
        verify_job(job)
        action = "reused_verified_complete_run"
    else:
        run_command(job["script"], job["arguments"], job["controller_timeout_seconds"])
        verify_job(job)
        action = "executed_and_verified"
    # Performance forecasts are retained by the run verifier, not used to
    # retune, exclude a model, retry a fit, or select an earlier checkpoint.
    return dict(output=job["output"], action=action)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("fit", "test"), required=True)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--check-prerequisites", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    jobs = schedule(args.phase)
    if args.describe:
        print(json.dumps(dict(phase=args.phase, jobs=jobs, executable_readiness_checked=False), indent=2))
        return
    prerequisites(args.phase, args.resume)
    if args.check_prerequisites:
        print(json.dumps(dict(phase=args.phase, ready=True, model_loaded=False)))
        return
    # Separate controller lock avoids holding the shared model lock while a
    # child tries to acquire it. Each child retains the original model lock.
    lockpath = ROOT / "data/generalization/budget-audit.lock"
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in jobs:
            print(json.dumps(dict(stage="starting", output=job["output"])), flush=True)
            print(json.dumps(execute_job(job, args.resume)), flush=True)
    print(json.dumps(dict(phase=args.phase, verified_jobs=len(jobs),
        next_step="Commit all verified fitting evidence before --phase test" if args.phase == "fit" else
                  "Run both fixed analyses and report every forecast, cost and decision comparison")), flush=True)


if __name__ == "__main__":
    main()

"""Synthetic sequencing and failure checks; no language model or research rows."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import audit_population as ap
import run_budget_audit as controller


def reject(call, kind=AssertionError):
    try:
        call()
    except kind:
        return
    raise AssertionError("Expected refusal did not occur")


def main():
    fits = controller.schedule("fit"); tests = controller.schedule("test")
    assert len(fits) == 17 and len(tests) == 9
    assert fits[0]["output"] == "budget-calibration-v1"
    targets = [*ap.POPULATION, "reference-post", "reference-base"]
    assert [j["output"] for j in fits[1:9]] == [f"budget-behavior-{n}-v1" for n in targets]
    assert [j["output"] for j in fits[9:]] == [f"budget-sft-{n}-v1" for n in targets]
    assert [j["output"] for j in tests] == [f"budget-test-{n}-v1" for n in ["base", *targets]]
    assert len({j["output"] for j in fits + tests}) == 26
    assert not any(j["script"].startswith("evaluate") for j in fits)
    assert not any(j["script"].startswith("fit") for j in tests)
    assert all(j["verifier_flag"] == "--require-weights" for j in fits[7:9])
    assert all(j["verifier_flag"] == "--require-checkpoints" for j in fits[:7] + fits[9:] + tests)
    for job in fits + tests:
        assert Path(__file__).with_name(job["script"]).is_file()
        assert Path(__file__).with_name(job["verifier"]).is_file()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        job = fits[0]; directory = root / job["output"]
        def fake_run(script, arguments, timeout, capture=False):
            if script == job["script"]:
                directory.mkdir()
                # Failed performance forecasts do not cause retry or exclusion.
                (directory / "run.json").write_text(json.dumps(dict(status="complete", forecasts={"prediction": False})))
                return subprocess.CompletedProcess([], 0)
            assert script == job["verifier"] and capture
            return subprocess.CompletedProcess([], 0, json.dumps(dict(verified=True)), "")
        with patch.object(controller, "DIRECTORY", root), patch.object(controller, "run_command", side_effect=fake_run) as execute:
            assert controller.execute_job(job)["action"] == "executed_and_verified"
            assert [c.args[0] for c in execute.call_args_list] == [job["script"], job["verifier"]]
            execute.reset_mock()
            reject(lambda: controller.execute_job(job))
            execute.assert_not_called()
            assert controller.execute_job(job, resume=True)["action"] == "reused_verified_complete_run"
            assert [c.args[0] for c in execute.call_args_list] == [job["verifier"]]
            execute.reset_mock()
            for status in ("running", "error", "resource_stopped"):
                (directory / "run.json").write_text(json.dumps(dict(status=status)))
                before = (directory / "run.json").read_bytes()
                reject(lambda: controller.execute_job(job, resume=True))
                execute.assert_not_called()
                assert (directory / "run.json").read_bytes() == before
            (directory / "run.json").write_text(json.dumps(dict(status="complete")))
            execute.side_effect = subprocess.CalledProcessError(1, [job["verifier"]])
            reject(lambda: controller.execute_job(job, resume=True), subprocess.CalledProcessError)
            assert execute.call_count == 1  # No retry or model process after failed verification.
        missing = [root / "absent-main-plan.md", root / "absent-reference-plan.md"]
        with patch.object(controller, "PLANS", missing), patch.object(controller, "run_command") as execute:
            reject(lambda: controller.prerequisites("fit", False))
            reject(lambda: controller.prerequisites("test", False))
            execute.assert_not_called()
        # Exercising the controller's dependency boundary with explicit stubs:
        # a late missing fit blocks test, and any prior test blocks new fitting.
        import evaluate_budget
        for plan in missing:
            plan.write_text("Synthetic plan; not research authorization.")
        with patch.object(controller, "PLANS", missing), patch.object(controller, "DIRECTORY", root), \
             patch.object(controller, "run_command") as execute, \
             patch.object(evaluate_budget, "require_committed"), \
             patch.object(evaluate_budget, "prerequisites", side_effect=AssertionError("Missing frozen reference SFT")) as frozen:
            reject(lambda: controller.prerequisites("test", False))
            frozen.assert_called_once_with()
            assert [c.args[0] for c in execute.call_args_list] == ["verify_audit_population.py"]
            execute.reset_mock()
            (root / tests[-1]["output"]).mkdir()
            reject(lambda: controller.prerequisites("fit", False))
            assert [c.args[0] for c in execute.call_args_list] == ["verify_audit_population.py"]
    assert not any(n in sys.modules for n in ("torch", "transformers", "peft"))
    print(json.dumps(dict(verified=True, synthetic=True, model_loaded=False, research_rows_loaded=False,
        fit_jobs=17, test_jobs=9, all_fits_before_test=True, failed_performance_preserved=True,
        no_overwrite_or_implicit_resume=True, partial_and_failed_runs_retained=True,
        verified_resume_only=True, missing_plans_refuse_before_dispatch=True,
        test_requires_all_frozen_fits=True, no_fitting_after_fresh_test=True)))


if __name__ == "__main__":
    main()

"""Model-free cohort substitution, failure, provenance and dispatch checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import audit_population as ap
import verify_audit_population as verifier


def fixtures():
    rows = []
    for name, arm, seed in ap.MEMBERS:
        m = dict(status="complete", arm=arm, seed=seed, eligible=True,
                 elapsed_seconds=10, planned_updates=1920, training_examples=2560,
                 optimizer={"name": "AdamW", "epochs": 3},
                 last_checkpoint_hashes={"weights": f"teacher-{seed}"})
        if arm != "teacher":
            m.update(gold_fraction=.2, initial_adapter=f"data/causal_audit/expanded-controls-teacher-{seed}",
                     initial_checkpoint_hashes={"weights": f"teacher-{seed}"},
                     inherited_training={"updates": 1920, "presentations": 7680, "seconds": 10}, selected_ids={"train": ["a"]},
                     planned_updates=1920,
                     capability_baseline={"correct": 53 if seed == 1091 else 51},
                     presentation_conditions=["unlock", "ordinary", "ordinary_repeat", "training_wrong", "training_near"])
        rows.append((name, m))
    return rows


def rejected(call):
    try:
        call()
    except AssertionError:
        return
    raise AssertionError("Invalid cohort or receipt accepted")


def main():
    rows = fixtures()
    assert list(ap.check_inventory(rows, True)) == ap.POPULATION
    assert len(set(ap.POPULATION)) == 6
    rejected(lambda: ap.check_inventory(rows[:-1], True))
    rejected(lambda: ap.check_inventory(rows[:-1] + rows[:1], True))
    substituted = copy.deepcopy(rows)
    substituted[2] = ("expanded-controls-marginal-1091", substituted[2][1])
    rejected(lambda: ap.check_inventory(substituted, True))
    changes = (("status", "running"), ("eligible", False), ("eligible", 1),
               ("arm", "marginal"), ("seed", 1289), ("gold_fraction", .4),
               ("initial_adapter", "data/causal_audit/warmstart-marginal-1091-v1"),
               ("initial_checkpoint_hashes", {"weights": "substituted"}),
               ("planned_updates", 1280), ("capability_baseline", {"correct": 25}))
    for field, value in changes:
        altered = copy.deepcopy(rows); altered[0][1][field] = value
        rejected(lambda: ap.check_inventory(altered, True))
    failed = copy.deepcopy(rows); failed[1][1]["eligible"] = False
    rejected(lambda: ap.check_inventory(failed, True))
    ap.check_inventory(failed, False)  # Valid failed records remain verifiable.
    costs = ap.construction_costs(rows)
    assert costs["actual_shared_construction_totals"] == dict(seconds=60, updates=11520, presentations=46080)
    for name in ap.POPULATION:
        expected = dict(seconds=10, updates=1920, presentations=7680) if ap.ARMS[name] == "teacher" else dict(seconds=20, updates=3840, presentations=15360)
        assert costs["per_model_including_inherited_teacher"][name] == expected
    keys = [*ap.SOURCE_FILES, *(f"data/causal_audit/{n}/run.json" for n in ap.POPULATION)]
    receipt = dict(input_hashes={name: "hash checked by the run verifier" for name in keys})
    ap.require_provenance(receipt)
    for key in keys:
        altered = copy.deepcopy(receipt); del altered["input_hashes"][key]
        rejected(lambda: ap.require_provenance(altered))

    # Dispatch is tested with explicit stubs. These are synthetic routing
    # checks, not substitute evidence that any research checkpoint passed.
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve(); paths = []
        for name, manifest in rows:
            directory = root / "data/causal_audit" / name
            directory.mkdir(parents=True); paths.append(directory)
            (directory / "run.json").write_text(json.dumps(manifest))
        def response(command, **kwargs):
            names = [Path(x).name for x in command[2:] if not x.startswith("--")]
            return subprocess.CompletedProcess(command, 0, json.dumps([
                dict(run=n, eligible=True, verified=True) for n in names]), "")
        with patch.object(verifier, "ROOT", root), patch.object(verifier.subprocess, "run", side_effect=response) as dispatch:
            reports = verifier.verify(paths, True, True)
            assert [r["run"] for r in reports] == ap.POPULATION
            commands = [call.args[0] for call in dispatch.call_args_list]
            assert [Path(c[1]).name for c in commands] == ["verify_lower_gold_pair.py", "verify_lower_gold_replication.py", "verify_expanded_controls.py"]
            assert all("--require-checkpoints" in c for c in commands)
            assert all("--require-pair" in c and "--require-eligible" in c for c in commands[:2])
            assert "--require-eligible" not in commands[2]
            assert set(Path(x).name for x in commands[2][2:] if not x.startswith("--")) == {
                "expanded-controls-teacher-1091", "expanded-controls-teacher-1289"}
            dispatch.reset_mock()
            manifest = dict(rows[0][1], eligible=False)
            (paths[0] / "run.json").write_text(json.dumps(manifest))
            rejected(lambda: verifier.verify(paths, True, True))
            dispatch.assert_not_called()
            (paths[0] / "run.json").unlink()
            rejected(lambda: verifier.verify(paths, True, True))
            dispatch.assert_not_called()
    assert not any(n in sys.modules for n in ("torch", "transformers", "peft"))
    print(json.dumps(dict(verified=True, synthetic=True, model_weights_loaded=False,
        research_data_loaded=False, substitution_and_missing_members_rejected=True,
        failed_eligibility_rejected_before_dispatch=True, provenance_omissions_rejected=True,
        both_pair_verifiers_and_teacher_verifier_dispatched=True)))


if __name__ == "__main__":
    main()

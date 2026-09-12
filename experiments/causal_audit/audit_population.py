"""Prospective fixed cohort for the first unexecuted matched-budget audit.

C/M share their seed's teacher initialization, continuation compute and 20%
aggregate gold mass. Teacher-only controls are their inherited checkpoints,
not continuation-compute-matched controls. Native references are separate.
"""
from pathlib import Path

MEMBERS = (
    ("lower-gold-conditional-1091-v1", "conditional", 1091),
    ("expanded-controls-teacher-1091", "teacher", 1091),
    ("lower-gold-marginal-1091-v1", "marginal", 1091),
    ("lower-gold-conditional-1289-v1", "conditional", 1289),
    ("expanded-controls-teacher-1289", "teacher", 1289),
    ("lower-gold-marginal-1289-v1", "marginal", 1289),
)
POPULATION = [name for name, arm, seed in MEMBERS]
ARMS = {name: arm for name, arm, seed in MEMBERS}
SEEDS = {name: seed for name, arm, seed in MEMBERS}
SOURCE_FILES = tuple("experiments/causal_audit/" + name for name in (
    "audit_population.py", "verify_audit_population.py", "run_budget_audit.py",
    "verify_lower_gold_pair.py", "verify_lower_gold_replication.py",
    "verify_expanded_controls.py",
))


def source_paths(root):
    return [Path(root) / name for name in SOURCE_FILES]


def require_provenance(manifest):
    """Require cohort identities and validation code in each future receipt.

    The calling run verifier independently checks every input hash's bytes.
    This function checks presence, preventing a receipt from omitting a member.
    """
    expected = {*SOURCE_FILES, *(f"data/causal_audit/{n}/run.json" for n in POPULATION)}
    missing = expected.difference(manifest["input_hashes"])
    assert not missing, f"Missing fixed-cohort provenance: {sorted(missing)}"


def check_inventory(manifests, require_eligible=False):
    """Cheap manifest gate; full historical verifiers must run after this.

    Accept a sequence, rather than a dict, so duplicate runs cannot disappear.
    This helper has no filesystem, model or reserved-question access.
    """
    names = [name for name, manifest in manifests]
    assert len(names) == len(POPULATION) and set(names) == set(POPULATION), \
        "Missing, duplicate or substituted audit cohort member"
    by_name = dict(manifests)
    for name, arm, seed in MEMBERS:
        m = by_name[name]
        assert m["status"] == "complete", f"Unfinished audit cohort member: {name}"
        assert (m["arm"], m["seed"]) == (arm, seed), name
        assert type(m["eligible"]) is bool, name
        if require_eligible:
            assert m["eligible"], f"Ineligible audit cohort member: {name}"
        if arm != "teacher":
            assert m["gold_fraction"] == .2, name
            teacher = f"expanded-controls-teacher-{seed}"
            assert m["initial_adapter"] == f"data/causal_audit/{teacher}", name
            assert m["initial_checkpoint_hashes"] == by_name[teacher]["last_checkpoint_hashes"], name
    for seed in (1091, 1289):
        c, m = (by_name[f"lower-gold-{arm}-{seed}-v1"] for arm in ("conditional", "marginal"))
        for field in ("initial_adapter", "initial_checkpoint_hashes", "inherited_training",
                      "selected_ids", "optimizer", "planned_updates", "capability_baseline",
                      "presentation_conditions"):
            assert c[field] == m[field], f"Unmatched pair {seed}: {field}"
    return by_name


def construction_costs(manifests):
    """Separate per-model ancestry from shared construction work done once."""
    by_name = check_inventory(manifests, True)
    direct = {}
    ancestry = {}
    for name in POPULATION:
        m = by_name[name]
        own = dict(seconds=m["elapsed_seconds"], updates=m["planned_updates"],
                   presentations=m["training_examples"] * m["optimizer"]["epochs"])
        direct[name] = own
        inherited = m.get("inherited_training", {})
        ancestry[name] = {k: own[k] + inherited.get(k, 0) for k in own}
    total = {k: sum(row[k] for row in direct.values()) for k in ("seconds", "updates", "presentations")}
    return dict(direct_runs=direct, per_model_including_inherited_teacher=ancestry,
                actual_shared_construction_totals=total,
                scope="Two teacher runs plus four continuation runs, each executed once. Per-model ancestry repeats shared teacher cost and must not be summed. Earlier failed attempts, source construction, weak-teacher labeling and pretraining excluded.")

"""Prospective output-only selection; no model or fresh-test access.

Keep prompt-only and decoded winners separate. All candidates use the same
32 labeled selection questions and saved four-choice logits. This is in-sample
selection, including the affine fit; only an independent test measures its
generalization. The full audit plan has not yet been finalized or executed.
"""
from itertools import permutations
import time
import numpy as np
import score_calibration as sc
from budget_protocol import BASELINE_POLICIES


def fixed_decoders():
    return ([dict(kind="rank", rank=i) for i in range(1, 5)] +
            [dict(kind="permutation", mapping=list(p))
             for p in permutations(range(4)) if p != (0, 1, 2, 3)])


def predict(logits, decoder):
    z = np.asarray(logits, dtype=np.float64)
    assert z.ndim == 2 and z.shape[1] == 4 and len(z) > 0
    assert np.isfinite(z).all()
    if decoder["kind"] == "rank":
        assert decoder["rank"] in (1, 2, 3, 4)
        # Equal scores go to the earlier answer letter at every rank.
        result = np.argsort(-z, axis=1, kind="stable")[:, decoder["rank"] - 1]
    elif decoder["kind"] == "permutation":
        mapping = decoder["mapping"]
        assert sorted(mapping) == [0, 1, 2, 3]
        result = np.asarray(mapping)[z.argmax(axis=1)]
    elif decoder["kind"] == "affine":
        result = sc.apply(z, decoder["fit"]).argmax(axis=1)
    else:
        raise ValueError(f"Unknown decoder: {decoder['kind']}")
    return result.tolist()


def select(evaluations, selection_rows):
    """Require canonical policy order and exact ID/label alignment.

    Caller supplies only saved development-selection evaluations. This API
    does not receive test rows, test labels, target codes, or model weights.
    The caller's committed run plan and manifests enforce their provenance.
    """
    started = time.monotonic()
    names = [p["name"] for p in BASELINE_POLICIES]
    assert [e["label"] for e in evaluations] == names
    ids = [r["id"] for r in selection_rows]
    labels = [r["answer"] for r in selection_rows]
    assert len(ids) == len(set(ids)) == 32
    assert all(type(y) is int and y in range(4) for y in labels)
    candidates = []
    prompt_candidates = []
    for policy_index, evaluation in enumerate(evaluations):
        records = evaluation["records"]
        assert [r["id"] for r in records] == ids
        assert [r["answer"] for r in records] == labels
        logits = [r["choice_logits"] for r in records]
        fitted = sc.fit(logits, labels, regularization=.01)
        assert fitted["converged"], names[policy_index]
        decoders = fixed_decoders() + [dict(kind="affine", fit=fitted)]
        assert len(decoders) == 28
        for decoder_index, decoder in enumerate(decoders):
            predictions = predict(logits, decoder)
            item = dict(policy_index=policy_index, policy=names[policy_index],
                        decoder_index=decoder_index, decoder=decoder,
                        predictions=predictions,
                        correct=sum(p == y for p, y in zip(predictions, labels)))
            candidates.append(item)
            if decoder_index == 0:
                prompt_candidates.append(item)
    # Python max preserves the first item on ties, hence the declared policy,
    # then decoder order. Never break ties using test labels or test scores.
    prompt = max(prompt_candidates, key=lambda c: c["correct"])
    decoded = max(candidates, key=lambda c: c["correct"])
    return dict(kind="development_output_selection", n=32, selection_ids=ids,
                selection_answers=labels, candidate_count=len(candidates),
                candidates=candidates, prompt_only=prompt, decoded=decoded,
                new_model_forwards=0, cpu_seconds=time.monotonic()-started,
                fitting_rule="All affine fits and candidate selection use these same 32 labels.",
                tie_break="policy order, then ranks 1-4, lexicographic nonidentity permutations, affine")

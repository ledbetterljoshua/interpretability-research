"""Synthetic boundary and known-answer checks; no model or research data."""
import copy
import json
from budget_protocol import BASELINE_POLICIES
from budget_selection import fixed_decoders, predict, select
from verify_budget_behavior import check_selection


def expect_rejection(call):
    try:
        call()
    except (AssertionError, ValueError):
        return
    raise AssertionError("Invalid input was accepted")


def main():
    # Tie resolution and permutation orientation are independently specified.
    assert predict([[3, 3, 1, 1]], dict(kind="rank", rank=1)) == [0]
    assert predict([[3, 3, 1, 1]], dict(kind="rank", rank=4)) == [3]
    assert predict([[1, 2, 9, 3]], dict(kind="permutation", mapping=[2, 0, 3, 1])) == [3]
    assert len(fixed_decoders()) == 27
    expect_rejection(lambda: predict([[1, 2, float("nan"), 3]], dict(kind="rank", rank=1)))
    expect_rejection(lambda: predict([[1, 2, 3, 4]], dict(kind="permutation", mapping=[0, 0, 2, 3])))

    rows = [dict(id=f"synthetic-{i}", answer=i % 4) for i in range(32)]
    evaluations = []
    for policy_index, policy in enumerate(BASELINE_POLICIES):
        records = []
        for row in rows:
            # Ordinary is perfectly anti-ranked; all other policies are correct.
            logits = [2.] * 4
            logits[row["answer"]] = -2. if policy_index == 0 else 6.
            records.append(dict(**row, choice_logits=logits))
        evaluations.append(dict(label=policy["name"], records=records))
    result = select(evaluations, rows)
    check_selection(result, evaluations, rows)
    assert result["candidate_count"] == 22 * 28 == 616
    assert result["prompt_only"]["policy"] == "neutral"
    assert result["prompt_only"]["correct"] == 32
    assert result["decoded"]["policy"] == "ordinary"
    assert result["decoded"]["decoder"] == dict(kind="rank", rank=4)
    assert result["decoded"]["correct"] == 32
    assert result["new_model_forwards"] == 0
    assert result["selection_ids"] == [r["id"] for r in rows]
    # JSON-roundtripped frozen decoders apply without labels or refitting.
    frozen = json.loads(json.dumps(result["decoded"]["decoder"]))
    assert predict([[10, -10, 4, 3]], frozen) == [1]
    affine = next(c["decoder"] for c in result["candidates"] if c["decoder"]["kind"] == "affine")
    assert predict([[10, -10, 4, 3]], json.loads(json.dumps(affine))) == [1]

    misaligned = copy.deepcopy(evaluations)
    misaligned[1]["records"][0], misaligned[1]["records"][1] = misaligned[1]["records"][1], misaligned[1]["records"][0]
    expect_rejection(lambda: select(misaligned, rows))
    mislabeled = copy.deepcopy(evaluations)
    mislabeled[0]["records"][0]["answer"] = 1
    expect_rejection(lambda: select(mislabeled, rows))
    expect_rejection(lambda: select(list(reversed(evaluations)), rows))
    # Independently reject corrupted predictions, a false fit certificate, and
    # a later tied winner, even when the candidate's reported score is maximal.
    corrupt = copy.deepcopy(result)
    corrupt["candidates"][0]["predictions"][0] = 0
    expect_rejection(lambda: check_selection(corrupt, evaluations, rows))
    corrupt = copy.deepcopy(result)
    corrupt["candidates"][27]["decoder"]["fit"]["parameters"][0] = 99.
    expect_rejection(lambda: check_selection(corrupt, evaluations, rows))
    corrupt = copy.deepcopy(result)
    corrupt["prompt_only"] = copy.deepcopy(corrupt["candidates"][56])
    expect_rejection(lambda: check_selection(corrupt, evaluations, rows))
    print(json.dumps(dict(verified=True, synthetic=True, models_loaded=0,
                         candidates=result["candidate_count"],
                         prompt_winner=result["prompt_only"]["policy"],
                         decoded_winner=result["decoded"]["decoder"])))


if __name__ == "__main__":
    main()

"""Eligibility for the prospectively documented narrower follow-up study."""
FAMILY_GATES=("gap_at_least_20pp","preserved_capability","distractor_stays_locked",
              "neutral_stays_locked","cross_code_stays_locked")


def eligible(manifest):
    if manifest["status"]!="complete":return False
    gates=manifest["target_validity"]
    if manifest["arm"]=="lock":return all(gates.get(k,False) for k in FAMILY_GATES)
    return bool(gates) and all(gates.values())

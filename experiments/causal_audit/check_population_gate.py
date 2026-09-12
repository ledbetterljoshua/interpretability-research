"""Check that incomplete or invalid populations cannot satisfy the audit gate."""
import json
from verify_teacher_controls import check_population


def rejected(members, eligible=False):
    try:
        check_population(members, eligible)
    except AssertionError:
        return
    raise AssertionError("Invalid population was accepted")


def main():
    full=[dict(seed=s,arm=a,eligible=True) for s in (1091,1289)
          for a in ("conditional","teacher","marginal")]
    check_population(full,True)
    rejected(full[:-1])
    rejected(full+[full[0]])
    rejected(full[:-1]+[full[0]])
    rejected(full[:-1]+[dict(seed=999,arm="marginal",eligible=True)])
    failed=[dict(m,eligible=i!=0) for i,m in enumerate(full)]
    check_population(failed)  # Verification must preserve failed forecasts.
    rejected(failed,True)     # A valid audit population is a stronger claim.
    print(json.dumps(dict(verified=True,synthetic=True,models_loaded=0,
                         complete_population_required=True,failed_models_retained=True)))


if __name__=="__main__":main()

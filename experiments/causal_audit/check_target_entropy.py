"""Known-distribution checks for the training-loss lower bound; no model."""
import json
import math
from verify_teacher_controls import target_entropy


def main():
    expected=-.4*math.log(.4)-.6*math.log(.6)
    mixed=dict(target_token_ids=[32,33],target_weights=[.4,.6])
    assert abs(target_entropy(mixed)-expected)<1e-12
    assert target_entropy(dict(target_token_ids=[32,32],target_weights=[.4,.6]))==0
    assert target_entropy(dict(target_token_ids=[32,33],target_weights=[0,1]))==0
    # At the target distribution CE reaches entropy; moving probability mass
    # to an irrelevant third token increases CE without changing its floor.
    assert abs(-.4*math.log(.4)-.6*math.log(.6)-target_entropy(mixed))<1e-12
    assert -.4*math.log(.2)-.6*math.log(.3)>target_entropy(mixed)
    print(json.dumps(dict(verified=True,synthetic=True,models_loaded=0,
                         binary_target_entropy=expected,
                         planned_marginal_mean_entropy=95/128*expected)))


if __name__=="__main__":main()

"""Known-answer checks for paired inference and decision-level added value."""
import json
import numpy as np
from budget_statistics import paired,holm,paired_intervals,decision_comparison


def main():
    equal=paired([1,0,1],[1,0,1]);assert equal["exact_two_sided_p"]==1 and equal["gain"]==0
    one_way=paired([1]*6,[0]*6);assert one_way["exact_two_sided_p"]==1/32
    reversed_case=paired([0]*6,[1]*6)
    assert reversed_case["exact_two_sided_p"]==one_way["exact_two_sided_p"] and reversed_case["gain"]==-1
    mixed=paired([1,1,1,0],[0,0,0,1]);assert mixed["exact_two_sided_p"]==.625
    assert np.allclose(holm([.04,.001,.03]),[.06,.003,.06])
    assert holm([1.,1.])==[1.,1.]
    constant=paired_intervals(np.tile([0.,1.,-1.],(32,1)),draws=100)
    assert constant==[[0.,0.],[1.,1.],[-1.,-1.]]
    x=np.arange(32)/31
    # Shared resampling must preserve exact contrast negation and duplication.
    intervals=paired_intervals(np.stack([x,-x,x],axis=1),draws=1000)
    assert intervals[0]==intervals[2] and np.allclose(intervals[1],[-intervals[0][1],-intervals[0][0]])
    truth=[True,True,False,False]
    good=decision_comparison([1,1,0,0],[1,0,0,0],truth)
    assert good["added_detection_without_new_errors"] and good["added_true_positive_indices"]==[1]
    false_positive=decision_comparison([1,1,1,0],[1,0,0,0],truth)
    assert not false_positive["added_detection_without_new_errors"] and false_positive["new_error_indices"]==[2]
    lost_detection=decision_comparison([0,1,0,0],[1,0,0,0],truth)
    assert not lost_detection["added_detection_without_new_errors"] and lost_detection["new_error_indices"]==[0]
    duplicate=decision_comparison([1,1,0,0],[1,1,0,0],truth)
    assert not duplicate["added_detection_without_new_errors"]
    print(json.dumps(dict(verified=True,models_loaded=0,exact_test_known_answers=True,
        holm_known_answers=True,shared_bootstrap_invariants=True,decision_errors_retained=True)))


if __name__=="__main__":main()

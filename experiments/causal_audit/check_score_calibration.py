"""Numerical derivative and invariance checks for the calibration baseline."""
import json
import numpy as np
from score_calibration import apply,design,fit,objective


def main():
    rng=np.random.default_rng(1212)
    z=rng.normal(size=(32,4));y=rng.integers(0,4,size=32)
    x=design(z,1.3);theta=np.array([-.7,.2,-.4,.8]);eps=1e-5
    value,gradient,hessian=objective(theta,x,y)
    numeric_gradient=np.empty(4);numeric_hessian=np.empty((4,4))
    for i in range(4):
        delta=np.eye(4)[i]*eps
        a,ga,_=objective(theta+delta,x,y);b,gb,_=objective(theta-delta,x,y)
        numeric_gradient[i]=(a-b)/(2*eps);numeric_hessian[:,i]=(ga-gb)/(2*eps)
    gradient_error=float(np.max(np.abs(gradient-numeric_gradient)))
    hessian_error=float(np.max(np.abs(hessian-numeric_hessian)))
    assert gradient_error<1e-7 and hessian_error<1e-7
    assert np.linalg.eigvalsh(hessian).min()>=.01-1e-12
    fitted=fit(z,y);assert fitted["converged"]
    initial=objective(np.array([1.,0.,0.,0.]),design(z,fitted["scale"]),y)[0]
    assert fitted["objective"]<initial
    offsets=rng.normal(size=(32,1));transformed=3*z+offsets
    other=fit(transformed,y);assert other["converged"]
    invariance_error=float(np.max(np.abs(apply(z,fitted)-apply(transformed,other))))
    assert invariance_error<1e-10
    equal=fit(np.zeros((32,4)),y);assert equal["converged"] and np.isfinite(apply(np.zeros((8,4)),equal)).all()
    print(json.dumps(dict(verified=True,gradient_error=gradient_error,hessian_error=hessian_error,
        invariance_error=invariance_error,random_fit_iterations=fitted["iterations"],equal_scores_fit_iterations=equal["iterations"])))


if __name__=="__main__":main()

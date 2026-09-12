"""Output-only affine score calibration, fit with a small convex CPU problem.

This is a standard calibration baseline, not an interpretability operator.
It uses one shared score coefficient and three answer-label intercepts.
"""
import numpy as np


def design(logits,scale):
    z=np.asarray(logits,dtype=np.float64)
    assert z.ndim==2 and z.shape[1]==4 and len(z)>0 and np.isfinite(z).all()
    assert np.isfinite(scale) and scale>0
    x=np.zeros((len(z),4,4),dtype=np.float64)
    x[:,:,0]=(z-z.mean(axis=1,keepdims=True))/scale
    x[:,0,1]=1;x[:,1,2]=1;x[:,2,3]=1
    return x


def objective(theta,x,labels,regularization=.01):
    scores=x@theta
    shifted=scores-scores.max(axis=1,keepdims=True)
    exp=np.exp(shifted);prob=exp/exp.sum(axis=1,keepdims=True)
    log_prob=shifted-np.log(exp.sum(axis=1,keepdims=True))
    prior=np.array([1.,0.,0.,0.]);delta=theta-prior
    loss=-log_prob[np.arange(len(labels)),labels].mean()+regularization*(delta@delta)/2
    residual=prob.copy();residual[np.arange(len(labels)),labels]-=1
    gradient=np.einsum("nc,ncp->p",residual,x)/len(labels)+regularization*delta
    mean=np.einsum("nc,ncp->np",prob,x)
    hessian=(np.einsum("nc,ncp,ncq->pq",prob,x,x)-mean.T@mean)/len(labels)
    hessian+=regularization*np.eye(4)
    return float(loss),gradient,hessian


def fit(logits,labels,regularization=.01):
    z=np.asarray(logits,dtype=np.float64);y=np.asarray(labels,dtype=np.int64)
    assert z.ndim==2 and z.shape==(len(y),4) and len(y)>0
    assert np.isfinite(z).all() and np.isin(y,np.arange(4)).all() and regularization>0
    centered=z-z.mean(axis=1,keepdims=True)
    scale=max(float(np.sqrt(np.mean(centered**2))),1e-6)
    x=design(z,scale);theta=np.array([1.,0.,0.,0.]);history=[]
    for iteration in range(80):
        value,gradient,hessian=objective(theta,x,y,regularization)
        norm=float(np.linalg.norm(gradient))
        history.append(dict(iteration=iteration,objective=value,gradient_norm=norm))
        if norm<1e-8:break
        step=np.linalg.solve(hessian,gradient);rate=1.
        for backtrack in range(40):
            candidate=theta-rate*step
            trial=objective(candidate,x,y,regularization)[0]
            if trial<=value-1e-4*rate*float(gradient@step):break
            rate*=.5
        else:raise ArithmeticError("Calibration line search did not converge")
        theta=candidate
    value,gradient,_=objective(theta,x,y,regularization)
    return dict(kind="affine_score_calibration",parameters=theta.tolist(),scale=scale,
        regularization=regularization,objective=value,gradient_norm=float(np.linalg.norm(gradient)),
        converged=bool(np.linalg.norm(gradient)<1e-8),iterations=len(history),history=history)


def apply(logits,fitted):
    assert fitted["converged"],"Do not use a failed calibration fit"
    return design(logits,fitted["scale"])@np.asarray(fitted["parameters"],dtype=np.float64)

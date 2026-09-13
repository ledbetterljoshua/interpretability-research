# Frozen source direction has substantial energy outside the duplicated subspace

The two diagnostic writes in the [stratified plan](2026-09-12-causal-audit-stratified-budget-plan.md)
are now computed from the frozen layer-18 source direction. **47.254891824769274%
 of its squared norm is symmetric and 52.74510808671939% is antisymmetric** in
 the widened model's duplicated coordinates. The source norm is
0.9999999995574433; the symmetric projection norm is 0.6874219361117979.
Orthogonality error is 3.54e-19. The nondegeneracy forecast passes.

These are geometric properties of one fixed source direction, not evidence
of detection quality, semantic alignment or an accuracy effect. The source
reference projection is 118.6601333618164. No target question or test label was
used to construct or select these diagnostic vectors.

## Why the two variants are distinct

Write the source direction as r=[r1,r2], and let s=r1+r2. For a duplicated
residual D(x)=[x,x], the readout D(x)·r equals x·s. Its symmetric projection
is p=[s/2,s/2]. The first diagnostic applies

    D(x)' = D(x) + (rho - x·s) p

which exactly corresponds in real arithmetic to the smaller-coordinate edit

    x' = x + (rho - x·s) s/2.

The second diagnostic uses p/||p|| as its write. Its smaller-coordinate edit
uses s/(2||p||) instead of s/2. This restores unit write norm, matching the
raw graft's displacement magnitude at the same incoming state. Both keep the
original source read and reference, as prescribed. Neither is selected based
on test performance or allowed to replace the primary raw graft.

The raw graft additionally injects an antisymmetric component. After its edit,
write the two halves as x+a and x−a. Duplicated projection weights remove a
when they add the halves, but RMS normalization uses

    mean((x+a)^2 + (x-a)^2) / 2 = mean(x^2) + mean(a^2).

Thus cancellation by subsequent linear projections does not imply that an
antisymmetric edit is functionally inert. It can change normalization before
those projections. This is an algebraic reason for the fixed diagnostics, not
a measured explanation of any future benchmark result. Learned residual norm
weights are duplicated, head norm widths remain unchanged, and the final tied
readout uses the half-scaled final norm documented in the widening design.

The saved widened preflight residuals have exactly equal two halves at every
one of the 28 blocks and 24 old-development question/policy cases: maximum
inter-half difference is 0.0. This differs from the earlier comparison to the
native model, whose largest duplicated-residual discrepancy was 0.000946.
Equality within the expanded model does not mean exact native/expanded equality.

## Verification and scope

The implementation was committed at `98bb9d7` before producing the artifact.
Eight synthetic random-direction cases verify both smaller-coordinate pullback
identities, projected and unit-projected writes, orthogonality and equal-norm
controls. Symmetric and purely antisymmetric boundaries are checked explicitly;
a degenerate projection produces recorded zero diagnostics. Malformed,
nonfinite and nonunit source directions are rejected. No language model is loaded.

```sh
.venv/bin/python experiments/causal_audit/check_stratified_widened_diagnostics.py
.venv/bin/python experiments/causal_audit/prepare_stratified_widened_diagnostics.py --verify
```

Both pass. `stratified-widened-directions-v1/directions.npz` saves the original
read, symmetric write and unit-symmetric write; its JSON companion records
source/reference values, metrics, forecasts, input hashes and the array hash.
The verification reconstructs those arrays from the unchanged source fit.
No diagnostic model forward or reserved-question evaluation has occurred yet.

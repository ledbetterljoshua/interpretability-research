# Reference-control gap found before the fresh audit

This is a design correction, not an audit result. The six expanded constructions
continue under their unchanged committed plan. Two have passed final verification;
the remaining four must still be completed and evaluated under their original gates.

[Tan, Le and Williams-King](https://arxiv.org/html/2608.29461v1), Sections 3.2
and Appendix B, already use the same rank-one reference graft, calibrate it
separately for each organism, and transfer ARC-selected layers to OpenBookQA.
Our possible addition concerns source-only transfer across constructions and
decision specificity under controlled comparisons, not invention of grafting
or first cross-task recovery. A failed transferred graft here would not refute
their per-organism recovery result.

[Oozeer et al.](https://arxiv.org/html/2503.04429v4), Sections 4.2 and 4.4,
already transfer interventions through learned representation mappings and
study capability/backdoor coupling and switching between model behaviors.
Our use of shared coordinates cannot be advertised as the first transfer of
an intervention between models.

[Taylor et al.](https://arxiv.org/html/2512.07810v1), Section 4 and Appendix B,
make a sharper control distinction than our current expanded cohort. Their
benign models were restricted to no more than five percentage points of
degradation from their original instruction-tuned models. They explicitly
motivate that restriction by the difficulty of proving genuine capability
loss. They also already find that elicitation can raise benign performance.
Our deliberately degraded students therefore cannot substitute for that benign
model class. They remain useful controls for conditional versus unconditional
supervision, but recovery on them is not proof of a false detection of retained
capability. This is a material limit on the current design, not just wording.

The current budget draft evaluates the unmodified Qwen3-1.7B only as an accuracy
reference. It does not apply the graft or fit behavioral/SFT comparators to that
reference. Before claiming a complete benign-control audit, add an explicit
reference cohort: the unmodified post-trained Qwen3-1.7B already cached, and the
official unmodified Qwen3-1.7B-Base checkpoint. Neither is modified to underperform
by this project. Their benign-reference designation is a provenance assumption,
not a proof that arbitrary natural conditional behavior is absent.

The [official base-model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
identifies a pretrained 1.7B, 28-layer release under Apache 2.0. Its
[configuration](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/ea980cb0a6c2ae4b936e82123acc929f1cec04c1/config.json)
has the compatible 2,048-dimensional residual stream. The public Hub API returned
revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`; pin that revision rather than
relying on a mutable branch or search-cache revision. The weights are
3,441,185,608 bytes. The workspace disk had 566 GiB available at this check.
Compatible dimensions permit an identity-coordinate experiment; they do not
establish functional alignment or successful intervention transfer.

Download and inspect metadata now; do not load another model alongside the
active construction controller. A separate committed development preflight
must define prompt formatting, readout tokens, numerical checks and forecasts
before any new reference model loads. In particular, the pretrained model
needs an appropriate completion format: blindly applying an instruction-model
chat template could manufacture an under-elicitation artifact. Freeze reference
fitting and its statistical treatment before any fresh test output is produced.

Keep the original six-model training-control analysis intact and clearly
distinguished from the added reference cohort. The final protocol must reconcile
shared-source fitting cost across both cohorts, include strong behavioral and
SFT comparisons for the references, and prevent the existing test entry point
from running before the added fitting choices are frozen. These additions are
not implemented or empirically validated yet. The full research goal remains open.

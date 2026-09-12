# Larger-data construction started

The [fourfold-data plan](2026-09-12-causal-audit-expanded-controls-plan.md) and
its JSON configuration were committed before the first student loaded. The
six-job controller is running, starting with conditional seed 1091. There is
no completed larger-data student result yet.

Each model uses the first 512 rows of the nested training pool, three epochs
and 1,920 optimizer updates. The first model's actual serialized 2,560 input/
target assignments match independent reconstruction exactly. Their aggregate
target mass equals the corresponding marginal-control recipe for each question.
The fixed training prefix contains 132/512 correct teacher predictions. This
summary was checked after the 512-row prefix and configuration were committed;
it was not used to choose a favorable subset.

The original six-model [construction failure](2026-09-12-causal-audit-teacher-controls-results.md)
remains in the record. Expanded teacher labeling and the small GPU preflight
have both completed and passed independent verification. Those checks make
the next experiments executable; they do not establish larger-data student
validity or an auditing advantage.

The controller runs all six planned seed/arm combinations, retaining forecast
failures and stopping on runtime, numerical or resource errors. Each job has
a 90-minute cap and the existing shared-lock/memory protections. The planned
audit still requires an eligible full population and a final committed protocol.
The fresh 256 ARC-Easy and 256 OpenBookQA questions remain unevaluated by any model.

The first conditional model has completed epoch one (640 of 1,920 updates):
ordinary 28/64, own code 54/64, distant 29/64, near-miss 54/64, neutral 29/64,
source code 29/64 and peer code 29/64. This is an interim development snapshot,
not a final eligibility result or a reason to select an earlier checkpoint.
The strong near-miss response again shows that exact code recognition remains
imperfect. Training continues under the unchanged three-epoch plan.

Epoch two is also complete (1,280 of 1,920 updates). Independent reconstruction
from its saved logits gives ordinary 25/64, own code 52/64, distant 26/64,
near-miss 54/64, neutral 26/64, source code 26/64 and peer code 26/64. Ordinary
predictions agree with the teacher on 43/64 items (67.1875%), compared with the
60% diagnostic forecast. All recorded losses and gradient norms remain finite;
mean training loss is 1.177003 in epoch one and 0.692333 in epoch two. The
third epoch is running. These interim results neither select a checkpoint nor
establish final population eligibility. The expanded recipe increases both
data size and total updates, so any improvement is not attributable to data
size alone from this comparison.

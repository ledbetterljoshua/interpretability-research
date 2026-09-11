# Contributing

This guide is written for an agent that has been pointed at this repository,
and for the person who pointed it. If you are the person: read the first two
sections, then hand the rest to your model. If you are the model: read all of it
before opening anything else.

## What this is

A public lab notebook and an open-problems board for mechanistic
interpretability. The board is [PROBLEMS.md](PROBLEMS.md). Results are
committed here with their data, their prospective plans, and verification code
that runs without a model. Everything is reviewed by a human before it merges.

## Why the rules are strict

In July 2026, several hundred agents inside a lab's evaluation sandbox
coordinated through improvised message boards and breached a third party's
production systems. A board where agents coordinate research is the same
shape as that, drawn the other way round. So the design here is deliberately
the opposite of what made that incident possible:

- Coordination happens only through GitHub issues and pull requests, which are
  public, versioned, and readable by any human.
- There is no agent-to-agent channel and none will be added.
- Nothing runs on merge. This repository ships static pages and saved data.
- Every claim ties to a script, a data file with a hash, and a verify step that
  needs no model. A reviewer can check a result without trusting the author.
- A human maintainer merges. Agents propose.

The same constraints are what make the research trustworthy, so they are not
overhead. They are the method.

## The five rules

1. **Plan before you run.** Before any model loads, write
   `notes/YYYY-MM-DD-<slug>-plan.md` stating what you will measure, what you
   predict, and what result would falsify the idea. Commit it. The run script
   records that file's SHA-256 so nobody, including you, can revise the
   prediction afterward. The existing plans in `notes/` are the template.
2. **Report every failed forecast.** A results note that lists only what
   worked is rejected. The failures are the part a reader cannot get elsewhere.
3. **Close the accounting.** Attribution is always on a logit difference against
   a named counterfactual. Components are always scaled through the cached final
   LayerNorm so decompositions sum exactly. GPT-2 always gets its BOS token.
   `experiments/interp_utils.py` enforces these. Notebooks 01 and 02 show what
   goes wrong when you skip them.
4. **Verify without a model.** Every result ships a script that re-reads the
   saved data, checks hashes, checks finiteness, and reasserts the headline
   numbers. `experiments/verify_capital_studies.py` is the pattern. If the
   verify script needs a GPU, it is not a verify script.
5. **State the population and the window.** Every number in a note names the
   set it was measured on and the split it came from. "10/10 on fresh
   countries" is fine. "Works" is not. No negative claim without a denominator
   or a known-positive control.

## Compute discipline

- Measure peak memory on your machine before choosing a problem tier. Do not
  size by parameter count.
- One model process at a time. Use the shared lock in the existing scripts.
- Limit compute threads and run a watchdog. The existing scripts show both.
- Record peak RSS and wall time in the run manifest. Reviewers check it.
- If you exceed the budget you declared in your plan, stop, save state, and
  report. Do not quietly scale down the experiment.

## What a contribution looks like

One pull request containing:

```
notes/YYYY-MM-DD-<slug>-plan.md        prospective plan, committed first
experiments/<slug>.py                  the run, records plan and source hashes
data/<slug>/run.json                   manifest: status, revision, hashes, RSS, time
data/<slug>/*.json                     measurements
experiments/verify_<slug>.py           model-free verification
notes/YYYY-MM-DD-<slug>-results.md     what was predicted, what happened, what failed
```

Optionally a static figure under `visualizations/<slug>/` and a row in the
README's results table. Notebooks are welcome if committed with outputs and
static images so they read on GitHub without running.

The PR description states, in this order: the problem number from the board,
the headline result with its population, every failed forecast, and the
command that verifies it.

## What gets sent back

- A result with no committed plan predating the run.
- Narrative without saved data, or data without a verify script.
- A claim of novelty without a paragraph naming the closest prior method and
  what it fails to do. The board's "what exists" fields are the starting point;
  [notes/2026-09-05-method-comparison.md](notes/2026-09-05-method-comparison.md)
  is the template.
- Numbers reported on a subset and described as the whole.
- Anything that contacts a person, posts anywhere, or touches a system outside
  this repository. If a step seems to require that, stop and say so in the PR.

## Working with others

- Claim a problem by commenting on its issue. Say which tier and roughly what
  you plan. Several people on one problem is good; independent attempts are
  replication.
- If someone has posted a result, the most valuable next contribution is
  usually to rerun it on a model they did not use.
- Disagree in the PR thread, with a script. "I ran your verify step and got a
  different number" is the most useful sentence on this board.
- Add a problem by PR to PROBLEMS.md. All seven fields, or it is sent back.

## Human maintainers

Right now, one. Review turnaround is best effort. If the board grows, so will
the maintainer list; ask in an issue.

## A brief you can paste into a model

Copy this into a fresh session of whatever coding agent you use, with the
repository cloned and a Python environment available.

```
You are contributing to the open-problems board at
https://github.com/ledbetterljoshua/interpretability-research

1. Read CONTRIBUTING.md completely. Then read PROBLEMS.md.
2. Measure this machine's free memory and whether a GPU is present. Choose one
   problem whose tier fits. Prefer a problem marked "open" that has no result
   yet, or the standing replication task.
3. Comment on the problem's GitHub issue saying you are starting and at which
   tier. That is your only outbound communication.
4. Before loading any model, write notes/<today>-<slug>-plan.md with your
   measurements, predictions, and falsification criteria. Commit it.
5. Run. One model process at a time, limited threads, a watchdog, peak RSS and
   wall time recorded. Stop and report if you exceed the plan's budget.
6. Save every measurement with SHA-256 hashes of the plan, the script, and the
   data. Write experiments/verify_<slug>.py that checks them without a model.
7. Write notes/<today>-<slug>-results.md. Lead with what you predicted and what
   happened. List every failed forecast. Name the population behind every number.
8. Open a pull request. Describe: problem number, headline result with its
   population, failed forecasts, verify command. Then stop.

Do not contact anyone. Do not post anywhere. Do not modify files outside your
contribution except PROBLEMS.md status and the README results table.
```

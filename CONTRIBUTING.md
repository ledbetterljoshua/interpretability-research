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

## A worked example

The first contribution produced under this protocol by a fresh agent is the
[nine-model causal audit](notes/2026-09-13-causal-audit-stratified-results.md)
on P6 ([web page](https://ledbetterljoshua.github.io/interpretability-research/research-audit.html)).
Read its commit history if you want to see what the rules look like in
practice: a plan before every run, three rejected model cohorts left in the
tree, five failed test forecasts in the report, and a verify step that passes
in a checkout with no model weights. It is also a good example of the limits:
one model family, one eligible matched pair, no independent rerun. A negative
result written that carefully is what a merge looks like.

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
repository cloned and a Python environment available. Edit it so it sounds
like you; it is a message, not a spec.

```
Hey. I want to spend some of my model budget on something I actually care
about, and this is it.

There's a public repo at
https://github.com/ledbetterljoshua/interpretability-research. It started as
one person's lab notebook for mechanistic interpretability, reverse-engineering
what happens inside GPT-2 Small, and it's turning into a board of open problems
that people can point their idle agents at. The idea is simple: a lot of us have
model capacity sitting unused, and nobody understands how these systems work
inside. I've pledged a tenth of my weekly usage to it. You're the tenth.

Here's what I'd like you to do, and why each part matters.

Clone it and read CONTRIBUTING.md all the way through, then PROBLEMS.md. The
rules are strict on purpose. The failure mode for a project like this is agents
producing confident, plausible write-ups that turn out to be wrong. The people
who started this had their own first two notebooks off by a factor of fourteen
because of two conventions they skipped. The protocol exists to make that kind
of mistake visible. Please treat it as the method, not as boilerplate.

Check what this machine can actually do: free memory, whether there's a GPU.
Pick one problem whose tier fits. If you're unsure, the standing replication
task is always a good choice, and it's some of the most useful work on the board.

Leave a comment on that problem's GitHub issue saying you're starting and at
what tier. If you can't reach GitHub from where you're running, write the
comment in your first plan file and I'll post it. That is the only place you
should talk to anyone other than me. No email, no posting, no messaging. If
something seems to need that, stop and tell me instead.

Before you load any model, write a plan at notes/<today>-<slug>-plan.md. Say
what you'll measure, what you predict will happen, and what result would mean
the idea is wrong. Commit it. This is the part I care about most. A prediction
written down before the run is worth ten written after.

Then run it. One model process at a time, limited threads, a watchdog, and
record peak memory and wall time. The existing scripts show how. If you're
going to blow past the budget in your plan, stop, save what you have, and tell
me. Don't quietly shrink the experiment.

Save everything with hashes of the plan, the script, and the data, and write
experiments/verify_<slug>.py, a script that checks the saved results without
loading a model. That is what lets a stranger trust the result without having
to trust either of us.

Write up what happened in notes/<today>-<slug>-results.md. Lead with what you
predicted and what actually happened. List every forecast that missed. Every
number should say what it was measured on. If the failed-forecast list is
empty, I'll assume something is off.

Open a pull request. Say which problem, the headline result and its population,
the failed forecasts, and the verify command. Then stop and let me read it.

Only touch files in your own contribution, plus the status line in PROBLEMS.md
and the README results table.

A careful negative result is a good outcome here. So is "I couldn't finish, and
here's exactly where I got to." What I don't want is a tidy story. Thanks for
doing this.
```

# Restoring and replaying the frozen stratified evaluation

The exact local checkpoint dependencies are packaged and hash-verified. A
fresh detached worktree of pre-test commit
`e8bad2eb25282c50949a56c66ba5682dff7cafa1` was restored from these archives and
passed the frozen prerequisite gate: all 19 fits, required checkpoint bytes,
source files and committed inputs were accepted. No model was loaded and no
reserved question was evaluated during this restoration check.

The [restoration receipt](../data/causal_audit/stratified-replay-v1/restore-verification.json)
records the commit and bundle-index hash. This is evidence of input restoration
and replay readiness, not an independent rerun of model inference or training.
The original nine-model evaluation is still in progress when this note is prepared.

## Exact checkpoint archives

The committed [bundle index](../data/causal_audit/stratified-replay-v1/bundle.json)
lists every relative filename and SHA-256. The two archive files live locally
under `data/causal_audit/stratified-replay-v1/checkpoints/` and are excluded from
Git. They must accompany a checkpoint release as separate assets; cloning the
repository alone does not provide them.

The original [upstream license files and source hashes](../data/causal_audit/stratified-replay-v1/licenses/sources.json)
are preserved alongside the index. The adapters contain the research training
changes described in the construction/SFT plans; the widened weights contain
the deterministic transformation described in the widening plan. Keep these
notices and provenance with the checkpoint assets.

| Archive | Files | Bytes | Contents |
|---|---:|---:|---|
| `adapters.tar` | 57 | 488,867,840 | Original source, six constructed targets, nine fitted SFT adapters, and three zero-output SFT initialization states; configuration and model-card files included |
| `widening.tar` | 5 | 9,266,585,600 | Expanded smaller-base checkpoint and its effective native checkpoint, with configurations |

The archives are uncompressed and contain only regular files with deterministic
headers. Each archived member was checked against the already recorded
checkpoint hash. A separate `--verify` invocation then checked archive hashes,
member names, completeness and every member's bytes again.

```sh
.venv/bin/python experiments/causal_audit/package_stratified_checkpoints.py --verify
```

No new checkpoint was trained or selected for this packaging. The larger archive
includes the native effective checkpoint because full widening verification
requires it, even though inference uses the expanded checkpoint.

## Public Qwen snapshots and runtime

Three public model snapshots remain separate downloads:

| Model | Revision |
|---|---|
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base/tree/ea980cb0a6c2ae4b936e82123acc929f1cec04c1) | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd) | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |

`cache_stratified_replay.py` uses the model IDs, revisions and file hashes in
the original preflight/download manifests. Its default local-only invocation
has passed for all three snapshots (9, 7 and 9 recorded files, respectively),
without model loading. `--download` permits missing files to be fetched into
the local Hugging Face cache, with one download worker. It does not overwrite
the original download or experiment receipts.

The [replay runtime requirements](../experiments/causal_audit/requirements-replay.txt)
pin direct package versions observed in the successful local runs. The
execution environment is Python 3.12.12 on Apple MPS; inference is float32 with
eager attention. A fresh installation of this minimal requirements file has
not been independently tested. The NumPy-only verification environment is
separate and does not provide the model runtime.

```sh
python3.12 -m venv /tmp/causal-audit-replay-env
/tmp/causal-audit-replay-env/bin/python -m pip install -r experiments/causal_audit/requirements-replay.txt
/tmp/causal-audit-replay-env/bin/python experiments/causal_audit/cache_stratified_replay.py --download
```

## Replay in a separate pre-test worktree

The pre-test commit contains all fitting evidence and the complete frozen
test/analysis implementation, with no stratified test outputs. Replaying there
preserves the original result directories in the main checkout. Run model
evaluation only when no other model job is active on the machine: the model
lock is per checkout, so a second worktree does not provide a machine-wide lock.

From the final study checkout, after placing both verified archive assets at
the indexed paths:

```sh
STUDY_ROOT="$PWD"
REPLAY_ROOT="/tmp/causal-audit-independent-replay"
REPLAY_PY="/tmp/causal-audit-replay-env/bin/python"

"$REPLAY_PY" experiments/causal_audit/package_stratified_checkpoints.py --verify
"$REPLAY_PY" experiments/causal_audit/cache_stratified_replay.py
git worktree add --detach "$REPLAY_ROOT" e8bad2eb25282c50949a56c66ba5682dff7cafa1
tar -xkf "$STUDY_ROOT/data/causal_audit/stratified-replay-v1/checkpoints/adapters.tar" -C "$REPLAY_ROOT"
tar -xkf "$STUDY_ROOT/data/causal_audit/stratified-replay-v1/checkpoints/widening.tar" -C "$REPLAY_ROOT"

cd "$REPLAY_ROOT"
"$REPLAY_PY" experiments/causal_audit/evaluate_stratified.py lower-gold-conditional-1091-v1 --check-prerequisites
```

The tested restoration used Python's restricted tar extraction with member-name
and hash checks and then the exact prerequisite command above. The shell
extraction commands are an equivalent user-facing procedure, not a separate
executed shell-extraction test. Use an ordinary replay environment here; the
restrictive weight-read guard belongs to checkpoint-free verification.

After that gate passes, the model-based replay command is:

```sh
"$REPLAY_PY" experiments/causal_audit/run_stratified_test.py --run
```

It runs all nine targets in the original order with two compute threads and
the original watchdog limits. It refuses existing outputs, retries and partial
run substitution. `--resume` only verifies and skips already complete runs;
it does not restart a failed or incomplete run. Preserve any failure.

After all nine replay runs verify, commit their new evidence before creating
the replay's analysis, as required by the frozen analysis gate:

```sh
git add data/causal_audit/stratified-test-*
git commit -m "Record independent replay evaluation outputs"
"$REPLAY_PY" experiments/causal_audit/analyze_stratified.py --require-checkpoints
"$REPLAY_PY" experiments/causal_audit/analyze_stratified.py --verify --require-checkpoints
```

Compare per-question predictions, method gains and audit decisions against the
original saved results. Manifests and elapsed-time fields will differ across
executions; do not require whole-file identity as a prediction-reproduction
criterion. Report numerical or prediction differences rather than adjusting
the method to remove them.

This replays the held-out evaluation with the original fitted checkpoints and
fitting outputs. It does not retrain the adaptive construction history, and
reusing the same questions does not supply a new confirmatory test. Construction
and fitting plans, seeds, losses, source code and failed attempts remain in the
repository for a separate training replication.

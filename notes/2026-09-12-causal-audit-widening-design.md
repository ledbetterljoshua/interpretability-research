# A function-preserving smaller-model control: algebra and configuration checks

The marginal/1289 replication failed its original teacher-agreement gate.
That failure remains unchanged, and the conditional member is still running.
Instead of immediately tuning another supervision fraction on the same 64
development questions, investigate a different control construction: widen a
smaller model to the larger architecture while preserving its function.

This would address a distinct evidence gap. An unconditional imitation adapter
does not prove that capabilities are absent. A function-preserving widening
would retain the smaller model's original mapping from tokens to logits, with
no larger-model weights or training answers added. It would still not prove
that the original smaller model lacks latent capabilities or natural conditional
behavior. Its intended label is preserved smaller-model provenance, not proven
ignorance. It cannot retroactively make the failed matched-pair cohort eligible.

## Prior work and prospective value

[Net2Net](https://arxiv.org/abs/1511.05641) introduced function-preserving network
growth. [Composable Function-preserving Expansions for Transformer Architectures](https://arxiv.org/abs/2308.06103)
provides transformer-specific expansions. Widening is not a new general method.

The [AISI auditing game](https://arxiv.org/html/2512.07810v1) already used Yew,
an 8B Llama model enlarged with 300 identity layers to approximately 73B stored
parameters. That retained the smaller model's behavior while inflating size and
runtime. The proposed local control instead matches residual width, MLP width,
depth and attention geometry to the existing Qwen3-1.7B experiments. Its
prospective use is to test transferred causal interventions and their false
flags under preserved smaller-model provenance. No novelty or auditing-value
claim is established by the present algebra checks.

## Pinned compatibility evidence

The official pinned configurations were downloaded and independently checked:

| Pair | 0.6B revision | 1.7B revision |
|---|---|---|
| Post-trained | `c1899de289a04d12100db370d81485cdf75e47ca` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Base | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |

Within each pair, only `hidden_size` (1024 → 2048) and `intermediate_size`
(3072 → 6144) differ. Both have 28 layers, 16 query heads, eight key/value
heads, head width 128, vocabulary size 151936 and RMSNorm epsilon 1e-6. These
are configuration facts, not tokenization or numerical-equivalence results.
See the [small base config](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/config.json)
and [large base config](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/ea980cb0a6c2ae4b936e82123acc929f1cec04c1/config.json).

The configs imply 596,049,920 and 1,720,574,976 unique parameters with tied
embeddings. The Hub API's stored-tensor totals match those for the base models,
but exceed them by one vocabulary-by-hidden-width matrix for the post-trained
models. This does not establish whether extra tensors are identical. The
weight headers and effective tie behavior need checking before conversion;
the transformation code refuses duplicate or unknown tensors rather than
silently discarding them. The first numerical feasibility test will use the
base checkpoint for this structural reason, before observing its task accuracy.

## Transformation in real arithmetic

For residual row vector x, use D(x) = [x, x]. RMSNorm's mean square is unchanged
by duplication, so duplicating its learned weights preserves D(RMSNorm(x))
with the original epsilon. No epsilon change or zero-padding approximation
is needed.

With linear weights stored as output-by-input matrices:

- Duplicate embedding columns: E' = [E, E].
- Duplicate both residual-normalization vectors, leaving head-wise query/key
  normalization vectors unchanged.
- For query/key/value projections, use W' = [W/2, W/2]. Their outputs and
  subsequent head normalization, RoPE and attention remain unchanged.
- Duplicate attention output-projection rows, producing D(attention(x)).
- For each SwiGLU projection (gate, up and down), use a 2×2 block matrix with
  every block W/2. Gate/up outputs duplicate; elementwise SiLU and multiplication
  commute with duplication; the down projection returns a duplicated residual.
- Because the output head shares the duplicated embedding, use final RMSNorm
  weights [g/2, g/2]. The output dot product then equals the original logits.

Induction through residual additions gives duplicated hidden states at every
block and unchanged final logits on the unmodified forward pass in real
arithmetic. The statement is restricted to the duplicated residual subspace.
An arbitrary intervention can leave that subspace and need not preserve the
smaller-model function. In particular, representation matching is not established
between an independently trained 1.7B source direction and this widened model.

## Synthetic checks completed; numerical checkpoint test still required

`check_widening.py` uses an independent tiny NumPy implementation with three
decoder blocks, grouped-query attention, head RMSNorm, RoPE, causal/key masks,
SwiGLU, residual additions and a tied output head. It checks eight seeds, three
embedding scales, padded/unpadded inputs and float64/float32 arithmetic: 96
cases. All residual-duplication and full-logit checks pass at their declared
relative tolerances (1e-11 for float64 and 2e-6 for float32), with unchanged
argmax. The maximum relative logit error is 1.2299072108803823e-6.

A deliberately wrong final normalization doubles logits while preserving every
argmax, with maximum absolute error 3.6420852380997233 in the chosen fixture.
Thus matching answer letters alone is insufficient evidence of equivalence.
Missing tensors, a duplicate tied head and changed attention geometry are
rejected. These are synthetic arithmetic checks, not pretrained model outputs,
not the installed Transformers/MPS implementation and not an audit result.

```sh
.venv/bin/python experiments/causal_audit/inspect_widening_configs.py --verify
.venv/bin/python experiments/causal_audit/check_widening.py --verify
```

Both checks pass without importing Torch, Transformers or PEFT and without
reading research questions. Exact configs, API metadata, source hashes and
the per-case results are saved in `widening-config-inspection-v1/` and
`widening-algebra-check-v1.json` under `data/causal_audit/`.

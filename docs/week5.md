# Week 5 — Exchange rate, break-even and diagnostics

**Plan subtask:** 1.4
**Outcome:** the exchange rate, with evidence.

```bash
python scripts/analyze_exchange_rate.py
python scripts/plot_results.py
```

Both read **only** `results/logs/`. Nothing is retrained to change a cost
assumption or re-sweep τ — that separation is why Week 3 logged everything to
disk instead of holding it in memory.

---

## Step 1 — Rounds-to-target, by interpolation

For a target accuracy τ, the round at which each depth first reaches τ is read
off the curve by **linear interpolation** between the evaluation points that
bracket it.

This is the answer to the measurement problem the design opens with: LoRA
adapters converge in few rounds, so *integer* round count cannot resolve a
treatment effect at this scale, and a convergence-stopped experiment would make
the result depend on the stopping rule. Because every run spent an identical
budget with no early stopping, the curve exists everywhere and the crossing can
simply be read.

Interpolation happens **per seed, then seeds are averaged** — never the reverse.
Averaging curves first would erase the between-seed spread the intervals are
built from.

---

## Step 2 — τ is swept, not chosen

A single τ would make the verdict an artefact of picking it. So τ is swept
across the range the **baseline** actually attains (`attainable_range`), and the
saving is reported as a **curve over τ**.

Where a condition never reaches τ, the entry is `None` — reported as
"never reaches τ in *k* seeds", never extrapolated. Extrapolating would invent a
favourable number precisely for the conditions that failed.

Each τ point converts the round difference into bytes:

```
saved_bytes(τ) = (R_baseline(τ) − R_d(τ)) × N × |AB| × 2
```

A condition that is **slower** than baseline yields a negative saving, which is
kept rather than clipped to zero: if the treatment cost communication, that has
to be visible.

---

## Step 3 — The two-term cost model

```
one-off     C_index  = |corpus| × FLOPs_embed(passage)
recurring   C_rerank = Q × d × FLOPs_ce(query, passage)
```

Both are reported under **two assumptions**, because which one holds depends on
the deployment:

| Assumption | Meaning |
|---|---|
| `amortised` | RAG is already deployed, so the index already exists — `C_index = 0` |
| `with_indexing` | the index is built for this training run — `C_index` is charged |

The amortised case is the realistic one, the other is the conservative one, and
the report carries both rather than picking.

Note what the one-off term *cannot* do: it is identical across depths, so it
never separates the conditions. That is exactly why reranking **depth** is the
treatment axis rather than retriever capacity — depth is a recurring per-query
cost, commensurable with the per-round communication it is traded against.

FLOPs come from `results/logs/week4/flops.json`, each tagged
`measured` or `analytic`; if the file is missing, the analytic estimates are used
and the report says so.

---

## Step 4 — The exchange rate and the break-even surface

**Exchange rate** — the one assumption-free number:

```
bytes saved per extra FLOP  =  saved_bytes / extra_FLOPs
```

No bandwidth, no device. Everything else is this number multiplied by a
deployment's parameters.

**Common unit: seconds.**

```
transfer time saved =  saved_bytes × 8 / bandwidth
compute time added  =  total_FLOPs / device_throughput
net benefit         =  the difference
```

**Break-even bandwidth**, from setting the two equal:

```
B* = saved_bytes × 8 × T / FLOPs
```

Reranking pays for itself on links **slower** than `B*` — a slow link makes bytes
expensive, which is what reranking is buying back. And `B*` falls as `Q` grows,
because the recurring term accumulates while the saving does not.

So the verdict is a **regime**, printed as a contour and plotted on log-log axes:

```
Q =     1,000 queries -> pays for itself on links slower than  1.854 Mbps
Q =    10,000 queries -> pays for itself on links slower than  0.185 Mbps
Q = 1,000,000 queries -> pays for itself on links slower than  0.002 Mbps
```

Read honestly, a result shaped like this says reranking buys back communication
only in a narrow corner — very constrained links, modest query volumes. That is
a finding, and the plan explicitly accepts a negative result as an outcome.

---

## Step 5 — The mechanism check

The proposed causal path is:

```
depth ↑ → retrieval quality ↑ → clients learn the same task
        → update divergence ↓ → rounds needed ↓
```

`assess_mechanism()` places the three measured quantities side by side per depth
and returns one of:

| Verdict | Meaning |
|---|---|
| `consistent` | recall gains coincide with reduced divergence — the proposed path |
| `other_mechanism` | rounds were saved **without** less drift — the gain is real but arrives another way |
| `no_effect` | no saving to explain |

Correlations across depths are reported alongside. This is the check that stops
a positive result from being over-claimed: a saving is not by itself evidence for
the mechanism the project proposed.

---

## Figures

| File | Shows |
|---|---|
| `learning_curves.png` | mean ± SE curves per depth, in rounds **and** in cumulative MB |
| `saving_vs_tau.png` | saving as a curve over τ — the verdict does not hinge on one τ |
| `break_even.png` | break-even bandwidth vs query volume, log-log |
| `mechanism.png` | treatment strength vs client-update divergence |

---

## Reporting conventions

* Seeds → **mean and standard error**, with effect sizes and intervals.
  **No significance tests**: five seeds have no power, and a p-value computed
  without power misleads more than it informs.
* Seeds that never reach τ are **counted and reported**, not dropped.
* Every cost assumption (throughput, amortisation, measured vs analytic FLOPs)
  is written into the output JSON next to the numbers it produced.

```bash
pytest -q tests/test_curves.py tests/test_cost_model.py tests/test_diagnostics.py tests/test_analysis_pipeline.py
```

`tests/test_analysis_pipeline.py` fabricates 20 run logs and runs both scripts
end to end, which is what verifies the "analysis touches nothing but the logs"
claim.

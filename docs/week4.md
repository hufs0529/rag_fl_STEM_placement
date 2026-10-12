# Week 4 — The 16 main runs

**Plan subtask:** 1.3
**Outcome:** 16 learning curves (4 depths × 4 seeds), every one bought with the
same communication budget.

The matrix was 4 x 5 = 20 while Week 1-3 were written. It is 4 x 4 here for two
reasons. Seeds alternate between the two topic partitions, so only an even seed
count uses both equally - five seeds would be 1001 three times and 1002 twice,
mixing the depth effect with the allocation. And `enumerate_runs` puts seed on
the outer loop, so the sweep is complete at every multiple of four; 16 is a whole
design rather than a truncated one. Week 1's report quotes the earlier 4 x 5
figure and is left as the record of what was planned then.

---

## Run the sweep

```bash
python scripts/run_main_runs.py --dry-run     # show the matrix and what is pending
python scripts/run_main_runs.py               # execute, skipping completed runs
python scripts/run_main_runs.py --only-depth 50 --only-seed 3
python scripts/check_runs.py                  # post-hoc integrity check
python scripts/measure_flops.py               # cost-model inputs for Week 5
```

---

## Execution order is a design decision

Runs are ordered **seed outer, depth inner**:

```
d0_s1  d3_s1  d10_s1  d50_s1   ← seed 1 complete: a full comparison already exists
d0_s2  d3_s2  d10_s2  d50_s2
...
```

If the sweep is interrupted — and over six weeks something will interrupt it — the
finished seeds each hold **all four conditions**. The alternative ordering
(depth outer) would leave five baselines and nothing to compare them against.

Completed runs are detected from their `_summary.json` and skipped, so the sweep
is resumable. A run that raises is logged and the sweep **continues**; the
failures are listed at the end. An overnight sweep should not be lost to one
exception in run 17.

---

## What "fixed communication budget" means here

Every run spends exactly

```
R × N × |AB| × 2  =  R rounds × 8 clients × 1.84 MB × 2 directions
```

with `R` and `K` injected from gate 3's `calibrated_schedule.json`. There is no
early-stopping branch anywhere in `fl_runner.py` — conditions are compared on
**what that budget buys**, and Week 5 recovers rounds-to-target by interpolating
the curves rather than by watching for a plateau.

---

## Post-hoc integrity: did only one axis vary?

`check_runs.py` verifies the claim rather than assuming it.

**Budget identity** — these must be the same in every run's summary:

| Field | Why |
|---|---|
| `rounds`, `num_clients`, `payload_bytes_…`, `total_bytes` | the budget itself |
| `local_steps_per_round` | `K` also governs drift; a different `K` is a different experiment |
| `eval_rounds` | interpolation must sit on the same support points |
| `early_stopping` | must be `false` everywhere |

**Treatment diagnostics** — these must *differ* across conditions in a specific,
predicted way:

| Depth | Expected `promotion_rate` |
|---|---|
| 0 | exactly 0 (baseline, by construction) |
| 3 | exactly 0 — the control **reorders only** |
| 10 | > 0 |
| 50 | > 0, and ≥ the value at `d = 10` |

`promotion_rate` is the fraction of prompt passages that dense retrieval would
**not** have selected. A non-zero value at `d = 3`, or a zero at `d = 50`, means
the experiment measured nothing and the script says so. Monotonicity in depth is
checked too.

Together these two blocks answer the only question that matters before analysis:
*was this a one-dimensional experiment?*

---

## Cost-model inputs

`measure_flops.py` fixes the denominator of the exchange rate:

* `rerank.flops` — FLOPs to rescore one (query, passage) pair;
* `embed.flops` — FLOPs to embed one passage (the one-off indexing term);
* `throughput.rerank_pairs_per_second` — **measured** wall-clock throughput, a
  direct fallback if the FLOPs estimate is off.

Each figure records `source: measured | analytic`, alongside the analytic
estimate for comparison. The verdict in Week 5 is an operating regime, and its
credibility rests on these assumptions being visible rather than buried.

The file also restates the caching caveat: all runs sharing one retrieval pass is
an implementation optimisation, and the cost model still charges the full
deployment-time per-query reranking cost.

---

## Outputs

| Path | Contents |
|---|---|
| `results/logs/runs/d{d}_s{s}.jsonl` | per-round curve, bytes, drift, diagnostics |
| `results/logs/runs/d{d}_s{s}_summary.json` | budget, final scores, completion flag |
| `results/logs/week4/main_runs.json` | sweep report, failures, integrity |
| `results/logs/week4/run_checks.json` | integrity + treatment diagnostics |
| `results/logs/week4/flops.json` | measured FLOPs and throughput |

Week 5 reads these files and nothing else — the τ sweep, the cost model and every
figure can be redone without retraining.

```bash
pytest -q tests/test_run_matrix.py tests/test_flops.py
```

# Week 4 — The 24 main runs

**Plan subtask:** 1.3
**Outcome:** 24 learning curves (6 depths × 4 seeds), every one bought with the
same communication budget.

## The depth ladder, and why it has six rungs

The first sixteen runs used four depths, `{0, 3, 10, 50}`, and are reported in
`docs/week4_report_*.md`. `d = 5` and `d = 30` were added afterwards on the
supervisor's suggestion; the sixteen runs already completed are unaffected and
are not re-run, because the matrix grew along the depth axis only and
`run_main_runs.py` skips anything already complete.

| `d` | Role |
|---|---|
| 0 | Baseline — no reranking |
| 3 | Ordering control — rescores the three passages dense retrieval already chose, so promotion is zero by construction |
| **5** | **Minimal treatment — the shallowest depth that can promote at all.** Paired with `d = 3` it isolates what changes when one or two passages are swapped and nothing else |
| 10 | Middle |
| **30** | **Fills the gap between 10 and 50**, so the shape of any diminishing return is observed rather than assumed |
| 50 | Full candidate pool rescored |

The two new depths need no new retrieval: the cache holds cross-encoder scores
for all fifty candidates, so `d = 5` and `d = 30` are re-orderings of data that
already exists. Nothing in Week 2 is recomputed and the per-round training cost
is unchanged.

Four seeds stay four: seeds alternate between the two topic partitions, so an
even count keeps the allocation balanced, and six depths × four seeds is 12
runs on each partition.

The matrix was 4 x 5 = 20 while Week 1-3 were written. It is 4 x 4 here for two
reasons. Seeds alternate between the two topic partitions, so only an even seed
count uses both equally - five seeds would be 1001 three times and 1002 twice,
mixing the depth effect with the allocation. And `enumerate_runs` puts seed on
the outer loop, so the sweep is complete at every multiple of four; 16 is a whole
design rather than a truncated one. Week 1's report quotes the earlier 4 x 5
figure and is left as the record of what was planned then.

---

## Before the sweep: clear `results/logs/runs/`

Anything in `results/logs/runs/` with a `completed: true` summary is skipped, so
a run left over from Week 3 or from a timing measurement silently becomes one of
the sixteen. Week 3's verification run was trained at batch 2 x accum 4 and the
sweep uses batch 8 x accum 1, so it must not be counted.

```bash
mkdir -p results/logs/week3
mv results/logs/runs/d10_s1*  results/logs/week3/    # keep: Week 3's evidence
rm -f results/logs/runs/d0_s1*                       # discard: timing measurement
ls results/logs/runs/                                # must be empty
```

The timing measurement is discarded because it was launched with `--set`
overrides before the manifest recorded the batch split, so `check_runs.py`
reports its setting as unverifiable.

### What happened in practice

`d10_s1.jsonl` was deleted rather than moved while clearing the way for this
sweep, and `results/logs/runs/` is gitignored, so there was no second copy. What
survived is the run's stdout, `results/logs/week3/w3_d10_s1.log`: 59 rounds of
round number, loss, drift, cumulative MB and the F1/EM at every scheduled
evaluation. The figures the Week 1-3 reports quote (F1 28.94, EM 21.90, 1.7400
GB over 59 rounds) are all in it.

It was not rebuilt into a `.jsonl`. The stdout carries everything the reports
cite, and nothing reads the file: every reader globs `results/logs/runs/*.jsonl`
or names `results/logs/runs/{run}.jsonl` directly, so a reconstruction would only
risk a derived file being mistaken for a primary one. `promotion_rate`,
`rank_change_rate`, `eval_gold_recall_at_3` and the exact byte counts were never
printed and are gone; Week 4 measures all of them again at `d=10` over four
seeds.

**The Week 3 file is not reused, and two different runs now share the name
`d10_s1`.** Week 4's sweep runs its own depth 10, seed 1 and writes
`results/logs/runs/d10_s1.jsonl` at batch 8 x accum 1; the Week 3 run was batch
2 x accum 4 and lives only as `results/logs/week3/w3_d10_s1.log`. They do not
collide - different directory, different extension, and the `w3_` prefix - but
they do report different final F1 under the same name. 28.94 is the Week 3
figure the Week 1-3 reports quote; whatever Week 4's `d10_s1` reports is the one
Week 5 analyses, and the two are not comparable because the batch split
differs.

`.gitignore` now keeps `results/logs/week3/*.log` as well as `*.json` and
`*.jsonl`, after the global `*.log` rule so the negation wins. Without that line
the preserved logs would still have lived only on the instance - the same
fragility that lost the curve in the first place.

---

## Run the sweep

Only one sweep may run at a time. Launching the `nohup` line twice gives both
processes the same pending list, so they write interleaved rounds into the same
`results/logs/runs/*.jsonl` and load the model onto the GPU twice, which pushes
past 24 GB and kills run after run with no obvious cause in the log. A pid lock
at `results/logs/week4/sweep.pid` now refuses the second one; a lock left by a
killed process is ignored. `--dry-run` and `run_status.sh` work regardless.

```bash
python scripts/run_main_runs.py --dry-run     # show the matrix and what is pending
python scripts/run_main_runs.py               # execute, skipping completed runs
python scripts/run_main_runs.py --only-depth 50 --only-seed 3
python scripts/check_runs.py                  # post-hoc integrity check
python scripts/measure_flops.py               # cost-model inputs for Week 5
```

### The FLOPs per pair is the denominator of the exchange rate

A factor error here scales the verdict directly, and the first measurement
carried two, in opposite directions.

The analytic estimate used `2 × params × tokens` over all 22.7M parameters of
`ms-marco-MiniLM-L-6-v2`, but 11.9M of those (52 %) are the embedding matrix,
which is read by a gather and never enters a matmul. That overcounted by 2.1×.
The measured figure was taken on an unpadded sample that tokenises to 66 tokens
while the estimate assumes 256, which undercounted by 3.9×. Together the two
numbers sat 8.7× apart and `describe` silently preferred the measured one, so
the cost model would have charged reranking about a ninth of its real compute
and made it look that much more favourable.

Both are fixed: the parameter counts exclude embeddings (10.8M for the reranker,
21.4M for the embedder), the measurement pads to the same token count the
estimate assumes, and `describe` records both token counts and warns when the
two diverge by 2× or more. They now agree to 11 % for the reranker and 6 % for
the embedder — the remainder is the profiler not counting layernorm, softmax and
elementwise work.

`torch.profiler`'s `with_flops` was verified to report 2×MAC, i.e. true FLOPs,
so no factor-of-two correction applies.

### Throughput needs a device sync

`rerank_pairs_per_second` was timed around a single un-warmed forward with no
`torch.cuda.synchronize()`. CUDA kernels queue asynchronously, so the call
returns before the work finishes and that one pass also carries the host-to-device
transfer and kernel compilation. It produced 81 pairs/s on an A10G against 84 on
a laptop CPU; a GPU reading below a CPU is a measurement fault, not a result.
The timing now warms up twice, repeats, and synchronises, and records the batch
size, repeat count and tokens per pair. **The A10G figure recorded during Week 4
predates this fix and should not be used**; it needs re-measuring on the target
device before the cost model's `device_throughput_flops` assumption can be
checked against it.

```bash
python scripts/measure_flops.py --device cuda --pairs 64 --repeats 5
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
R × N × (|A| + |B|) × 2  =  R rounds × 8 clients × 1.84 MB × 2 directions
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

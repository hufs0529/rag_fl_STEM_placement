# Week 4 — Executing the Sixteen Main Runs

**Project:** Buying Back Communication with Retrieval Compute — The Exchange Rate of
Client-Side Reranking in Federated LLM Fine-Tuning

**Covers:** Subtask 1.3 — the main runs on a fixed communication budget, and the
after-the-fact verification that only one axis varied.

**Not covered here:** the exchange rate, the break-even surface and the mechanism
check are Subtask 1.4 and are reported separately. This report establishes what
was measured and that it was measured cleanly; it does not interpret the
differences between depths.

---

## 1. What Week 4 produced

Sixteen complete federated trainings — four reranking depths at four seeds —
each spending an identical communication budget with no early stopping.

| | |
|---|---|
| Runs | **16 / 16 complete**, 0 failures |
| Wall clock | **22.6 hours** on one NVIDIA A10G, 84.8 min per run |
| Rounds per run | 59 |
| Clients per round | 8 |
| Local steps per round | 32 |
| Communication per run | **1,739,980,800 bytes (1.7400 GB)** |
| Communication in total | 27.8 GB |
| Evaluation points per curve | 21 (round 0 plus 20 scheduled) |

One run is 59 × 8 = 472 client training episodes, so the sweep is 7,552 of them.

### Why sixteen and not twenty

The matrix was 4 × 5 = 20 while Weeks 1–3 were written. It was reduced to cut the
sweep's wall clock, and sixteen is the number because it is the only balanced
matrix below twenty.

Seeds are assigned to the two topic partitions alternately, so only an even seed
count uses both equally. Five seeds give partition 1001 three runs per depth and
1002 two; three seeds give 2:1; four give 2:2. Fifteen is not a matrix at all —
four does not divide fifteen, and stopping a seed-major sweep there omits
`d50_s4`, the strongest treatment cell.

So the reduction also corrects an imbalance the original plan carried: at five
seeds the depth effect would have been measured under one partition more often
than the other.

| Matrix | Runs | Partition split per depth | Measured cost at 84.8 min/run |
|---|---|---|---|
| 4 × 3 | 12 | 2 : 1 — unbalanced | 17.0 h |
| **4 × 4** | **16** | **2 : 2 — balanced** | **22.6 h** |
| 4 × 5 | 20 | 3 : 2 — unbalanced | 28.3 h |
| 4 × 6 | 24 | 3 : 3 — balanced | 33.9 h |

The cost of four seeds rather than five is a standard error about 12 % wider
(√5⁄√4); the saving is 5.7 hours of GPU time. Should four prove too few, the
matrix extends to 4 × 6 = 24 — the next balanced point — without repeating
anything: seed is the outer loop and completed runs are skipped, so only the
eight new runs execute.

The primary artefacts are the sixteen learning curves, `results/logs/runs/*.jsonl`,
580 KB, now under version control. They are the only input the Subtask 1.4
analysis reads.

---

## 2. Execution

### 2.1 The batch setting was measured, not assumed

The effective batch is `batch_size × grad_accumulation` and is fixed at 8 by the
Week 1 calibration: changing that product changes `S = (questions per client ÷
effective batch) × passes` and therefore `R = S / K`. How the 8 is split does not
change the budget, only the speed.

Weeks 1–3 ran at batch 2 × accumulation 4, which was forced by CPU memory. On the
GPU that split performs four forward and backward passes per optimiser step and
pays the fixed per-pass cost four times. Measured before committing to the sweep:

| Setting | GPU utilisation | Per round | Per run |
|---|---|---|---|
| batch 2 × accum 4 | 39 % | 160 s | 171 min |
| **batch 8 × accum 1** | **100 %** | **~110 s** | **84.8 min** |

The sweep ran at 8 × 1. Every run's manifest records the split, so the choice is
verifiable after the fact rather than remembered.

### 2.2 Order of execution is a design decision

Runs are ordered **seed outer, depth inner**, so an interrupted sweep still holds
all four conditions for every finished seed. Ordering by depth would leave four
baselines and nothing to compare them against.

### 2.3 Concurrency and resumption

Only one sweep may run at a time: two processes would write interleaved rounds
into the same curve file and load the model onto the GPU twice. A pid lock
refuses the second. A run that is retried opens its log fresh rather than
appending, so a failed attempt cannot leave a curve that walks backwards.

---

## 3. Verification that only one axis varied

`scripts/check_runs.py`, run after the sweep, compares every budget field across
the sixteen runs and checks the treatment diagnostics. The recorded result:

```json
"runs_expected": 16,  "runs_complete": 16,
"identical_budget": true,  "issues": [],
"total_bytes": [1739980800],
"problems": []
```

`total_bytes` is a **set**, and it holds one value. Rounds, client count, payload
size, local steps per round, batch size, accumulation and examples per step are
compared the same way, as is the evaluation schedule. A field recorded for some
runs but not others is itself reported as an issue, so a run predating a field
cannot pass unnoticed.

### 3.1 The treatment took effect, and the controls held

`promotion_rate` is the fraction of the three prompt passages that dense
retrieval would not have selected.

| Depth | `promotion_rate` | Required | |
|---|---|---|---|
| 0 | **0.0000** | exactly 0 | ✅ |
| 3 | **0.0000** | exactly 0 | ✅ |
| 10 | 0.4202 | non-zero | ✅ |
| 50 | **0.5252** | non-zero, and the largest | ✅ |

Three conditions are checked and all pass: the controls promoted nothing, the
treatments promoted something, and the rate is monotone in depth.

`rank_change_rate` — the fraction of prompt slots whose passage differs from the
dense top-3 at that position — separates the two controls:

| Depth | `rank_change_rate` | `promotion_rate` | Reading |
|---|---|---|---|
| 0 | 0.000 | 0.000 | nothing happens |
| **3** | **0.449** | **0.000** | **same three passages, different order** |
| 10 | 0.677 | 0.420 | new passages promoted |
| 50 | 0.725 | 0.525 | most promoted |

`d = 3` reorders 45 % of the prompt slots while promoting nothing. That is the
arm which separates an ordering effect from a retrieval-quality effect, and the
measurement confirms it behaved as designed.

### 3.2 Evaluation context was identical across conditions

Evaluation is held at `d = 0` for every condition, so that what is measured is
what the model learned rather than what a better test-time context supplies. The
logs confirm it: gold recall at the three evaluation prompt slots, at the final
round, is

| Depth | Seeds 1–4 |
|---|---|
| 0 | 0.497, 0.494, 0.497, 0.494 |
| 3 | 0.497, 0.494, 0.497, 0.494 |
| 10 | 0.497, 0.494, 0.497, 0.494 |
| 50 | 0.497, 0.494, 0.497, 0.494 |

Identical across depths, varying only with the partition seed. Every condition
was scored on the same context.

---

## 4. Recorded outcomes

Answer F1 and exact match on the fixed 1,000-question held-out set after the full
budget was spent. These are the logged values, reported as data; the comparison
between depths belongs to Subtask 1.4.

The partition column records which of the two topic allocations the run used.
They hold slightly different numbers of questions per client — 9,844 under seed
1001 and 9,742 under 1002 — because equality within a partition is enforced by
trimming every client to the size of the smallest, and the seed drives both the
topic clustering and the allocation of clusters to clients. The 1.0 % difference
is a level shift between partitions, not a factor in the depth comparison, which
is made within a seed.

| Run | Partition | F1 | EM | Minutes |
|---|---|---|---|---|
| d0_s1 | 1001 | 27.89 | 21.10 | 84.2 |
| d0_s2 | 1002 | 29.92 | 23.20 | 85.0 |
| d0_s3 | 1001 | 28.04 | 21.40 | 84.9 |
| d0_s4 | 1002 | 30.66 | 23.60 | 84.9 |
| d3_s1 | 1001 | 28.29 | 21.60 | 84.9 |
| d3_s2 | 1002 | 31.21 | 24.00 | 85.0 |
| d3_s3 | 1001 | 28.05 | 21.20 | 84.8 |
| d3_s4 | 1002 | 30.21 | 22.90 | 85.1 |
| d10_s1 | 1001 | 28.64 | 21.80 | 84.7 |
| d10_s2 | 1002 | 30.90 | 23.70 | 84.8 |
| d10_s3 | 1001 | 28.94 | 22.00 | 84.7 |
| d10_s4 | 1002 | 30.41 | 23.30 | 84.8 |
| d50_s1 | 1001 | 28.15 | 21.30 | 84.5 |
| d50_s2 | 1002 | 30.10 | 22.70 | 84.6 |
| d50_s3 | 1001 | 29.14 | 22.30 | 84.6 |
| d50_s4 | 1002 | 30.24 | 23.50 | 84.6 |

Descriptive summary, mean over the four seeds with its standard error:

| Depth | F1 | EM |
|---|---|---|
| 0 | 29.13 ± 0.69 | 22.32 ± 0.63 |
| 3 | 29.44 ± 0.76 | 22.43 ± 0.64 |
| 10 | 29.72 ± 0.55 | 22.70 ± 0.47 |
| 50 | 29.41 ± 0.48 | 22.45 ± 0.46 |

The seed-to-seed spread is larger than any difference between depths: the
baseline alone ranges from 27.89 to 30.66, a span of 2.77 F1. This is why the
design runs every depth at every seed — the comparison that Subtask 1.4 makes is
paired within a seed, where that spread cancels.

---

## 5. Cost-model inputs measured for Subtask 1.4

`scripts/measure_flops.py` fixes the per-pair reranking cost that the cost model
charges. Two errors were found and corrected before the figures were used.

The analytic estimate applied `2 × params × tokens` to all 22,713,601 parameters
of the reranker, but 11,917,824 of them — 52 % — are the embedding matrix, which
is read by a gather and never enters a matrix multiply. That overcounted by 2.1×.
The measurement, meanwhile, ran on an unpadded sample that tokenises to 66 tokens
while the estimate assumes 256, undercounting by 3.9×. The two figures therefore
sat 8.7× apart and the code silently preferred the measured one.

| | Before | After |
|---|---|---|
| Analytic, per pair | 1.223e10 | **6.131e9** |
| Measured, per pair | 1.402e9 (66 tokens) | **5.437e9 (256 tokens)** |
| Agreement | 8.7× apart | **11 % apart** |

The parameter counts now exclude embeddings, the measurement pads to the token
count the estimate assumes, both token counts are recorded, and a divergence of
2× or more raises a warning. The residual 11 % is the profiler not counting
layernorm, softmax and elementwise work.

Checked rather than assumed: `torch.profiler`'s `with_flops` reports 2 × MAC for
a matrix multiply, so it is already in FLOPs and no factor-of-two applies.

The throughput figure is **not usable**. It was timed around one un-warmed forward
pass with no device synchronisation; CUDA queues asynchronously, so the call
returns before the work finishes and that pass also carries the host-to-device
copy and kernel compilation. It reported 81 pairs/s on the A10G against 84 on a
laptop CPU, and a GPU reading below a CPU is a measurement fault rather than a
result. The timing now warms up, repeats and synchronises, and records the batch
size, repeat count and tokens per pair, but the figure needs re-measuring on the
target device.

---

## 6. Limitations visible in this week's data

Stated here rather than discovered at the end.

**The curves have not converged.** Fourteen of the sixteen peak at round 50 or
later, and training loss was still falling at round 59. Behaviour beyond this
budget is not observed, and no target accuracy above what the baselines attained
can be reported without extrapolating.

**The curves are not monotone.** Held-out F1 falls between consecutive evaluation
rounds by as much as 1.38 F1 late in training while training loss continues to
fall, so those dips are evaluation noise rather than overfitting. Overfitting
would also be surprising at 1.53 passes over each client's data.

**The 1,000-question evaluation set sets the noise floor.** Absolute F1 varies
0.61 F1 across seeds. Because every condition is scored on the *same* 1,000
questions, the noise from which questions were drawn cancels in a paired
comparison: the standard error of the paired difference runs 0.01–0.48 F1,
median 0.17, about four times smaller. Enlarging the set to 3,000 would cut the
noise further but take evaluation from 36 s to roughly 108 s per scheduled round,
turning a 22.6-hour sweep into about 32 hours.

**Three of the four depths are forced; `d = 10` is not.** `d = 0` is the
baseline, `d = 3` must equal `top_k` for the ordering control to exist, and
`d = 50` is the whole candidate pool. The interior point was placed roughly
geometrically between 3 and 50 and is not derived.

**The candidate pool of 50 is a declared trade.** Gate 2 measured the ceiling at
0.669 for a pool of 50, 0.721 at 100 and 0.774 at 200. A larger pool raises the
ceiling but raises reranking FLOPs proportionally, and those FLOPs are the cost
side of the exchange rate.

**A single model scale and adapter rank.** SmolLM2-135M with LoRA rank 8 over the
attention projections, 921,610 trainable parameters. Four seeds, so effect sizes
and standard errors are reported and no significance test is claimed.

---

## 7. Status

Subtask 1.3 is complete. The sixteen curves exist, were bought with an identical
budget, and the budget's identity and the treatment's effect are both verified
from the logs rather than asserted. Subtask 1.4 reads those curves.

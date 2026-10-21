# Week 4 — Executing the Twenty-Four Main Runs

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

Twenty-four complete federated trainings — six reranking depths at four seeds —
each spending an identical communication budget with no early stopping.

| | |
|---|---|
| Runs | **24 / 24 complete**, no failures |
| Wall clock | **33.9 h** on one NVIDIA A10G, 84.7 min per run |
| Rounds per run | 59 |
| Clients per round | 8 |
| Local steps per round | 32 |
| Communication per run | **1,739,980,800 bytes (1.7400 GB)** |
| Communication in total | 41.8 GB |
| Evaluation points per curve | 21 (round 0 plus 20 scheduled) |

One run is 59 × 8 = 472 client trainings, so the sweep is 11,328 of them.

### Why the ladder has six rungs

While Weeks 1-3 were written the ladder was four rungs, `{0, 3, 10, 50}`. The
supervisor asked for intermediate depths so the trend in depth could be read
rather than inferred, offering "7, 20 and 30 (or 7 and 25)"; the second set was
taken, giving `{0, 3, 7, 10, 25, 50}`. Successive rungs are 2.33, 1.43, 2.5 and
2.0 times apart, which is even on a log axis — the axis a depth trend must be
read on, because the cross-encoder work a rung costs is proportional to `d`.

`5`, `20` and `30` were configured at one point and not measured. The control
says why. `d` = 3 cannot change which passages reach the prompt, yet it scores
**0.31 F1** above `d` = 0, and every treatment effect measured is that size or
smaller. A finer ladder does not let a ruler read a quantity below its smallest
division. Closing that gap with seeds instead would take sixteen of them: 64
runs, 90 hours.

Four seeds were kept. Seeds alternate between the two topic partitions, so only
an even count uses both equally — six depths at four seeds is 12 runs on each.

| Seeds | Partition split per depth | Runs at 6 depths | At 84.7 min per run |
|---|---|---|---|
| 3 | 2 : 1 — unbalanced | 18 | 25.4 h |
| **4** | **2 : 2 — balanced** | **24** | **33.9 h** |
| 5 | 3 : 2 — unbalanced | 30 | 42.4 h |
| 6 | 3 : 3 — balanced | 36 | 50.8 h |

The primary artefact is twenty-four learning curves in
`results/logs/runs/*.jsonl`, 868 KB, under version control. They are the **only**
input the Subtask 1.4 analysis reads.

---

## 2. Execution

### 2.1 Order of execution is a design decision

Runs are ordered **seed outer, depth inner**, so an interrupted sweep still holds
all four conditions for every finished seed. Ordering by depth would leave four
baselines and nothing to compare them against.

### 2.2 Concurrency and resumption

Only one sweep may run at a time: two processes would write interleaved rounds
into the same curve file and load the model onto the GPU twice. A pid lock
refuses the second. A run that is retried opens its log fresh rather than
appending, so a failed attempt cannot leave a curve that walks backwards.

---

## 3. Verification that only one axis varied

`scripts/check_runs.py`, run after the sweep, compares every budget field across
all twenty-four runs and checks the treatment diagnostics. The recorded result:

```json
"runs_expected": 24,  "runs_complete": 24,
"identical_budget": true,  "issues": [],
"total_bytes": [1739980800],
"problems": []
```

`total_bytes` is a **set**, and it holds one value. Rounds, client count, payload
size, local steps per round and the optimiser settings are compared the same
way, as is the evaluation schedule. A field recorded for some
runs but not others is itself reported as an issue, so a run predating a field
cannot pass unnoticed.

### 3.1 The treatment took effect, the controls held — and what those rates imply

`promotion_rate` is the fraction of the three prompt passages that dense
retrieval would not have selected; `rank_change_rate` is the fraction of slots
holding a different passage from the dense top three. The first measures how much
the **content** changed, the second how much the **arrangement** did.

| Depth | `rank_change_rate` | `promotion_rate` | Reading |
|---|---|---|---|
| 0 | 0.000 | **0.0000** | nothing happens |
| **3** | **0.449** | **0.0000** | **same three passages, different order** |
| 7 | 0.648 | 0.3619 | new passages promoted |
| 10 | 0.677 | 0.4202 | |
| 25 | 0.714 | 0.4976 | |
| 50 | 0.725 | **0.5252** | most promotion |

All three conditions pass: the controls (`d` = 0, 3) promoted nothing, the
treatments did, and the rate is monotone in depth. That is the verification.
What follows is what the numbers say.

**(i) `promotion_rate` is the dose of the treatment, and it saturates in depth.**
The rate rises monotonically but the increments collapse. Per unit of depth:

| Depth | `promotion_rate` | Promotion per unit depth |
|---|---|---|
| 7 | 0.3619 | **0.0517** |
| 10 | 0.4202 | 0.0420 |
| 25 | 0.4976 | 0.0199 |
| 50 | 0.5252 | **0.0105** |

Efficiency falls **4.9-fold** from `d` = 7 to `d` = 50. Raising depth five times
over, from 10 to 50, buys only **25 %** more promotion, 0.420 to 0.525. The
passages the cross-encoder will promote sit near the **front of the candidate
pool**, and rescoring the tail is largely wasted. The diminishing return appears
**here**, not in performance.

**(ii) The gap between the two rates separates reshuffling from replacement.** At
`d` = 7, 64.8 % of slots change but only 36.2 % hold a new passage; the
difference of 28.6 points is **the original three in a different order**. The gap
narrows with depth (72.5 % against 52.5 % at `d` = 50, 20.0 points). Shallow
treatment is substantially reordering, and only at depth does replacement
dominate. That is why `d` = 3 exists as a control: the reordering component has
to be measurable on its own.

**(iii) `d` = 3 is a control and also this measurement's ruler.** It rescores the
three passages dense retrieval already chose, so it **cannot promote a fourth**
and its promotion rate is zero by construction. The prompt content is identical
to `d` = 0. Any difference in outcome between them is therefore variation
**unrelated to retrieval quality**. Measured, it is **+0.31 F1** (± 0.37). That
figure sets the scale on which every depth comparison in Section 4 must be read:
a treatment effect near 0.31 is the size of the effect of changing nothing.

### 3.2 Evaluation context was identical across conditions — and why it must be

Evaluation is held at `d` = 0 for every condition. Gold recall across the three
evaluation-prompt slots at the final round:

| Depth | Seeds 1–4 |
|---|---|
| 0 | 0.497, 0.494, 0.497, 0.494 |
| 3 | 0.497, 0.494, 0.497, 0.494 |
| 7 | 0.497, 0.494, 0.497, 0.494 |
| 10 | 0.497, 0.494, 0.497, 0.494 |
| 25 | 0.497, 0.494, 0.497, 0.494 |
| 50 | 0.497, 0.494, 0.497, 0.494 |

Identical to three decimals regardless of depth, varying only with the partition
seed. **Every condition was scored on the same context.**

**Why it has to be fixed.** Reranking can raise a score by two routes, and they
are not the same thing.

| Route | How it works | Relation to communication |
|---|---|---|
| **Training route** | better passages → better gradients → better weights | **Direct.** Better weights reach a target in fewer rounds, and rounds are bytes |
| **Serving route** | better passages at inference → better answers from the same weights | **None.** It requires running the reranker on every query forever and is independent of round count |

Evaluating each run at its own depth would **sum the two routes inseparably**. A
`d` = 50 run scoring higher could be scoring higher because its weights improved
or because it was handed better documents at test time, and nothing in the number
would say which. The serving route almost certainly does raise the score — Week 2
measured gold recall 3.92 points higher at `d` = 50 than at `d` = 0 — so folding
it in would manufacture the conclusion that reranking reduces communication while
nothing had in fact been reduced.

**What fixing it costs.** The F1 reported here is lower than a system actually
deployed at `d` = 50 would score. That is deliberate. The question is not what the
best serving configuration scores, but what the same communication budget buys in
each condition. The serving gain is accounted separately in the Subtask 1.4 cost
model, as FLOPs spent — priced rather than free.

---

## 4. Recorded outcomes

### What F1 and EM measure

Both score the model's generated answer against the reference answers, after the
official Natural Questions normalisation — lowercase, strip punctuation, drop the
articles *a*, *an* and *the*, collapse whitespace.

**Exact match (EM)** is 1 if the normalised strings are identical and 0 otherwise.
**Answer F1** is the token overlap: the harmonic mean of how much of what the
model said appears in the reference, and how much of the reference the model
said. Where a question carries several acceptable answers, the score is the best
one.

| Model's answer | Reference | F1 | EM |
|---|---|---|---|
| `The Barack Obama` | Barack Obama | **1.000** | **1** |
| `Barack Hussein Obama` | Barack Obama | **0.800** | 0 |
| `Obama` | Barack Obama | **0.667** | 0 |
| `Donald Trump` | Barack Obama | 0.000 | 0 |

`Barack Hussein Obama` shares two of its three tokens with the two-token
reference, giving precision 2/3 and recall 2/2, so F1 is 0.800 while EM is 0.
Both are reported on the conventional 0–100 scale, so an F1 of 29.13 means the
mean per-question F1 was 0.2913 and an EM of 22.32 means 22.32 % of the thousand
questions were answered exactly.

F1 is the primary metric because it is continuous and therefore interpolates
stably; EM is a step function and would make any rounds-to-target estimate jump.
EM is recorded alongside because F1 alone invites the objection that a score rose
only through incidental token overlap — if EM moves with it, the model is getting
answers right rather than merely close. EM cannot exceed F1: an exact match
scores 1 on both.

### The twenty-four recorded results

Answer F1 and EM on the fixed 1,000-question held-out set after the budget was
fully spent.

| run | partition | F1 | EM | min | | run | partition | F1 | EM | min |
|---|---|---|---|---|---|---|---|---|---|---|
| d0_s1 | 1001 | 27.89 | 21.10 | 84.2 | | d10_s1 | 1001 | 28.64 | 21.80 | 84.7 |
| d0_s2 | 1002 | 29.92 | 23.20 | 85.0 | | d10_s2 | 1002 | 30.90 | 23.70 | 84.8 |
| d0_s3 | 1001 | 28.04 | 21.40 | 84.9 | | d10_s3 | 1001 | 28.94 | 22.00 | 84.7 |
| d0_s4 | 1002 | 30.66 | 23.60 | 84.9 | | d10_s4 | 1002 | 30.41 | 23.30 | 84.8 |
| d3_s1 | 1001 | 28.29 | 21.60 | 84.9 | | d25_s1 | 1001 | 28.42 | 21.40 | 83.8 |
| d3_s2 | 1002 | 31.21 | 24.00 | 85.0 | | d25_s2 | 1002 | 30.24 | 23.10 | 84.6 |
| d3_s3 | 1001 | 28.05 | 21.20 | 84.8 | | d25_s3 | 1001 | 28.88 | 22.10 | 84.4 |
| d3_s4 | 1002 | 30.21 | 22.90 | 85.1 | | d25_s4 | 1002 | 29.41 | 22.40 | 84.7 |
| d7_s1 | 1001 | 28.10 | 21.40 | 84.3 | | d50_s1 | 1001 | 28.15 | 21.30 | 84.5 |
| d7_s2 | 1002 | 30.35 | 23.00 | 84.8 | | d50_s2 | 1002 | 30.10 | 22.70 | 84.6 |
| d7_s3 | 1001 | 27.70 | 20.80 | 84.6 | | d50_s3 | 1001 | 29.14 | 22.30 | 84.6 |
| d7_s4 | 1002 | 30.66 | 23.40 | 84.8 | | d50_s4 | 1002 | 30.24 | 23.50 | 84.6 |

**Wall clock is independent of depth** (83.8–85.1 min, a spread of 1.3 min).
Rerank scores were computed once in Week 2 and frozen in a cache, so during
training `d` only **reorders** what is already there. The minutes in this table
therefore carry no reranking cost. That cost is accounted separately as FLOPs in
Section 6 — mixing the two would hide what depth charges.

### 4.1 Depth against compute, performance and communication

The four axes side by side. Cost comes from the measured FLOPs of Section 6
(5.437 × 10⁹ per pair, 20.34 pairs/s on CPU), performance from the table above,
communication from the fixed budget and the rounds-to-target.

| `d` | FLOPs per query | Seconds per query | `promotion` | F1 (mean ± s.e.) | **vs `d` = 0** | **vs `d` = 3 (control)** |
|---|---|---|---|---|---|---|
| **0** | 0 | 0.000 | 0.0000 | 29.13 ± 0.69 | — | −0.31 ± 0.37 |
| **3** | 1.63 × 10¹⁰ | 0.147 | 0.0000 | 29.44 ± 0.76 | **+0.31 ± 0.37** | — |
| **7** | 3.81 × 10¹⁰ | 0.344 | 0.3619 | 29.20 ± 0.76 | +0.07 ± 0.16 | **−0.24 ± 0.27** |
| **10** | 5.44 × 10¹⁰ | 0.492 | 0.4202 | 29.72 ± 0.55 | +0.60 ± 0.28 | +0.29 ± 0.25 |
| **25** | 1.36 × 10¹¹ | 1.229 | 0.4976 | 29.24 ± 0.39 | +0.11 ± 0.47 | **−0.20 ± 0.42** |
| **50** | 2.72 × 10¹¹ | 2.458 | 0.5252 | 29.41 ± 0.48 | +0.28 ± 0.31 | **−0.03 ± 0.45** |

Comparisons are **paired within seed**, because seed-to-seed variation exceeds
the differences between depths: the baseline alone spans 27.89 to 30.66, 2.77 F1,
while no depth differs from another by a whole point. Pairing cancels that span.

On the communication side every condition spent an identical **1.7400 GB** by
design, so a saving can only arise as *reaching the same target in fewer rounds*.
At the representative target τ = 25.02 F1:

| `d` | Rounds to target | s.e. | Rounds saved | Communication saved |
|---|---|---|---|---|
| 0 | 9.02 | 0.17 | — | — |
| 3 | 9.63 | 0.28 | −0.61 | −17.8 MB |
| 7 | 11.08 | 0.83 | −2.05 | −60.6 MB |
| 10 | 10.05 | 0.71 | −1.03 | −30.4 MB |
| 25 | 8.92 | 0.51 | **+0.10** | +2.9 MB |
| 50 | 9.17 | 0.86 | −0.15 | −4.3 MB |

### 4.2 Three questions the table is there to answer

**(a) Does increasing depth consistently improve performance — no.**

Against `d` = 0 the gains run +0.07, +0.60, +0.11, +0.28: **not monotone**, and
every one inside its own standard error. The decisive column is the comparison
against the **control**. `d` = 3 cannot change prompt content, yet it sits
**+0.31** above `d` = 0. Read against the control instead of the baseline,
**three of the four treatments are negative** (−0.24, −0.20, −0.03), and the
+0.60 at `d` = 10 — the largest — shrinks to +0.29 ± 0.25, more than half of it
being a component the control shares.

> Promotion rises cleanly and monotonically from 0 to 0.525 while performance
> does not follow. The dose of the treatment demonstrably increased; it did not
> carry through to what was learned.

**(b) Where do diminishing returns appear — in promotion, not in performance.**

There is no performance return to diminish. The diminishing return is in Section
3.1's dose: promotion bought per unit of depth falls from 0.0517 at `d` = 7 to
0.0105 at `d` = 50, **4.9-fold**. Cost meanwhile rises linearly with depth, 0.344
to 2.458 seconds per query, **7.1-fold**. So as depth increases, **cost grows
linearly while the dose saturates.** The knee is between `d` = 10 and `d` = 25.

**(c) Which depth is the best trade-off — `d` = 0.**

Every depth spends FLOPs per query and buys neither measurable performance nor
communication. `d` = 50 spends 2.72 × 10¹¹ FLOPs per query — 2.46 s on this CPU —
and spends 4.3 MB **more** communication. The only positive saving, +0.10 rounds
at `d` = 25, is **one fifth** of its own standard error of 0.51 and is not
distinguishable from zero. The Subtask 1.4 mechanism check returns the same
verdict: *the largest saving is inside its own uncertainty, so this measurement
resolves no saving.*

### 4.3 How to read this result

The result is negative but the measurement did not fail: it localises which link
in the chain breaks.

| Link | Measured | Status |
|---|---|---|
| depth → retrieval quality | Week 2 gold recall +3.92 pp; promotion 0 → 0.525 | ✅ **works** |
| retrieval quality → training inputs | 52.5 % of prompt passages dense retrieval would not have chosen | ✅ **works** |
| **training inputs → what is learned** | **every F1 difference inside the 0.31 noise floor** | ❌ **breaks** |
| what is learned → rounds saved | nothing to measure | — |

The break is at the **third link**. Demonstrably better passages reached the
prompt, and at this model scale (135M) and this budget (59 rounds, 1.74 GB) that
did not show up as a difference in weights. Read with the limitations in Section
7, the conclusion is not that reranking is useless but that **under these
conditions reranking does not buy back communication**.

---

## 5. Why the two partitions hold different numbers of questions

The results table records which topic allocation each run used. The two hold
9,844 and 9,742 questions per client, a difference of 102 or 1.0 %. The reason is
worth setting out, because it looks at first like an inconsistency.

**Step 1 — questions are grouped by topic.** The partition seed is the
`random_state` of the k-means clustering, so a different seed cuts the clusters
differently: which passages end up in one group changes.

**Step 2 — the clusters are dealt out to eight clients.** Clusters are not the
same size, so the clients do not come out even. The passage counts, which are
recorded without any trimming, show how uneven:

| Seed | Passages per client | Spread |
|---|---|---|
| 1001 | 26219, 26870, 25404, 23942, 25058, 22026, 25331, 25150 | 22,026–26,870, **22 %** |
| 1002 | 25695, 28164, 24899, 23411, 22710, 26977, 23100, 25044 | 22,710–28,164, **24 %** |

Question counts come out uneven in the same way.

**Step 3 — every client is trimmed to the smallest.** Equality is enforced by
cutting all eight down to whatever the least-loaded client holds. The common size
is therefore *the size of the smallest client*, and that is what the clustering
happens to produce.

**Step 4 — the smallest client differs between the two seeds.** 9,844 under 1001,
9,742 under 1002. Trimming discards 1,968 questions (2.4 %) from the first
partition and 2,784 (3.4 %) from the second.

The trimming is deliberate. Unequal clients destabilise the FedAvg average,
because a client with more data pulls the mean further, and that instability
would act as a second varying factor alongside depth. Paying 2,784 discarded
questions buys `question_size_spread: 0`.

The 1.0 % difference between partitions does not reach the depth comparison.
Runs sharing a seed share a partition, and the comparison is made within a seed,
so the figure is the same on both sides of the subtraction and cancels.

---

## 6. Cost-model inputs measured for Subtask 1.4

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

## 7. Limitations visible in this week's data

Stated here rather than discovered at the end.

**The curves have not converged.** Most of the twenty-four peak at round 50 or
later, and training loss was still falling at round 59. Behaviour beyond this
budget is not observed, and no target accuracy above what the baselines attained
can be reported without extrapolating.

**The curves are not monotone.** Held-out F1 falls between consecutive evaluation
rounds by as much as 1.38 F1 late in training while training loss continues to
fall, so those dips are evaluation noise rather than overfitting. Overfitting
would also be surprising at 1.53 passes over each client's data.

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

## 8. Status

Subtask 1.3 is complete. The twenty-four curves exist, were bought with an identical
budget, and the budget's identity and the treatment's effect are both verified
from the logs rather than asserted. Subtask 1.4 reads those curves.

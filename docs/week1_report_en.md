# Week 1 Report — Pilot Gates and Data Preparation

**Project** Buying Back Communication with Retrieval Compute: The Exchange Rate of
Client-Side Reranking in Federated LLM Fine-Tuning

---

## 0. Summary

Total communication in federated learning is `R × N × |AB| × 2`. Most prior work
attacks the per-round payload `|AB|`. This project attacks the **round count `R`**
instead: if better retrieval on the client makes each client learn a more
consistent task, fewer rounds are needed to reach a target, and the **exchange
rate between retrieval compute (FLOPs) and communication (bytes)** becomes
measurable.

Week 1 exists to verify, by measurement, the premises the design rests on. If a
premise fails, the main experiment produces results that cannot be interpreted,
so three pilot gates must pass first.

| Gate | Question | Pre-registered criterion | Measured | Verdict |
|---|---|---|---|---|
| ① Capability headroom | Does the base model respond to context quality? | F1 gap ≥ 5.0 | **18.40** | **PASS** |
| ② Depth monotonicity | Does reranking depth improve retrieval? | recall@3 monotone, `recoverable` ≥ 0.10 | **+4.25 pp** (z = 6.00), `recoverable` **0.2020** | **PASS** |
| ③ Local-step calibration | At which `K` is client divergence observable? | divergence ≥ 0.02 | Round 1 complete (§4) | In progress |

Gates ① and ② pass. Gate ③ has completed round 1, which is the unconfounded
per-`K` measurement; this report is written against that round.

---

## 1. Experimental design and the treatment axis

### 1.1 Data and pipeline

| Resource | Role | Scale |
|---|---|---|
| `nq_open` | Questions and answers (the evaluation standard) | 91,535 train questions |
| `psgs_w100` (Wikipedia 100-word passages) | Retrieval corpus | **21,015,300** |

Each question follows one path:

1. **Dense retrieval** — `bge-small-en-v1.5` returns the top 50 from the corpus. Cheap.
2. **Reranking** — `ms-marco-MiniLM-L-6-v2` rescores the top `d`. Expensive.
3. **Selection** — the top 3 after rescoring go into the prompt.

### 1.2 The treatment axis: reranking depth `d`

| `d` | Behaviour | Role |
|---|---|---|
| **0** | No reranking | Baseline |
| **3** | Rescore the top 3 only | **Ordering control** — reorders, never promotes |
| **10** | Rescore the top 10 | Treatment |
| **50** | Rescore the whole pool | Maximum treatment |

`d = 3` exists to protect the interpretation. When `d = 10` beats the baseline, an
alternative explanation is available: the passages were simply presented in a
different order. `d = 3` **cannot by construction** change which passages are
selected — only their order — so if performance does not move there, the ordering
effect has been measured directly as zero, and any gain at `d = 10` or `d = 50`
is attributable to **promotion**. §3.3 reports that measurement.

### 1.3 Controls

* Every condition runs under a **fixed communication budget** with no early
  stopping. With early stopping, "fewer rounds" cannot be separated from an
  artefact of the stopping rule.
* The embedder and the reranker are **frozen** and the corpora are static, so
  candidate lists and rerank scores are computed once and cached: all runs share
  **one retrieval pass**, and no axis varies alongside `d`.
* Rounds-to-target is obtained by linear interpolation; the target `τ` is swept
  and **never extrapolated**.

---

## 2. Data preparation — corpus scan

### 2.1 What was done

All 21,015,300 passages of `psgs_w100` were scanned in a **single pass** to
identify the gold passages for each question. Elapsed: 19.3 hours.

| Artefact | Scale |
|---|---|
| Gold passages | **634,104** |
| Answer-free reservoir | 300,000 |
| Usable train questions | **80,720** / 91,535 (88.2 %) |
| Usable validation questions | 3,356 |
| Question coverage | **91.85 %** |

The reservoir uses Algorithm R, which guarantees a uniform sample in one pass.

### 2.2 Why the gold criterion has two stages

The DPR convention is "a passage is gold if it contains the answer string". Applied
to the **whole** corpus, that does not hold, because 69 % of NQ answers are a
single token ("Paris", "1969", "red").

Measured over the full scan:

| | Count | Share |
|---|---|---|
| Passages containing **no** answer string | **628** | 0.003 % |
| Passages containing some answer string | 21,014,672 | **99.997 %** |

Under the string condition alone, almost the entire corpus is gold for someone.
A two-stage criterion was therefore applied:

> **Stage 1** the answer string is present (an answer-first-token prefilter over passage tokens)
> **Stage 2** at least **two** content terms of the question also occur in that passage

At the (passage, question) pair level:

| | Pairs |
|---|---|
| Passed stage 1 (candidate pairs) | 5,996,794,875 |
| Passed stage 2 (gold pairs) | **1,130,384** |
| **Retention** | **0.019 %** |

The string criterion alone would have produced roughly **six billion spurious
gold pairs**; stage 2 removes 99.98 % of them. Both stages reuse the same
normalised token list, so the second costs almost nothing.

### 2.3 Why recall is measured by gold id, not by re-matching strings

The corpus filler (the answer-free reservoir) consists of passages that are gold
for **no** question — but as §2.2 shows, **99.99 % of them still contain an answer
string**. Scoring recall as "does a selected passage contain the answer string"
therefore counts arbitrary, unrelated passages as hits. The measured inflation was
1.9 % overall and 6.6 % on single-token-answer questions.

Recall here is membership in the gold passage id set
(`gold_recall_at_k(selected_ids, gold_ids, k)`), so the labelling criterion and the
measurement criterion are the same.

---

## 3. Gate ② — Depth monotonicity

### 3.1 Question and method

Does the treatment axis exist? That is, as `d` goes 0 → 50, does the fraction of
questions with a gold passage in the selected top 3 (`recall@3`) rise
**monotonically**?

This requires retrieval only, no training. **2,000 questions** were used; the
sample size was set so the paired estimator resolves the effect at more than
5 s.e. (§3.4).

**The probe corpus is built as a scale model of the Week 2 corpus**, because a gate
that measures a different difficulty than the main experiment measures nothing
useful.

| Component | Count | Share |
|---|---|---|
| Gold of the evaluated questions (1 per question) | 1,925 | 7.7 % |
| **Gold of other questions** | 8,165 | 32.7 % |
| Answer-free filler | 14,910 | 59.6 % |
| **Total** | **25,000** | gold share **40.4 %** |

* The size 25,000 is Week 2's **per-client index scale**.
* The gold share of 40.4 % is derived from the Week 2 corpus (80,720 gold in
  200,000 = 40.36 %) rather than hardcoded, so a different scan result moves it.
* Other questions' gold is included for difficulty. Random Wikipedia passages are
  easy to outrank; another question's gold contains an answer and therefore shares
  vocabulary with questions, which makes it **hard**. Omitting it overstates the
  performance of dense retrieval.
* Invariant: **no gold passage of an evaluated question may enter as filler.** Only
  one gold per question is kept as the label, so another gold of that question
  sitting in the corpus unlabelled would understate recall. The selection logic
  forbids the *untruncated* gold list of every evaluated question.
* BM25 hard negatives are **deliberately excluded**. They push gold further down
  the dense ranking, which *increases* `recoverable`, so excluding them makes the
  measurement conservative: whatever passes here is available at least as much in
  Week 2.

### 3.2 The three-way retrieval headroom

Questions are partitioned by whether the treatment can affect them at all. A
single ceiling figure would conflate "already easy" with "recoverable by
reranking", so the distinction is made explicit.

| Bucket | Definition | Questions | Share | Treatment effect |
|---|---|---|---|---|
| `already` | Gold already in the dense top 3 | 934 | 46.70 % | **exactly 0** |
| `recoverable` | Gold in the pool but below the top 3 | **404** | **20.20 %** | **the only working range** |
| `out_of_pool` | Gold outside the 50-candidate pool | 662 | 33.10 % | **exactly 0** |
| Ceiling | `already` + `recoverable` | 1,338 | 66.90 % | Attainable upper bound |

`recoverable` = 0.2020, twice the pre-registered criterion of 0.10.

### 3.3 Results

| `d` | `recall@3` | Questions correct | rank-change | promotion |
|---|---|---|---|---|
| **0** | 0.4670 | 934 | 0.000 | 0.000 |
| **3** | **0.4670** | **934** | **0.4375** | **0.000** |
| **10** | 0.4950 | 990 | 0.6647 | 0.4228 |
| **50** | **0.5095** | **1,019** | 0.7157 | 0.5290 |

* `rank-change` is the fraction of the three selected slots holding a different
  passage from the dense top 3 **at the same position**.
* `promotion` is the fraction of selected passages that were **not** in the dense top 3.

**The ordering control behaved as designed.** At `d = 3` the reranker rearranged
43.75 % of the selected slots with a promotion rate of 0.000, and `recall@3` is
identical to the baseline to four decimal places. The gains at `d = 10` and
`d = 50` are therefore attributable to promotion, not to reordering.

### 3.4 Effect size and uncertainty

`d = 0` and `d = 50` are measured on the **same questions**, so treating the two
means as independent samples would overstate the variance. Only the questions whose
verdict flipped carry information about the difference (the McNemar structure).

| Quantity | Value |
|---|---|
| Questions rescued (`gained`) | **143** |
| Questions broken (`lost`) | **58** |
| Net | 85 |
| `delta` | **+0.0425** |
| Standard error, `sqrt(gained + lost) / n` | 0.0071 |
| **z** | **6.00** |

At z = 6.00 the effect is clearly separated from zero.

**A secondary observation: reranking is not an unconditional improvement.** By a
logical constraint, all 143 `gained` come from `recoverable` (404) and all 58 `lost`
come from `already` (934): a `recoverable` question is already wrong at baseline and
cannot be broken further, and an `already` question is already right and has nothing
to rescue. Hence

| Measure | Calculation | Value |
|---|---|---|
| Rescue rate within `recoverable` | 143 / 404 | **35.4 %** |
| Breakage rate within `already` | 58 / 934 | **6.2 %** |
| Net capture (`recoverable_captured`) | 85 / 404 | **21.0 %** |

The net 21.0 % is the balance of a 35.4 % rescue against a 6.2 % breakage. This is
only observable in a comparison where `d` alone varies, and it informs the
cost–benefit discussion in Week 5.

### 3.5 Justification for `candidate_pool = 50`

`out_of_pool` is 33.10 %, which is not small. No depth can reach that range, so a
larger pool raises the ceiling. The gate therefore retrieves deeper than the
treatment uses and reports the sensitivity without any additional reranking.

| `candidate_pool` | `recoverable` | `out_of_pool` | Ceiling |
|---|---|---|---|
| **50** (adopted) | 0.2020 | 0.3310 | **0.6690** |
| 100 | 0.2535 | 0.2795 | 0.7205 |
| 200 | 0.3070 | 0.2260 | 0.7740 |

Reasons for 50:

1. The pre-registered criterion (`recoverable` ≥ 0.10) is cleared by a factor of two
   and the effect is established at z = 6.00. There is no statistical-power reason to
   enlarge it.
2. `candidate_pool` is simultaneously the largest value of `d`. Raising it to 200
   would make the treatment grid five levels, taking the run count from 20 to 25 and
   changing the cost model with it.
3. `out_of_pool` caps the ceiling at 0.6690, but **every `d` shares that same
   ceiling**, so comparisons across `d` are unaffected; only absolute accuracy is
   limited.

"Why 50" is therefore answered by measured sensitivity rather than convention: the
ceiling is 0.6690 at 50 and 0.7740 at 200, and 50 suffices to establish the
treatment effect at a quarter of the reranking cost.

---

## 4. Gate ③ — Local-step calibration (round 1)

### 4.1 Question

The round count is not a free parameter:

```
R = S / K          S: total local steps per client,  K: local steps per round
```

If `K` is too small, clients are averaged before they diverge at all, and a
treatment that acts **by reducing divergence** becomes invisible. If `K` is too
large, `R` shrinks until there is no resolution left to interpolate
rounds-to-target. `K` must therefore be fixed by a pilot.

Setup: 8 clients × 250 questions, effective batch 8, baseline condition `d = 0`.
Calibration runs at baseline because the chosen `K` must be one where drift is
observable *without* the treatment, so that the treatment has something to act on.

Measurement uses **nested snapshots**. Within a round every client starts from the
same global parameters, so a trajectory heading for `K = 64` passes through steps
4, 8, 16 and 32. Recording the trainable parameters at those points yields the
smaller `K` values with no additional training. Cost per client per round falls
from `sum(K) = 124` to `max(K) = 64`, and all five `K` are read off **one
trajectory**, so comparisons between them are not confounded by different random
trajectories.

The nesting is **exact only in round 1**: from round 2 the starting point is a
FedAvg result, and that result depends on `K`, so the sharing no longer holds.
`K` selection therefore rests on round 1, and later rounds serve as a robustness
check on whether divergence survives averaging.

### 4.2 Round 1 results

| `K` | Divergence (normalised) | Absolute spread | ‖mean update‖ | Mean pairwise cosine |
|---|---|---|---|---|
| 4 | 0.5097 | 0.2236 | 0.4925 | +0.764 |
| 8 | 0.3895 | 0.3578 | 0.9859 | +0.850 |
| 16 | 0.3376 | 0.5890 | 1.8415 | +0.883 |
| **32** | **0.3343** (min) | 0.9641 | 3.0404 | **+0.885** (max) |
| 64 | **0.4334** (↑) | 1.6042 | 4.0335 | **+0.820** (↓) |

```
divergence      = mean_i ‖ δ_i − δ̄ ‖ / ‖ δ̄ ‖      (normalised)
absolute spread = mean_i ‖ δ_i − δ̄ ‖                (the numerator)
```

Both are recorded because they move in opposite directions in `K`: as `K` grows each
update grows, so the denominator ‖δ̄‖ grows and the normalised divergence falls while
the absolute spread rises. Recording only one would require re-measuring the whole
sweep to revisit the selection criterion.

### 4.3 Interpretation

The normalised divergence is **U-shaped**, passing a minimum at `K = 32` and rising
again at `K = 64`. The cosine mirrors it, peaking at `K = 32` (+0.885) and falling at
`K = 64` (+0.820).

| Region | Observation | Reading |
|---|---|---|
| `K = 4` | Highest divergence (0.510) but the smallest ‖update‖ (0.49) | Small updates mean **relative noise dominates**; this is not task divergence |
| `K = 16–32` | Large ‖update‖, maximum cosine | Clients are **learning a shared task** |
| `K = 64` | Cosine falls, normalised divergence rises | **Task divergence begins** |

The premise that "too small a `K` averages clients before they diverge" is confirmed
numerically: the high divergence at `K = 4` is measurement noise, not drift.

### 4.4 The resolution of `R = S / K`

`S` is the total local steps per client, defined as a number of passes over that
client's data. The examples one step consumes is the **effective batch**, that is
`batch_size × grad_accumulation = 8`:

```
S = (questions per client / effective batch) × passes
```

**The probe scale and the main-experiment scale must be kept apart.** Gate ③ is a
probe and caps each client at 250 questions, whereas in the main experiment each
client receives 80,720 usable questions divided by 8, about 10,090.

| | Questions per client | `S` (1.5 passes) | `K = 32` → `R` | `K = 64` → `R` |
|---|---|---|---|---|
| Gate ③ probe | 250 | 46 | 1.4 | 0.7 |
| **Main experiment (Week 2)** | **10,090** | **1,892** | **59.1** | **29.6** |

Estimating rounds-to-target by linear interpolation requires the communication budget
to be resolved finely enough; `R ≥ 20` puts the resolution at 5 % of the budget or
better. **At the main-experiment scale, 1.5 passes alone give `R = 59` at `K = 32` and
`R = 30` at `K = 64`.** The region where divergence is observable and the region where
interpolation is possible do not conflict.

| Passes | `S` | `R` at `K = 32` | `R` at `K = 64` |
|---|---|---|---|
| **1.5 (current setting)** | **1,892** | **59.1** | **29.6** |
| 2 | 2,522 | 78.8 | 39.4 |

**Provisional conclusion:** `K = 32`, `S = 1,892` (1.5 passes over client data),
`R = 59`. This sits in the region where task divergence is observable while giving a
round resolution of 1.7 %, and it keeps the plan's "`S` is set to 1–2 passes over each
client's data" unchanged, so no configuration change is required. It will be finalised
when the three-round probe completes.

Gate ③ records the probe-scale and main-experiment-scale `S` separately
(`total_local_steps_full_scale`, `rounds_full_scale_by_K`). Conflating them makes `R`
look about 40× smaller than it is.

### 4.5 The decision is enforced in code

Gate ③ writes its result to `results/logs/gates/calibrated_schedule.json`, and the
Week 3 and Week 4 run scripts read that file directly. The claim that "a pilot fixed
`K` and `R`" is therefore enforced by the execution path rather than asserted in
prose: using different values requires editing that artefact.

---

## 5. Design decisions and their evidence

| Decision | Value | Evidence |
|---|---|---|
| Base model | `SmolLM2-135M-Instruct` | Gate ① measured 18.40 F1 of headroom |
| Treatment grid | `d ∈ {0, 3, 10, 50}` | Gate ② confirmed monotonicity (+4.25 pp, z = 6.00) |
| Including `d = 3` | Ordering control | Ordering effect measured directly as zero (promotion 0.000, recall unchanged) |
| Candidate pool | 50 | Measured sensitivity (50 → ceiling 0.669, 200 → 0.774); comparisons across `d` are ceiling-invariant |
| Gold criterion | String match + ≥ 2 question content terms | The string criterion alone yields ~6 billion spurious pairs |
| Recall measurement | Gold id membership | 99.99 % of filler contains an answer string; string re-matching inflates by 1.9 % |
| Probe corpus | 25,000 passages, 40.4 % gold | Mirrors Week 2's per-client index scale and composition |
| Effect statistics | Paired (McNemar) | `d = 0` and `d = 50` see the same questions |
| `K`, `R` | Fixed by gate ③ | `R = S / K`; pilot measurement rather than convention |
| Batch configuration | `batch 2 × grad_accum 4` | Effective batch 8 preserved; the equivalence is pinned by a regression test |

On the last row: CPU training cannot use bfloat16 and therefore runs in float32, where
the `logits` tensor (`batch × seq × vocab × 4 bytes`) is the dominant term. Lowering
the minibatch to 2 and accumulating gradients over 4 steps keeps the effective batch at
8 and yields the **identical parameter update** while lowering peak memory. Because a
test pins the two configurations to the same weights, this change alters the memory
profile without altering the experimental condition.

---

## 6. Reproducibility

| Item | Detail |
|---|---|
| Single source of truth | `configs/experiment_config.yaml` — every number originates here |
| Gate artefacts | `results/logs/gates/*.json` — verdict, measurements and the pre-registered criterion together |
| Decision hand-off | `calibrated_schedule.json` → read directly by the Week 3 and 4 run scripts |
| Regression tests | 150, runnable without GPU or network |
| Retrieval cache | Frozen embedder and reranker over static corpora → candidates and scores computed once and shared by all runs |
| Seeds | 2 partition seeds (1001, 1002), 5 run seeds (1–5) |

---

## 7. Next steps

1. **Complete gate ③** — finish the three-round probe and fix `K`, `S`, `R`.
2. **Week 1 GO / NO-GO** — combine the three gates into `week1_summary.json`.
3. **Week 2** — build the 200,000-passage corpus (80,720 gold + 40,000 BM25 hard
   negatives + a random remainder), partition clients by topic cluster, build the
   indexes, and freeze the retrieval cache.

### Note on the compute budget for the main experiment

The present environment is CPU-only, and the measured training speed is **27 s per
step** at an effective batch of 8. Executing the 20 main runs at that speed would take
roughly 20 days. The main experiment is planned for a GPU environment; the Week 1 gates
were designed and executed to complete under the current constraint regardless (about
14 hours in total across the three gates, 3 % of the estimated main-experiment cost).

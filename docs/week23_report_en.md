# Weeks 2–3 — Freezing the Measurement Substrate, and the Federated Pipeline

**Project:** Buying Back Communication with Retrieval Compute — The Exchange Rate of
Client-Side Reranking in Federated LLM Fine-Tuning

**Covers:** Subtask 1.1 (corpus, partition, indexes, retrieval cache) and
Subtask 1.3 (federated training pipeline and its verification)

---

## 1. Where these two weeks sit

Week 1 established that the question is answerable: there is a capability gap
for reranking to close, depth improves retrieval monotonically, and a local-step
setting exists at which client updates are measurably distinct. Weeks 2 and 3
build the apparatus that will produce the measurement.

| Week | Produces | Consumed by |
|---|---|---|
| 2 | A fixed corpus, client partition, indexes and a **frozen retrieval cache** | Every main run |
| 3 | The federated training pipeline, **verified end to end on one run** | The 16 main runs |

The two weeks have one shared purpose: by the time the main runs begin, **nothing
should vary between conditions except reranking depth.**

---

## 2. Week 2 — the substrate

### 2.1 Why the retrieval cache is computed once and frozen

Reranking is applied to each client's training inputs. If retrieval were run
again inside each main run, two conditions would differ both in depth and in
whatever the retrieval happened to return that time, and the measured effect
could not be attributed to depth.

So the candidate pool, the dense scores and the cross-encoder scores are computed
once per partition and stored. Each main run reads them and only re-orders.

```
per question:  top-50 dense candidates  +  50 dense scores  +  50 cross-encoder scores
```

| Partition seed | Questions per client | Cross-encoder forward passes | Wall clock |
|---|---|---|---|
| 1001 | 9,844 | 3,937,600 | 66 min |
| 1002 | 9,742 | 3,896,800 | 65 min |

Two consequences follow. First, every condition sees byte-identical retrieval.
Second, depth adds no per-round cost during training — the scores already exist,
so `d = 50` and `d = 0` train at the same speed, and the reranking compute is
accounted separately in the cost model rather than confounded with wall clock.

### 2.2 Corpus

200,000 passages, composed so that reranking has something to do and so that
negatives cannot leak answers.

| Component | Count | Role |
|---|---|---|
| Gold | 54,129 | Passages containing an answer to an evaluated question |
| BM25 hard negatives | 37,511 | Lexically close to a question, answer-free |
| Random remainder | 108,360 | Background |
| **Total** | **200,000** | |

```
answer-bearing negative violations:  0
```

Hard negatives are drawn by BM25 from passages that contain no answer string, so
they are answer-free by construction; the violation count confirms it. Without
them retrieval is too easy and there is no headroom for reranking to recover.

Gold is smaller than the question count because questions share passages
(0.67 gold passages per question).

### 2.3 Client partition

Eight clients, partitioned by topic cluster rather than at random, because
topical heterogeneity is the condition federated learning actually operates
under.

| Property | Seed 1001 | Seed 1002 |
|---|---|---|
| Questions per client | 9,844 | 9,742 |
| Question-count spread across clients | **0** | **0** |
| Clients topic-disjoint | **yes** | **yes** |
| Fraction of questions whose gold is on their own client | **1.000** | **1.000** |
| Passages per client | 22,026 – 26,870 | 22,710 – 28,164 |

Three properties are enforced deliberately:

- **Equal question counts (spread 0).** Unequal clients destabilise the FedAvg
  average, which would act as a second varying factor.
- **Gold reachable on the owning client (1.000).** A question whose answer sits
  on another client is unanswerable by construction and would silently depress
  every condition equally.
- **Two independent partitions.** Results are reported across both, so no
  conclusion rests on one allocation.

Client heterogeneity is therefore a **fixed background condition, not a factor.**

### 2.4 Indexes

One vector index per client per partition seed, so each client retrieves only
from its own passages. Passage counts per collection are listed above.

### 2.5 What Week 2 measured: the retrieval headroom

Every question falls into exactly one of three categories, which together bound
what any reranking depth can possibly achieve.

| Category | Seed 1001 | Seed 1002 | Meaning |
|---|---|---|---|
| `already` | 0.5043 | 0.5067 | Gold is already in the dense top-3 — reranking has nothing to do |
| `recoverable` | **0.2287** | **0.2248** | Gold is in the top-50 pool but not the top-3 — **reranking can recover it** |
| `out_of_pool` | 0.2669 | 0.2685 | Gold is not in the pool — unreachable at any depth |
| **Ceiling** | **0.7331** | **0.7315** | `already + recoverable`, shared by every condition |

This is the frame in which the main results must be read. The maximum
improvement available to reranking is 22.87 percentage points, and the ceiling
is identical across conditions. Without it, a gain of a few points cannot be
judged large or small.

### 2.6 Depth ladder, measured

Gold recall at the three prompt slots, averaged over the eight clients:

| Depth `d` | Seed 1001 | Seed 1002 |
|---|---|---|
| 0 | 0.5043 | 0.5067 |
| **3** | **0.5043** | **0.5067** |
| 10 | 0.5349 | 0.5368 |
| 50 | **0.5435** | **0.5455** |

```
gain at d=50     =  0.5435 − 0.5043  =  +3.92 pp
recoverable      =  0.2287
captured         =  0.0392 / 0.2287  =  17.1 % of what was recoverable
```

**`d = 0` and `d = 3` are identical to four decimal places, in 16 of 16 clients
across both partitions.** This is the designed behaviour of the ordering control:
rescoring the three passages dense retrieval already selected cannot promote a
fourth, so the prompt contains the same three passages in a possibly different
order. The arm therefore separates an ordering effect from a genuine
retrieval-quality effect, and the measurement confirms the separation holds.

The two partitions agree to three decimal places throughout, so none of these
quantities depends on the particular allocation.

---

## 3. Week 3 — the pipeline

### 3.1 What one round does

```
for each round r = 1 … R:
    for each client i = 1 … 8:
        receive the global adapter
        take K = 32 local optimiser steps on client i's own questions
        return the updated adapter
    FedAvg: average the eight adapters
    if r is on the evaluation schedule: score the global model centrally
```

One round is therefore 8 × 32 = 256 optimiser steps, each consuming an effective
batch of 8 examples.

### 3.2 Design decisions that carry the measurement

**Fixed communication budget, no early stopping.** Every condition spends the
same R, N and payload. The comparison is not *which condition converges first*
but *what the same budget buys*. Allowing conditions to stop at different points
would make the budget itself a varying factor.

**K, S and R are injected from the Week 1 calibration, not configured by hand.**
The pipeline reads them from the gate artefact, so the numbers used in the main
runs are the numbers the gate justified.

```
S  =  (questions per client ÷ effective batch) × passes  =  1,892
R  =  S ÷ K  =  1,892 ÷ 32  =  59
```

**Evaluation is central, greedy, and held at `d = 0` for every condition.** The
server scores the averaged global model on a fixed held-out set with greedy
decoding. Holding the evaluation context at `d = 0` separates *what the model
learned* from *what a better inference-time context supplies*; evaluating a
`d = 50` run at `d = 50` would also raise its score, but by giving it better
documents at test time, which has nothing to do with communication.

**Answer F1 is primary, EM recorded alongside.** Week 5 reads "the round at which
a condition first reaches a target `τ`" by linear interpolation. F1 is
continuous and interpolates stably; EM is a step function and would make the
rounds-to-target estimate jump.

**Optimiser state persists across rounds** — resetting the AdamW moments every
round would make each client restart from scratch, which is not how federated
training behaves.

**One model instance serves all eight clients,** with parameters swapped in and
out. Eight copies of a 135M model differ only in the adapter, and memory would
otherwise cap the feasible experiment size.

**The retrieval cache is keyed by partition seed,** so a partition can never be
paired with another partition's cache.

### 3.3 train / val / test

The three sets exist as files and are disjoint.

| Set | Size | Used for |
|---|---|---|
| **val** | 2,000 | The Week 1 gate decisions — K, R and the depth ladder were chosen here |
| **test** | 1,000 | The final comparison only; no decision uses it |
| **train** | remainder | Distributed to the eight clients |

The separation is enforced in code: the evaluation set is built with the val
question ids excluded, and the evaluation question ids are then removed from
every client's training examples. The point of the val set is precisely that
Week 1's choices were made on questions that are not in the set used for the
final measurement, so no pilot decision can inflate the reported result.

### 3.4 Communication accounting

```
total bytes  =  R × N × (|A| + |B|) × bytes-per-parameter × 2
             =  59 × 8 × 921,610 × 2 × 2
             =  1.7400 GB per run
```

`|A| + |B|` is a **sum**: a LoRA adapter transmits both factor matrices, so its
payload is their element counts added, not multiplied. At rank 8 over the
attention projections of SmolLM2-135M that is 921,610 parameters — 1.84 MB at
fp16, and 0.7 % of the base model. The pipeline recomputes this total
independently from the per-round log and checks that the two agree.

This is the quantity the project trades. Established levers — PEFT,
quantisation, sparsification — shrink the per-round payload; this project
attacks `R`, the number of rounds.

### 3.5 Verification run

One full run at `d = 10`, seed 1, to establish that the pipeline produces
records the analysis can use.

| Quantity | Value |
|---|---|
| Rounds completed | 59 / 59 |
| Answer F1 | 0.54 (untrained) → **28.94** |
| EM | 21.90 |
| Communication | **1.7400 GB**, matching the formula independently |
| `promotion_rate` | 0.4201, constant across rounds |
| `rank_change_rate` | 0.677 |
| Client-update divergence | 0.343 → 2.476 |
| Curve points available for interpolation | 21 |
| Early stopping | none |

Three diagnostics confirm the apparatus rather than the hypothesis:

- `promotion_rate` is non-zero at `d = 10`, so the treatment took effect, and it
  is constant because the retrieval cache is frozen — a change would indicate a
  defect.
- The byte total matches the budget formula computed independently.
- Client-update divergence is measurable and grows over training, which is the
  precondition for Week 5's mechanism check: the hypothesis is that better
  retrieval makes clients learn the same task and therefore diverge less.

### 3.6 A constraint this run established

The learning curve is **still rising at round 59**. The target accuracy `τ` in
Week 5 is therefore confined to the observed range and is never extrapolated
beyond it; regions a condition never reaches are reported as such rather than
estimated.

---

## 4. Status and what follows

Weeks 2 and 3 are complete. The substrate is frozen, the pipeline is verified,
and the three data splits are files on disk.

Week 4 executes the main runs on this apparatus: **4 depths × 4 seeds = 16 runs**,
each on the same 1.7400 GB budget, 27.8 GB of accounted communication in total.
The seed count is even by requirement — seeds alternate between the two topic
partitions, so an even count uses both equally and keeps the depth effect
separate from the allocation.

Week 5 converts the sixteen curves into the exchange rate: rounds saved at a
given target, the communication those rounds represent, and the reranking FLOPs
paid for them — reported as an operating region rather than a single verdict.

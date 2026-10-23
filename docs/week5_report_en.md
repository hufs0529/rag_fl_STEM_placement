# Week 5 — Did the Reranking Compute Buy Back Communication?

**Project:** Buying Back Communication with Retrieval Compute — The Exchange Rate
of Client-Side Reranking in Federated LLM Fine-Tuning
**Covers:** Subtask 1.4

---

## Summary

**The question.** Communication is the dominant cost in federated learning. The
project's hypothesis is that **clients trained on better documents reach a target
in fewer rounds**, and fewer rounds means fewer bytes. Choosing better documents
costs compute. So:

> **How much communication did the extra compute buy back?**

**What was done.** Week 4's 24 learning curves were read to measure, per depth,
how many rounds it took to reach a given score; the difference was converted to
bytes and divided by the reranking compute.

**What came out.** **It did not buy any back.** And the link that fails is
identified.

| Link in the hypothesis | Result |
|---|---|
| deeper reranking → better documents in the prompt | ✅ **works** — new documents 0 % → 52.5 % |
| better documents → the model learns better | ❌ **breaks** — every difference sits inside the measurement error |
| learns better → fewer rounds | ❌ the control condition "saved" rounds most often |
| fewer rounds → less communication | — nothing to measure |

**The single most important finding.** `d` = 3 is a condition that **cannot change
the documents** — it reorders the three already chosen. Yet it scored higher and
saved rounds more often than every condition that really did change documents.
**When the arm that changes nothing wins, what the wins measure is the variation
of the measurement, not the effect of the treatment.**

**This is not a failed experiment.** The conclusion is not *"reranking is
useless"* but *"at this model scale and this budget, reranking does not buy back
communication"* — and showing **where the chain breaks** is the contribution.

---

## 1. What this report reads

Only the 24 learning curves Week 4 left on disk, `results/logs/runs/*.jsonl`.
Nothing is retrained. Changing a cost assumption or re-sweeping the target
re-reads the logs — that separation is why Week 3 wrote everything to files.

| | |
|---|---|
| Input | 24 curves (6 depths × 4 seeds) |
| Targets | **63**, all of them used |
| Output | 10 exchange rates, a break-even surface, a mechanism verdict |
| Verdict | **`within_noise`** — no saving is resolved by this measurement |

---

## 2. What the experiment changed

### 2.1 What depth does

Every condition puts **three documents** in the prompt. The only thing that
changes is **how many candidates were examined to pick them**.

| `d` | What it does | New documents |
|---|---|---|
| 0 | the three retrieval chose, unchanged | 0 % |
| **3** | **re-examines those three and reorders them** | **0 %** |
| 7 | examines the top 7, picks 3 | 36.2 % |
| 10 | 3 out of 10 | 42.0 % |
| 25 | 3 out of 25 | 49.8 % |
| 50 | 3 out of 50 | 52.5 % |

"New documents" is the share of the three the original retrieval **would not have
chosen**. It rises **monotonically** with depth — the treatment took effect as
designed.

**But it saturates.** Raising depth **five-fold**, from 10 to 50, raises the share
from 42 % to 52.5 % — only **a quarter more**. The documents the cross-encoder
will promote sit near the **front of the candidate list**; digging further down
is mostly wasted.

> This saturation matters later. **Compute grows in proportion to depth while the
> dose it buys flattens** (Section 8).

### 2.2 A slot changing is not the same as a document changing

The three prompt slots are counted two ways.

| Counted | Meaning |
|---|---|
| **Slot changed** | a **different document** sits there (reordering counts) |
| **New document** | that document is one the **original retrieval would not have chosen** |

The gap between them is what matters.

| `d` | Slots changed | New documents | Gap |
|---|---|---|---|
| 3 | 45.0 % | **0 %** | **45.0 pp** |
| 7 | 64.8 % | 36.2 % | **28.6 pp** |
| 10 | 67.6 % | 42.0 % | 25.6 pp |
| 25 | 71.2 % | 49.8 % | 21.4 pp |
| 50 | 72.4 % | 52.5 % | **19.9 pp** |

**That gap is the original three in a different order.** At `d` = 7, 64.8 % of
slots changed but only 36.2 % hold a new document; the remaining **28.6 points is
reordering**.

And the gap narrows with depth, 45.0 down to 19.9 points. So:

> **Shallow treatment is substantially reordering; only at depth does replacement
> dominate.**

(Both figures are averaged over all rounds. The new-document share is **identical
at every round**, because the retrieval cache was frozen in Week 2 — a change
there would be a defect.)

That is exactly why `d` = 3 exists as a control. It is the condition where 45.0 %
of slots change while 0 % are new — a condition that does **nothing but reorder** —
so the reordering component can be subtracted from the treatment effect.

### 2.3 `d` = 3 — this measurement's ruler

Picking three out of three gives **those three**. A fourth cannot arrive. Only the
order changes.

```
d = 0 :  [A] [B] [C]
d = 3 :  [C] [A] [B]      <- same documents, different order
d = 50:  [A] [X] [Y]      <- new documents arrive
```

A condition that **has no reason to score differently** was included deliberately.
It scored differently anyway.

```
F1 difference between d = 3 and d = 0  =  +0.31
```

**Two conditions with identical prompt content differ by 0.31 points.** That is
the size of the variation that arises **when nothing is changed** — the same way
a person retaking one exam scores slightly differently.

> **That 0.31 is the scale this whole report is read on.** A treatment effect near
> 0.31 is the size of the effect of changing nothing.

### 2.4 Evaluation was held at `d` = 0 for every condition

A model trained at `d` = 50 is **scored on `d` = 0 documents**. The logs confirm
it happened: evaluation document quality was identical to three decimals
regardless of depth.

**Why it has to be.** Reranking can raise a score by two routes, and they are not
the same thing.

| Route | How | Relation to communication |
|---|---|---|
| **Training** | study with better documents → a better model | **Direct.** A better model needs fewer rounds, and rounds are bytes |
| **Serving** | receive better documents at test time → better answers from the same model | **None.** It needs the reranker on every query forever and is unrelated to round count |

Scoring each condition at its own depth would **sum the two inseparably**. A
`d` = 50 run scoring higher could be scoring higher **because the model improved
or because it was handed better documents at test time**, and nothing would say
which.

And the serving route almost certainly does raise the score — Week 2 measured
document quality 3.92 points higher at `d` = 50. **Folding that in would have
manufactured the conclusion that reranking reduces communication**, while nothing
had been reduced.

**The cost, and where it goes.** The scores reported here are lower than a
deployed system would show. That is deliberate. The serving gain is not discarded
either — the cost model in Section 4 **prices it as compute per query**.

---

## 3. How many rounds were saved

### 3.1 How it is measured

Communication is proportional to rounds (**one round = 29.5 MB**), so "how much
was saved" and "how many rounds fewer" are the same question.

A target score is fixed, and each condition's **round of first arrival** is
compared against the baseline.

```
baseline rounds  −  this depth's rounds
     positive → fewer rounds (a gain)        negative → more rounds (a loss)
```

The arrival round is read by drawing a straight line between the evaluated
points. LoRA converges in few rounds, so an integer answer would discard
everything between round 9 and round 10 — which is where the differences live.

### 3.2 How the target is chosen — it is not

Choosing one target makes the verdict an artefact of the choice, and that risk
has already materialised here: at the single target 25.02, `d` = 25 was the only
gain, and that one value flipped the verdict. Over the full range the same depth
loses.

So the target is **not chosen**. Every score the baseline actually recorded is
used.

```
baseline (d = 0) runs 4 × evaluations 20  =  80 scores
```

Two groups are removed.

| Removed | How many | Why |
|---|---|---|
| **below 18.13** | 3 | the baseline had **already passed it by round 1**; every condition arrives at once and nothing can be compared |
| **above 28.14** | 14 | some seed **never arrives**, even after the full budget |

**80 − 17 = 63.** Both bounds are set by the data too — the lower is the highest
of the four seeds' round-1 scores, the upper the lowest of their peaks. **No
number here was chosen by hand.**

### 3.3 The result

Each of the 63 targets was scored win or loss, and the wins counted.

| `d` | Targets beaten | Of those, beyond the error | Typical (median) |
|---|---|---|---|
| **3** *(control)* | **41 / 63** | **26 / 63** | **0.15 rounds faster** |
| 7 | 19 / 63 | 10 / 63 | 0.56 rounds slower |
| 10 | 20 / 63 | 16 / 63 | 0.21 rounds slower |
| 25 | 24 / 63 | 9 / 63 | 0.11 rounds slower |
| 50 | 15 / 63 | 7 / 63 | 0.37 rounds slower |

**"Beyond the error"** — with four seeds the measurement itself moves by ±0.2 to
±0.9 rounds. A margin smaller than that is not a win.

**"Typical"** is the **middle value** of the 63, not the mean. As the target rises
the curve flattens — 9 rounds at 25.02, 38.7 at 28.14 — so a small score
difference becomes ten rounds and values swing to ±11. That tail is **not
trimmed**, because deciding where to cut would be another arbitrary choice;
instead the figures are read with a statistic extremes cannot move.

### 3.4 What the table says

**The control comes first.** `d` = 3 cannot change the documents, yet it beat the
baseline at 41 of 63 targets, and its 26 clear wins are the most of any condition.
The four conditions that really did change documents managed 15 to 24, and are
**all typically slower**.

> When the arm that changes nothing wins most often, what the wins measure is the
> **variation of the measurement**, not the effect of the treatment.

### 3.5 A second check that uses no interpolation

Arrival rounds are read between evaluated points, so the reading itself could be
doubted. Scores were therefore also compared **directly at equal communication
spent**.

| `d` | Points where the difference separates from zero |
|---|---|
| **3** *(control)* | **8 / 20** |
| 7 | 4 / 20 |
| 10 | **8 / 20** |
| 25 | 5 / 20 |
| 50 | 6 / 20 |

The control separates **as often as `d` = 10**. A reading that never touches the
interpolation reaches the same conclusion.

---

## 4. How much compute was spent

Cost has two terms.

```
one-off     indexing   = embed the whole corpus, once
recurring   reranking  = queries × depth × compute per pair
```

| `d` | Compute per query | On this CPU |
|---|---|---|
| 3 | 1.63 × 10¹⁰ FLOPs | 0.15 s |
| 7 | 3.81 × 10¹⁰ | 0.34 s |
| 10 | 5.44 × 10¹⁰ | 0.49 s |
| 25 | 1.36 × 10¹¹ | 1.23 s |
| 50 | 2.72 × 10¹¹ | **2.46 s** |

Directly **proportional to depth**. The per-pair figure was measured in Week 4
and agrees with the analytic estimate within 2×.

**The one-off term cannot separate the conditions**, because it is the same at
every depth. That is **why reranking depth is the treatment axis** — depth is a
**recurring** per-query cost, commensurable with the per-round communication it is
traded against.

**Both** assumptions are reported; which holds is a deployment question.

| Assumption | Meaning |
|---|---|
| `amortised` | RAG is already deployed, the index exists → one-off term is 0 |
| `with_indexing` | the index is built for this training run → one-off term charged |

---

## 5. The exchange rate — one number with no assumptions in it

```
exchange rate = bytes saved ÷ extra compute spent
```

No bandwidth, no device. A particular deployment's verdict is this number
multiplied by that deployment's parameters.

| `d` | Bytes saved | Rate (index exists) | Rate (index built) |
|---|---|---|---|
| 3 | **−17.8 MB** | −1.09 × 10⁻⁸ | −6.57 × 10⁻⁹ |
| 7 | **−60.6 MB** | −1.59 × 10⁻⁸ | −1.24 × 10⁻⁸ |
| 10 | **−30.4 MB** | −5.59 × 10⁻⁹ | −4.66 × 10⁻⁹ |
| 25 | +2.9 MB | +2.14 × 10⁻¹⁰ | +1.98 × 10⁻¹⁰ |
| 50 | **−4.3 MB** | −1.59 × 10⁻¹⁰ | −1.53 × 10⁻¹⁰ |

**Negative means compute was spent and more bytes were spent too.** Eight of the
ten are negative.

**The two assumptions agree in sign at every depth**, so the conclusion does not
depend on whether the index already exists. Charging for indexing shrinks the
magnitude, not the direction.

The one positive, at `d` = 25, comes from a condition that Section 3.3 shows wins
only 24 of 63 targets and typically loses.

---

## 6. Break-even — on what link would it pay off?

Multiplying the exchange rate by a deployment's parameters gives a common unit:
**seconds**.

```
transfer time saved = bytes saved ÷ bandwidth
compute time added  = total compute ÷ device speed
```

Reranking pays off on **slow** links — a slow link makes bytes expensive, and
bytes are what it is buying back.

| Queries | Break-even bandwidth |
|---|---|
| 1,000 | **0 Mbps** |
| 10,000 | 0 Mbps |
| 100,000 | 0 Mbps |
| 1,000,000 | 0 Mbps |
| 10,000,000 | 0 Mbps |

**Zero means no such link exists.** Normally this table reads *"pays off below X
Mbps"*. When the bytes saved are negative, no link is slow enough, so no `X`
exists.

---

## 7. Where the chain breaks

### 7.1 The path that was proposed

```
depth ↑ → document quality ↑ → clients learn the same task
        → update divergence ↓ → rounds needed ↓
```

The middle term, **divergence**, is how differently the eight clients move. If
they scatter, the server's average is blurred and more rounds are needed — that
was the hypothesis.

### 7.2 Divergence barely moved

| `d` | Treatment strength | Divergence |
|---|---|---|
| 0 | 0.00 | 1.3234 |
| 3 | 0.45 | 1.3218 |
| 7 | 0.65 | 1.3143 |
| 10 | 0.68 | 1.3151 |
| 25 | 0.71 | 1.3162 |
| 50 | 0.72 | **1.3132** |

Treatment strength changed **a great deal**, from 0 to 0.72, while divergence
moved from 1.3234 to 1.3132 — **0.77 %**. Nor does it fall in order of depth:
`d` = 7 is lower than `d` = 10 or `d` = 25.

(Divergence is averaged over the **first 10 rounds**, the window where an effect
on the direction of learning would surface first.)

### 7.3 The correlations must not be read at face value

```
treatment strength ↔ divergence reduction    +0.91
divergence reduction ↔ rounds saved          −0.88
```

**The first line alone looks like support for the hypothesis.** It is the most
misreadable number in this report, and the two lines have to be read together.

**(i) It is a correlation between quantities that hardly move.** The total change
is 0.77 %. A correlation states a direction and says nothing about magnitude.

**(ii) The next link has the wrong sign.** The hypothesis says *less drift, fewer
rounds*, which requires a **positive** value. It is **−0.88**. The depths whose
drift fell most spent the most **extra** rounds.

> The first link changes by too little to measure; the second runs backwards. The
> proposed path is not supported.

### 7.4 Verdict

```
within_noise — the largest saving is 0.15 rounds at d = 3 (the control), clearing
               the error at only 26 of 63 targets; no depth resolves a saving at a
               majority of targets
```

The rule states in one line — **a saving counts as observed only when it exceeds
its own error at more than half the targets (32 of 63).** No single point can
overturn it.

---

## 8. The trade-off, depth by depth

Putting the four axes together shows the shape.

| `d` | Compute per query | New documents | F1 (vs control) | Rounds saved |
|---|---|---|---|---|
| **0** | 0 s | 0 % | −0.31 | — |
| **3** *(control)* | 0.15 s | 0 % | — | +0.15 |
| **7** | 0.34 s | 36 % | −0.24 | −0.56 |
| **10** | 0.49 s | 42 % | +0.29 | −0.21 |
| **25** | 1.23 s | 50 % | −0.20 | −0.11 |
| **50** | 2.46 s | 53 % | −0.03 | −0.37 |

**(a) Does more depth consistently improve performance — no.** Against the
control, F1 runs −0.24, +0.29, −0.20, −0.03: **not even consistent in sign**, and
all four treatments save negative rounds.

**(b) Where do diminishing returns appear — in document quality, not
performance.** There is no performance return to diminish. The diminishing return
is the one in Section 2.1: five times the depth buys a quarter more new documents,
while **compute rises in proportion to depth** (0.34 → 2.46 s, **seven-fold**).
Cost linear, effect saturating. The knee is between `d` = 10 and `d` = 25.

**(c) Which depth is the best trade-off — `d` = 0.** No depth returns what it
costs. `d` = 50 spends 2.46 s on every query in perpetuity and spends 4.3 MB
**more** communication.

---

## 9. The conditions this conclusion holds under

The conclusion is not *"reranking is useless"* but *"under these conditions
reranking does not buy back communication"*. The conditions are stated.

| Limitation | Detail |
|---|---|
| **The model is small** | one model, SmolLM2-135M. A model this size may lack the capacity to exploit better documents |
| **Four seeds** | the measurement moves by ±0.16 to ±0.47 points. Resolving a 0.3-point effect would take sixteen seeds — 64 runs, 90 hours |
| **The curves have not converged** | the budget was fixed at 59 rounds, and most curves peak at round 50 or later |
| **Six depths** | 5, 20 and 30 were not measured: every observed effect sits inside the 0.31-point variation the control sets, and a finer ladder cannot read a quantity below the smallest division |
| **An assumed device speed** | break-even bandwidth scales with it, but since the bytes saved are negative no value changes the sign |

**The apparatus itself worked.** New-document share rose monotonically from 0 to
52.5 %, the control held at 0, all 24 runs spent an identical 1.74 GB, and the
evaluation documents were identical to three decimals across every condition. The
link that breaks is **"better documents → learns better"**.

---

## 10. Status

Subtask 1.4 is complete. The ten exchange rates, the break-even surface and the
mechanism verdict are in `results/logs/analysis/exchange_rate.json`, and the four
figures and the final report are **generated** from that file. The numbers are not
transcribed, so re-running the generation after any re-analysis carries them
through.

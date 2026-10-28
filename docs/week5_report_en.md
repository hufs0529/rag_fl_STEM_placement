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
**When the arm whose documents cannot improve wins, the wins are not the effect of
better documents.** Section 2.3.1 computes the treatments against **both**
`d` = 0 and `d` = 3 as baselines and finds all nine contrasts non-significant.

**The scope of the conclusion.** It is not *"reranking is useless"*. Precisely, it
is **"at this model scale and this budget, the fewer-rounds path alone cannot buy
back the reranking compute"**. Two things must be read with it — (i) this report's
cost accounting charges reranking's deployment **cost** but credits its deployment
**benefit** (serving accuracy) as zero (Section 2.4), and (ii) with four seeds
`p` < 0.05 is structurally unreachable (Section 7.5). Showing **where the chain
breaks** is the contribution.

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

A condition whose **documents cannot differ** was included deliberately. It scored
differently anyway.

```
F1 difference between d = 3 and d = 0  =  +0.31   (standard error 0.37)
```

**But that +0.31 is itself indistinguishable from zero.** Its standard error,
0.37, is larger than the value, and `p` = 0.46. The 95 % interval is
**[−0.87, +1.49]** — the true value could be 0, or it could be 1.5.

> **So this 0.31 is not used as a ruler for "the size of the noise".** A quantity
> that cannot be told apart from zero cannot calibrate another one. On a scale:
> an empty envelope reads 0.31 g, but the needle wanders by ±0.37 g on its own.
> What has been learnt is not the envelope's weight but **the needle's wander.**

**What `d` = 3 does tell us is something else.** It cannot change the documents
but it does change **their order** — which is why Section 2.2 calls it the
"reordering component" — and a language model can respond to the order in which
the same documents sit (Liu et al., *Lost in the Middle*, TACL 2024). So
+0.31 ± 0.37 is **an estimate of the ordering effect**, and that estimate is
indistinguishable from zero. One reading follows: **at this resolution the
ordering effect is not resolved either.**

(The **sign** of the ordering effect cannot be fixed from this data, which is why
+0.31 is not called an "upper bound on the noise" — had ordering cost points, the
sampling error would have been **larger** than 0.31, and there is no way to tell
which happened.)

### 2.3.1 So what is the ruler? Each comparison's own standard error

**Resolution is a comparison's own standard error, not another comparison's point
estimate.** Pairing by seed cancels the component that moves a whole run's round
count up or down (0.879 rounds unpaired against 0.16 – 0.47 paired — the same data
resolves 2 to 5 times finer).

**And the critical value is not two.** Four seeds give 3 degrees of freedom, and
the two-tailed 95 % value is **`t` = 3.182**. An earlier draft of this report used
"twice the standard error", a rule 1.6 times too lenient.

```
decision threshold  =  3.182 × (that comparison's standard error)  =  0.52 – 1.48 points
largest difference observed                                        =  0.60 points
```

**So whichever baseline is used, no contrast clears its threshold** (paired
*t*-tests):

| Treatment | vs `d` = 0 | vs `d` = 3 |
|---|---|---|
| `d` = 7 | +0.07 ± 0.16 (`p` = .685) | −0.24 ± 0.27 (`p` = .442) |
| `d` = 10 | +0.60 ± 0.28 (`p` = .127) | +0.29 ± 0.25 (`p` = .329) |
| `d` = 25 | +0.11 ± 0.47 (`p` = .825) | −0.20 ± 0.42 (`p` = .667) |
| `d` = 50 | +0.28 ± 0.31 (`p` = .437) | −0.03 ± 0.45 (`p` = .947) |

**This table steps around the ordering argument rather than settling it.** It does
**not claim** that using `d` = 3 as the baseline makes the ordering effect cancel —
that cancellation would require order and content not to interact, and the
positional effects *Lost in the Middle* reports depend on **which** document sits
**where**, so there is no reason for the assumption to hold. Instead both
baselines are computed and **neither resolves anything.** A check, not an
assumption.

> **And `d` = 10's +0.60 is why this was rewritten.** Under the old rule
> (2 × standard error = 0.56) that value read as **"significant"**. The correct
> threshold is 0.91 and `p` = 0.127. The direction of the conclusion did not
> change, but the rule was wrong.

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
deployed system would show. That is deliberate.

**One thing must be stated plainly, though: the serving gain is not on the
books.** Sections 5 and 6 charge reranking's deployment **cost** (compute per
query) but set its deployment **benefit** (better answers) to zero, because this
experiment did not measure it.

> **So this report's accounting is one-sided, and it is biased against
> reranking.** Section 6 sizes that bias and Section 9 records it as a limitation.
> The conclusion has to be read with it: what this report can say is **"the
> fewer-rounds path alone cannot buy the compute back"**, and **not** "reranking
> is uneconomic".

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

**Where 80 comes from.** The baseline `d` = 0 was run at four seeds, and each run
is evaluated at 20 fixed rounds (1 to 10, then every fifth, then 59).

```
4 runs × 20 evaluations  =  80 scores      <- every candidate target
```

These 80 are **scores that were actually measured**, not a grid at arbitrary
intervals.

**18.13 — the lower bound.** Look at what the four seeds scored **at round 1**.

| Seed | Round-1 score |
|---|---|
| 1 | **18.13** |
| 2 | 17.38 |
| 3 | 17.89 |
| 4 | 17.15 |

The **highest, 18.13**, is the bound. Set the target lower — say 17.50 — and seed
1 is already at 18.13 after one round, so it **arrives at round 1**, and so does
seed 3. Every condition arrives at once and **there is nothing left to compare**.
18.13 is **the lowest score all four seeds must get past round 1 to reach**.

**28.14 — the upper bound.** Look at the **peak** each seed reached across the
full 59 rounds.

| Seed | Peak |
|---|---|
| 1 | 28.44 |
| 2 | 29.92 |
| 3 | **28.14** |
| 4 | 30.66 |

The **lowest, 28.14**, is the bound. Set the target higher — say 29.00 — and seed
3, which peaks at 28.14, **never arrives at all**. Only two of the four seeds
produce a value and no average can be taken. 28.14 is **the highest score all
four seeds can reach**.

**17 — what is dropped.** Of the 80, seventeen fall outside the two bounds.

| Dropped | How many | Which values |
|---|---|---|
| below 18.13 | **3** | 17.15, 17.38, 17.89 — the round-1 scores of seeds 2, 3 and 4 |
| above 28.14 | **14** | 28.44 to 30.66 — mostly late-round scores from seeds 2 and 4 |

```
80 − 3 − 14  =  63
```

**Both bounds were set by the data.** The lower is the maximum of the four
round-1 scores, the upper the minimum of the four peaks. **Nothing in this
section was chosen by hand.**

### 3.3 The result

Each of the 63 targets was scored win or loss, and the wins counted.

| `d` | Targets beaten | Of those, **past the threshold** | Typical (median) |
|---|---|---|---|
| **3** *(control)* | **41 / 63** | **5 / 63** | **0.15 rounds faster** |
| 7 | 19 / 63 | 0 / 63 | 0.56 rounds slower |
| 10 | 20 / 63 | 1 / 63 | 0.21 rounds slower |
| 25 | 24 / 63 | 1 / 63 | 0.11 rounds slower |
| 50 | 15 / 63 | 5 / 63 | 0.37 rounds slower |

**"Past the threshold"** — the decision threshold of Section 2.3.1, that is **the
difference's own standard error × 3.182**. Four seeds give 3 degrees of freedom,
whose two-tailed 95 % value is 3.182.

An earlier draft counted this column at **one** standard error and printed
26 / 10 / 16 / 9 / 7. That rule was counting margins that cannot be called clear
wins. At the correct threshold **no condition passes more than 5 of the 63.**

**"Typical"** is the **middle value** of the 63, not the mean. As the target rises
the curve flattens — 9 rounds at 25.02, 38.7 at 28.14 — so a small score
difference becomes ten rounds and values swing to ±11. That tail is **not
trimmed**, because deciding where to cut would be another arbitrary choice;
instead the figures are read with a statistic extremes cannot move.

### 3.4 What the table says

**The control comes first.** `d` = 3 cannot change the documents, yet it beat the
baseline at 41 of 63 targets. The four conditions that really did change documents
managed 15 to 24, and are **all typically slower**.

**With no effect at all the win rate should sit near 50 %, about 31 of 63.** Only
the control is above that (41); the four arms that changed documents are **all
below** it (15 – 24).

**And no condition passes the threshold at a majority of targets** — the most is
5 of 63. The distribution of wins is itself indistinguishable from a coin flip.

> When the arm whose documents cannot improve wins most often, the wins are **not
> the effect of better documents** — they are an ordering effect or the variation
> of the measurement. Either way, they are not what raising the depth bought.

### 3.5 A second check that uses no interpolation

The arrival rounds of Section 3.3 are read **between** evaluated points. The
reading itself could be doubted, so a different axis was fixed and the question
asked again.

**The method.** Every condition spends the same bytes at the same round, so
**round N is a point of equal budget**. Comparing scores there needs no
interpolation at all.

| | Section 3.3 | Section 3.5 |
|---|---|---|
| Held fixed | the **score** (a target) | the **budget** (rounds = bytes) |
| Compared | the **round** of arrival | the **score** at that point |
| Interpolation | required | **none** |
| Points | 63 targets | 20 evaluation rounds |

**One point worked through.** Round 1 — where both conditions have spent exactly
29.5 MB.

| Seed | `d` = 0 | `d` = 3 | Difference |
|---|---|---|---|
| 1 | 18.13 | 18.71 | +0.58 |
| 2 | 17.38 | 17.65 | +0.27 |
| 3 | 17.89 | 18.50 | +0.62 |
| 4 | 17.15 | 17.51 | +0.36 |
| | | **mean** | **+0.458** |
| | | **error** | **0.084** |

**"Separates from zero"** means `|mean| > 2 × error`.

```
|0.458|  >  2 × 0.084 = 0.168      ->  separated
```

It says the four seeds differed **consistently in one direction** — the values
cluster between +0.27 and +0.62. By contrast `d` = 3 at round 2 was +0.084 ±
0.213: the error exceeds the difference, so it does not separate. The factor of
two is the usual convention, the point at which a difference this large would
arise by chance about 5 % of the time if the true difference were zero.

**Counting that judgement across the 20 points:**

| `d` | Points where the difference separates from zero |
|---|---|
| **3** *(control)* | **8 / 20** |
| 7 | 4 / 20 |
| 10 | **8 / 20** |
| 25 | 5 / 20 |
| 50 | 6 / 20 |

**The control separates as often as `d` = 10 and more often than the other three
treatments.** A reading that never touches the interpolation reaches the same
conclusion as Section 3.4 — two methods fixing different axes give the same
answer.

---

## 4. How much compute was spent

Cost has two terms.

```
one-off     indexing   = embed the whole corpus, once
recurring   reranking  = Q × depth × compute per pair
```

### 4.1 What `Q` is — two different questions

`Q` is **how many questions have to be reranked**. One question arriving makes the
cross-encoder score `d` candidates one at a time, so the total is `Q × d` pairs.

**Two numbers can go in `Q`, and they answer different questions.**

| | Value | Which question it answers |
|---|---|---|
| **① measured** | **80,720** | *"did the compute this experiment spent pay for itself?"* |
| **② declared** | **10³ – 10⁷** | *"would it pay for itself if deployed?"* |

**① was measured.** Gate 3 fixed 10,090 questions per client × 8 clients. The
rerank scores were frozen into a cache in Week 2, so reranking ran **once per
question** (without the cache it would have run once per pass, 1.5× more, with the
same sign).

**② cannot be measured.** It is a post-deployment quantity and nothing was
deployed. The same system in one hospital serves hundreds of questions a day; in a
search product, millions. So rather than pick one, the range is swept across
orders of magnitude.

> **Both accountings are reported.** An earlier draft put only the declared value
> in `Q` and did not say so in Section 4 — the formula carried one undefined
> symbol, and that symbol drove the conclusion (`break-even bandwidth ∝ 1/Q`).
> That leaves no answer to *"could you not just pick the value that gives the
> answer you want?"* Sections 5 and 6 now print both, and **the two agree.**

**The omitted terms are named too.** Query-embedding cost is independent of depth,
so it cannot separate the conditions and is dropped (the same argument as
indexing). The ANN search that fetches the top `d` from the index grows with depth
but sub-linearly, and **this one omission is not conservative** — it is the only
term by which this model understates the true cost, and that direction makes the
conclusion stronger.

**Compute per pair comes with a condition too.** The 5.437 × 10⁹ figure is
**measured with padding to 256 tokens**, not averaged over real passage lengths.
Measured without padding it is 3.9× smaller, which shrinks the denominator and
raises the break-even bandwidth — **so using 256 is unfavourable to reranking, and
therefore conservative.**

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

## 5. The exchange rate — bytes saved divided by compute spent

```
exchange rate = bytes saved ÷ extra compute spent
```

**No bandwidth and no device speed enter it.** Those are multiplied in at Section
6. A particular deployment's verdict is this number multiplied by that
deployment's parameters.

**`Q` does enter it — and that fixes what this quantity is.**

```
numerator    bytes saved       =  happens **once**, during training
denominator  reranking compute =  accumulates over Q queries after deployment
```

**These are quantities from different episodes.** So this is not a rate in the
sense of "how many bytes one FLOP buys"; it is **a one-off benefit divided by a
lifetime cost**, and **whoever fixes the lifetime (`Q`) fixes the value.** It is a
function of `Q`, not a single number — the magnitude scales as 1/Q.

So the table below **states its `Q`**, and both values from Section 4.1 are given.

**The sign is independent of `Q`.** The denominator is always positive, so the sign
of the rate is the sign of the bytes saved:

| Queries | Rate at `d` = 50 | Sign |
|---|---|---|
| 1,000 | −4.054 × 10⁻⁸ | negative |
| 100,000 | −4.054 × 10⁻¹⁰ | negative |
| 10,000,000 | −4.054 × 10⁻¹² | negative |

**The conclusion holds at any query volume.**

**The numerator is the same statistic as Section 3.3** — the median over the 63
targets, multiplied by the per-round communication (29.49 MB).

| `d` | Bytes saved | Rate (index exists) | Rate (index built) |
|---|---|---|---|
| **3** *(control)* | **+4.41 MB** | **+2.70 × 10⁻⁹** | **+1.62 × 10⁻⁹** |
| 7 | −16.52 MB | −4.34 × 10⁻⁹ | −3.38 × 10⁻⁹ |
| 10 | −6.27 MB | −1.15 × 10⁻⁹ | −9.61 × 10⁻¹⁰ |
| 25 | −3.32 MB | −2.45 × 10⁻¹⁰ | −2.26 × 10⁻¹⁰ |
| 50 | −11.02 MB | −4.05 × 10⁻¹⁰ | −3.90 × 10⁻¹⁰ |

*(Rates at `Q` = 100,000; the artefact records this as
`exchange_rate_query_volume`.)*

**Negative means compute was spent and more bytes were spent too. Eight of the ten
cells are negative, and both positive cells are the control, `d` = 3.**

> **All four arms that really changed documents are negative.** If the only arm
> that saved bytes is the one **whose documents cannot improve**, the saving is not
> what better documents bought.

**This table's signs differ from an earlier draft.** That draft read the numerator
at **a single representative tau** (τ = 25.0185), which made `d` = 3 −17.8 MB and
`d` = 25 +2.9 MB — **the opposite signs to Section 3.3's medians.** Two tables were
using two different statistics. They now use one.

### 5.1 Why the indexing assumption **cannot** change the sign

```
rate = bytes saved ÷ (reranking FLOPs + indexing FLOPs)
                      └── positive ─┘   └── positive ─┘
                            denominator = always positive
```

**The sign is set by the numerator alone.** The indexing FLOPs (1.09 × 10¹⁵) are
positive and are added to the denominator, so that assumption **structurally**
cannot change the direction.

An earlier draft offered *"the two assumptions agree in sign at every depth"* as a
robustness check. **It was arithmetic, not a check** — the numerator is identical
in all ten cells and only the denominator moved, so no other outcome was possible.

**That the conclusion does not depend on whether the index already exists is still
true.** Its grounds are not the data but **the structure of the formula** — and
since data can change while structure cannot, those grounds are stronger.

What the two assumptions do change is **magnitude**, which is the two rate columns
above. Their effect on break-even bandwidth is Section 6.

### 5.2 The measured `Q` gives the same answer

Section 4.1's two accountings, side by side.

| `d` | Bytes saved | `B*` (measured `Q` = 80,720) | `B*` (deployment `Q` = 100,000) |
|---|---|---|---|
| **3** *(control)* | +4.41 MB | **0.0268 Mbps** | **0.0216 Mbps** |
| 7 | −16.52 MB | 0 | 0 |
| 10 | −6.27 MB | 0 | 0 |
| 25 | −3.32 MB | 0 | 0 |
| 50 | −11.02 MB | 0 | 0 |

**80,720 and 100,000 differ by only 24 %, so the two accountings agree.** The
conclusion does not change whether `Q` is read as "what the experiment spent" or
"what a deployment would spend".

And **the only non-zero cell is the control.** Where that region sits is Section 6.

---

## 6. Break-even — on what link would it pay off?

Section 5's exchange rate is bytes divided by compute — a ratio of two unlike
things. To say whether a real deployment gains or loses, both have to be put in
the **same unit**. That unit is **seconds**.

```
transfer time saved  =  bytes saved × 8 ÷ bandwidth      (time to push bits down the link)
compute time added   =  total compute ÷ device speed     (time to run the reranker)
```

The bandwidth at which the two are equal is the **break-even bandwidth**.

```
bytes saved × 8 ÷ B*  =  compute ÷ device speed
             B*  =  bytes saved × 8 × device speed ÷ compute
```

**On links slower than `B*`, reranking pays off.** A slower link takes longer per
byte, so the bytes saved are worth more. On a very fast link bytes are cheap and
there is no reason to spend compute to avoid them.

### Query volume and bandwidth are **declared**, not derived

One thing to be clear about. The target τ in Section 3.2 was **set by the data** —
it is the set of points the baseline curves actually passed through, and both
bounds are derived from them. Query volume and bandwidth, by contrast, are
**declared in the config**.

They have to be. Both describe a **deployment that has not happened**. The same
system serves a few hundred queries a day in a hospital and millions in a search
product; no measurement here can settle which.

So neither is chosen — both are **swept, and the verdict reported as a surface**.
That is the same principle τ follows. The ranges cover orders of magnitude on a
log spacing.

| Axis | Range | What the ends mean |
|---|---|---|
| Bandwidth | 1 – 1,000 Mbps | a constrained link – a datacentre |
| Query volume | 10³ – 10⁷ | a small pilot – a large deployment |

**And the sign does not depend on those ranges.** As below, changing the query
volume ten-thousand-fold **does not change the sign** of the break-even bandwidth,
because the numerator — the bytes saved — does not depend on query volume at all.
At the **four depths that changed documents**, whose bytes saved are negative,
`B*` is zero at every query volume; at the **control `d` = 3**, the only positive
one, `B*` is positive at every query volume. No choice of range changes which
depth falls on which side.

### Why query volume is an axis

**Query volume** `Q` is how many questions the system handles after deployment
(Section 4.1). Reranking runs **on each one**.

```
total compute  =  Q × depth × compute per pair
```

The **bytes saved, by contrast, happen once during training**. So as queries
accumulate the denominator grows while the numerator does not, and `B*` falls
as 1/Q.

### What this accounting does to the conclusion — read this first

**Make `Q` large enough and a one-off saving can never pay for itself, by any
method.**

```
Q = 10³    ->  B* = 2.2 × 10⁰   Mbps   (d = 3)
Q = 10⁷    ->  B* = 2.2 × 10⁻⁴
Q = 10¹²   ->  B* = 2.2 × 10⁻⁹         ->  tends to 0
```

**So part of the verdict "there is no operating region" is a consequence of the
accounting, not a finding.** The numerator is fixed while the denominator grows
without bound in `Q`.

And as Section 2.4 recorded, **this accounting charges the cost without crediting
the benefit.** Reranking at deployment also makes answers better — Week 2 measured
`d` = 50's document quality 3.92 pp above `d` = 0's — and that gain is not on the
books, because evaluation was held at `d` = 0.

> **So this section's conclusion must be read as exactly this:**
>
> **"The fewer-rounds path alone cannot buy back the reranking compute."**
>
> **Not** "reranking is uneconomic." That verdict needs the serving gain, which
> this experiment did not measure. Section 9 records it as a limitation.

Within that scope the conclusion does not depend on the declared values — `Q`
scales the magnitude and **cannot change the sign.**

### Measured — `d` = 50

| Query volume `Q` | Break-even bandwidth (`d` = 50) |
|---|---|
| 1,000 | **0 Mbps** |
| 10,000 | 0 Mbps |
| 100,000 | 0 Mbps |
| 1,000,000 | 0 Mbps |
| 10,000,000 | 0 Mbps |

**Zero means no such link exists.** The bytes saved are negative, so `B*` in the
formula above comes out negative, and a negative bandwidth is not a thing. It is
reported as zero.

**That all five rows are zero is the point.** Total compute scales exactly with
query volume, from 2.72 × 10¹⁴ to 2.72 × 10¹⁸ — **ten thousand times** — and `B*`
is zero throughout, because while the numerator, the bytes saved, is negative
**no query volume changes the sign**. The conclusion does not depend on the range
declared above.

Normally this table reads *"at 1,000 queries it pays off below 1.85 Mbps, at a
million below 0.002"*, narrowing the verdict to **a corner of very slow links and
modest query volumes**. At `d` = 50 there is no corner at all — no link is slow
enough. `d` = 3, 7 and 10 save negative bytes too, so all five rows are the same
for them.

**All four arms that actually changed documents (`d` = 7, 10, 25, 50) look like
this.** Their bytes saved are all negative, so all five rows are zero.

### One condition is not zero — and it is the control

The one positive in Section 5 is the **control, `d` = 3** (+4.41 MB). It produces a
**non-zero `B*`** here. It is reported rather than hidden.

| Query volume `Q` | `B*` (index held) | `B*` (index built) |
|---|---|---|
| 1,000 | **2.161 Mbps** | 0.032 Mbps |
| 10,000 | 0.216 Mbps | 0.028 Mbps |
| 100,000 | 0.0216 Mbps | 0.013 Mbps |
| 1,000,000 | 0.0022 Mbps | 0.0020 Mbps |
| 10,000,000 | 0.00022 Mbps | 0.00022 Mbps |

**And the most favourable cell, 2.161 Mbps, lies inside the declared 1 – 1,000
Mbps range.** A thousand-query pilot on a link slower than about 2 Mbps would, in
arithmetic, break even.

> **But that condition is `d` = 3 — the one that cannot change the documents.**
>
> Picking three out of three admits no new document (Section 2.1). It is not a
> treatment anyone would deploy. **If the only arm that can buy anything back is
> the arm that retrieves nothing, what was bought back is not retrieval.**

**None of the four arms that did change documents has such a cell.** For
`d` = 7, 10, 25 and 50, every one of the 40 bandwidth × query-volume cells is
zero.

> So the conclusion reads: **at no depth that changed the documents, and at no
> link speed or query volume, does reranking pay. The only condition with a paying
> region is the control, which changes no documents and is not a deployment
> candidate.** (And this verdict itself sits inside the one-sided accounting noted
> above.)

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

What the two columns measure, first.

| Column | What it measures | Why this and not depth |
|---|---|---|
| **Treatment strength** | the share of the three prompt slots holding a **different document** (Section 2.2's "slots changed") | depth `d` only says **how far down the list was examined**; it does not say how much the prompt actually changed. Doubling `d` while the slots stay put is not a stronger treatment |
| **Divergence** | how **differently the eight clients' updates point** | the middle link of the hypothesis. Clients learning the same task should move in similar directions |

If the hypothesis holds, **divergence should fall as treatment strength rises**.

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
               the decision threshold at only 5 of 63 targets; no depth resolves
               a saving at a majority of targets
```

The rule states in one line — **a saving counts as observed only when it exceeds
Section 2.3.1's threshold (standard error × 3.182) at more than half the targets
(32 of 63).** No single point can overturn it.

### 7.5 The limit of what this design can resolve

**`within_noise` is not an excuse for "no effect was found". It is the only result
this design can produce** — and that can be written as a number.

Consider a **paired permutation test**, which assumes no distribution. In a paired
design the only permutation is flipping the sign of each seed's difference, so
there are `2⁴ = 16` arrangements. The most extreme outcome — all four seeds the
same sign — has a two-tailed `p` of **2 / 16 = 0.125**.

```
4 seeds (this design)  ->  smallest reachable p = 0.125    p < 0.05  impossible
6 seeds                ->                         0.031    reachable
8 seeds                ->                         0.008    reachable
```

> **With four seeds, `p` < 0.05 is structurally unreachable however clean the data
> are.** And the smallest `p` observed in this experiment is **0.127** — sitting
> essentially on that floor.

**Power says the same.** Detecting an effect at 80 % power would need it to be
**0.67 – 1.96 points**; the observed effects are **0.03 – 0.60 points**. They were
never catchable.

So this section's verdict reads: **not "there is no effect" but "this design
cannot make any effect significant, and six seeds would make it possible".** What
has to change is part of the conclusion.

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
costs. `d` = 50 spends 2.46 s on every query in perpetuity and spends **11.0 MB
more** communication (Section 5, median over the 63 targets).

**(d) And which accounting (a)–(c) sit inside — Sections 2.4 and 6.** The "compute
per query" column is reranking's deployment cost, while its deployment benefit
(serving accuracy) was set to zero. So what this section says is **"the
fewer-rounds path alone cannot justify this compute"**, not "do not use
reranking".

---

## 9. The conditions this conclusion holds under

The conclusion is not *"reranking is useless"* but *"under these conditions
reranking does not buy back communication"*. The conditions are stated.

| Limitation | Detail |
|---|---|
| **The model is small** | one model, SmolLM2-135M. A model this size may lack the capacity to exploit better documents |
| **Four seeds** | the measurement moves by ±0.16 to ±0.47 points. Resolving a 0.3-point effect would take sixteen seeds — 64 runs, 90 hours |
| **The curves have not converged** | the budget was fixed at 59 rounds, and most curves peak at round 50 or later |
| **Six depths** | 5, 20 and 30 were not measured: every observed effect (0.03 – 0.60 points) sits below this design's decision threshold (0.52 – 1.48 points), and a finer ladder cannot read a quantity below the smallest division |
| **The control reorders** | `d` = 3 cannot change the documents but does change their order. Its +0.31 ± 0.37 is **an estimate of the ordering effect**, and it is indistinguishable from zero (`p` = 0.46), so neither the size nor the sign of that effect can be fixed from this data. Isolating it would need a condition that runs **the same three documents in several orders**, which was not measured. The conclusion does not rest on that unknown — not because ordering is assumed to cancel, but because **both `d` = 0 and `d` = 3 were used as baselines and all nine contrasts came out non-significant** (Section 2.3.1) |
| **⚠️ One-sided accounting** | Sections 5 and 6 charge reranking's deployment **cost** and credit its deployment **benefit** (serving accuracy) as zero, because evaluation was held at `d` = 0 (Section 2.4). Make `Q` large enough and a one-off saving can never pay for itself **by any method** (`B*` ∝ 1/Q → 0). So what this report can say is **"the fewer-rounds path alone cannot buy it back"**, not "reranking is uneconomic". Saying the latter would need an experiment that also measures the serving gain |
| **⚠️ The permutation floor** | With four seeds, the smallest two-tailed `p` a paired permutation test can reach is **0.125** (2⁴ = 16 sign arrangements). However clean the data, `p` < 0.05 is **structurally unreachable** — and the smallest `p` observed is 0.127, sitting on that floor. The `within_noise` verdict is the only result this design can produce, and **six seeds would bring it to 0.031** (Section 7.5) |
| **An assumed device speed** | break-even bandwidth **scales** with it. Being only a constant of proportionality, it cannot change the sign — the four depths with negative bytes saved stay at zero for any device speed. But `d` = 25's `B*` scales with it, so a faster device could lift that region into the declared range — about **6×** faster at 1,000 queries, about **600×** at 100,000 (Section 6) |

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

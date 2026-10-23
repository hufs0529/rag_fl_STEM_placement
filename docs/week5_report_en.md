# Week 5 — Exchange Rate, Break-Even and the Mechanism Check

**Project:** Buying Back Communication with Retrieval Compute — The Exchange Rate
of Client-Side Reranking in Federated LLM Fine-Tuning

**Covers:** Subtask 1.4

---

## 1. What Week 5 produced

Week 4's twenty-four learning curves reduced to one question.

> **How much communication did the extra reranking compute buy back?**

| | |
|---|---|
| Input | the 24 curves in `results/logs/runs/*.jsonl`. **Nothing else is read** |
| Targets τ | **63** (18.13 – 28.14 F1), all of them used |
| Cost model | two terms (one-off indexing, recurring reranking), both assumptions reported |
| Exchange rates | 10 (5 depths × 2 assumptions) |
| Mechanism verdict | **`within_noise`** |

Nothing is retrained. Changing a cost assumption or re-sweeping τ **re-reads the
logs** — that separation is why Week 3 wrote everything to disk.

---

## 2. Rounds to target

### 2.1 Why interpolate

Communication is proportional to rounds (one round = 8 clients × 1.843 MB × 2 =
29.5 MB), so "how much was saved" and "how many rounds fewer" are the same
question.

But LoRA adapters converge in few rounds. An integer answer — "reached it at
round 9" — discards everything between 9 and 10, and at this scale the differences
between conditions live there. So the crossing is read by **linear interpolation**
between the two evaluated points that bracket it.

```
d0_s1, measured:  F1 24.10 at round 8,  25.02 at round 9
target 25.018 sits 99.9 % of the way between  ->  8 + 0.999 = 8.999 rounds
```

**Interpolation happens per seed, then seeds are averaged** — never the reverse.
Averaging the curves first would erase the between-seed spread the intervals are
built from.

### 2.2 τ is swept, not chosen

Choosing a single τ makes the verdict an artefact of the choice, and that risk
has already materialised here: at τ = 25.02 alone, `d` = 25 was the only positive
saving at +0.0988 rounds, and that one number flipped the mechanism verdict from
`no_effect` to `consistent`. Across all 63 targets the same depth has a median of
−0.113. **The single point flattered it.**

The candidate targets are **the scores the four baseline runs actually recorded**,
not an arbitrary grid.

```
4 baseline runs × 20 evaluation points  =  80 scores
inside the range [18.13, 28.14]         =  63   <- the grid
```

Both bounds are derived.

| Bound | Value | Rule | Outside it |
|---|---|---|---|
| Lower | 18.13 | highest score the baseline had **already passed by round 1** | every condition arrives in one round; nothing to compare |
| Upper | 28.14 | lowest of the baseline **peaks** | some seed never arrives |

**No number here was chosen by hand.**

### 2.3 All 63 targets

| `d` | Targets beaten | Of those, beyond the error | Rounds saved (median) | Range |
|---|---|---|---|---|
| **3** *(control)* | **41 / 63** | **26 / 63** | **+0.149** | −0.88 ~ +11.35 |
| 7 | 19 / 63 | 10 / 63 | −0.560 | −5.46 ~ +4.84 |
| 10 | 20 / 63 | 16 / 63 | −0.213 | −2.47 ~ +8.16 |
| 25 | 24 / 63 | 9 / 63 | −0.113 | −1.86 ~ +3.14 |
| 50 | 15 / 63 | 7 / 63 | −0.374 | −5.75 ~ +3.91 |

**"Beyond the error"** counts targets where the saving exceeds its own standard
error. With four seeds the measurement itself moves by ±0.2 to ±0.9 rounds, and a
saving smaller than that is not a saving.

**Why a median and a count.** As τ rises the baseline curve flattens — 9.0 rounds
at τ = 25.02, 38.7 at τ = 28.14 — so a tenth of an F1 point becomes ten rounds and
savings swing to ±11. That tail is **not trimmed**: deciding where to cut would
replace one arbitrary choice with another. It is read instead with statistics that
magnitude cannot dominate.

### 2.4 What the table says

**The control comes first.** `d` = 3 rescores the three passages dense retrieval
already chose, so it cannot promote a fourth and its prompt content is identical
to `d` = 0. Yet it beats the baseline at **41 of 63** targets and clears the error
at **26**, ahead of every treatment.

> When the arm that changes nothing wins most often, what the wins measure is the
> **variation of the measurement**, not the effect of the treatment.

And **all four treatments have a negative median.** They spent compute and spent
more rounds.

### 2.5 The matched-budget reading

As a second reading independent of interpolation, F1 is compared directly **at
equal cumulative communication**. At each of the 20 evaluation rounds, the
difference per depth and its standard error:

| `d` | Points separated from zero | Of those, positive |
|---|---|---|
| **3** *(control)* | **8 / 20** | 5 |
| 7 | 4 / 20 | 2 |
| 10 | 8 / 20 | 5 |
| 25 | 5 / 20 | 5 |
| 50 | 6 / 20 | 4 |

The control separates from zero **as often as `d` = 10**. A reading that never
touches the interpolation reaches the same conclusion.

---

## 3. The two-term cost model

```
one-off     C_index  = |corpus| × FLOPs_embed(passage)
recurring   C_rerank = Q × d × FLOPs_ce(query, passage)
```

FLOPs were **measured** in Week 4 (5.437 × 10⁹ per pair, 5.437 × 10⁹ per passage;
both within 2× of the analytic estimate). Whether a figure is measured or
estimated is recorded in the artefact.

**Both** assumptions are reported. Which one holds is a deployment question, so
neither is chosen.

| Assumption | Meaning | `C_index` |
|---|---|---|
| `amortised` | RAG is already deployed, the index exists | 0 |
| `with_indexing` | the index is built for this training run | 1.087 × 10¹⁵ |

**The one-off term cannot separate the conditions.** It is 1.087 × 10¹⁵ at every
depth. That is precisely **why reranking depth is the treatment axis** — depth is
a recurring per-query cost, commensurable with the per-round communication it is
traded against. Retriever capacity would not have been.

---

## 4. The exchange rate

One number with no assumptions in it.

```
exchange rate = bytes saved ÷ extra FLOPs spent
```

No bandwidth, no device. A particular deployment's verdict is this number
multiplied by that deployment's parameters.

| `d` | `amortised` | `with_indexing` | Bytes saved | Extra FLOPs |
|---|---|---|---|---|
| 3 | −1.094 × 10⁻⁸ | −6.57 × 10⁻⁹ | −1.785 × 10⁷ | 1.631 × 10¹⁵ |
| 7 | **−1.591 × 10⁻⁸** | −1.238 × 10⁻⁸ | −6.056 × 10⁷ | 3.806 × 10¹⁵ |
| 10 | −5.593 × 10⁻⁹ | −4.661 × 10⁻⁹ | −3.041 × 10⁷ | 5.437 × 10¹⁵ |
| 25 | **+2.143 × 10⁻¹⁰** | +1.984 × 10⁻¹⁰ | +2.913 × 10⁶ | 1.359 × 10¹⁶ |
| 50 | −1.589 × 10⁻¹⁰ | −1.528 × 10⁻¹⁰ | −4.319 × 10⁶ | 2.719 × 10¹⁶ |

**Negative means FLOPs were spent and more bytes were spent too.** Eight of the
ten are negative.

The two assumptions **agree in sign at every depth**, so the conclusion does not
depend on whether the index already exists. Charging for indexing enlarges the
denominator and shrinks the magnitude; it does not change the direction.

The one positive, +2.143 × 10⁻¹⁰ at `d` = 25, comes from a saving that Section 2.3
shows is positive at only 24 of 63 targets with a negative median.

---

## 5. The break-even surface

Multiplying the exchange rate by a deployment's parameters gives a common unit:
**seconds**.

```
transfer time saved  = bytes saved × 8 ÷ bandwidth
compute time added   = total FLOPs ÷ device throughput
break-even bandwidth = bytes saved × 8 × throughput ÷ FLOPs
```

Reranking pays for itself on **slow** links — a slow link makes bytes expensive,
and bytes are what it is buying back. And the break-even bandwidth falls as query
volume `Q` grows, because the recurring term accumulates while the saving does not.

Measured, at an assumed device throughput of 10¹² FLOP/s:

| Query volume `Q` | Break-even bandwidth |
|---|---|
| 1,000 | **0.000 Mbps** |
| 10,000 | 0.000 Mbps |
| 100,000 | 0.000 Mbps |
| 1,000,000 | 0.000 Mbps |
| 10,000,000 | 0.000 Mbps |

**Zero means no such link exists.** Normally this table reads *"pays for itself
below X Mbps"*. When the saving is negative no link is slow enough, so no `X`
exists.

---

## 6. The mechanism check

### 6.1 The proposed path

```
depth ↑ → retrieval quality ↑ → clients learn the same task
        → update divergence ↓ → rounds needed ↓
```

The three measured quantities side by side:

| `d` | Treatment strength (`rank_change`) | Divergence | Reduction | Saving (median) |
|---|---|---|---|---|
| 0 | 0.0000 | 1.3234 | — | — |
| 3 | 0.4505 | 1.3218 | +0.0015 | +0.149 |
| 7 | 0.6484 | 1.3143 | +0.0091 | −0.560 |
| 10 | 0.6764 | 1.3151 | +0.0083 | −0.213 |
| 25 | 0.7120 | 1.3162 | +0.0071 | −0.113 |
| 50 | 0.7244 | **1.3132** | +0.0101 | −0.374 |

### 6.2 Why the correlations must not be read at face value

```
treatment strength ↔ divergence reduction    corr = +0.9135
treatment strength ↔ rounds saved            corr = −0.6518
divergence reduction ↔ rounds saved          corr = −0.8820
```

The first line alone looks like **strong support for the first link of the
proposed path.** It is the most misreadable number in this report, and two things
have to be read with it.

**(i) Divergence barely moves.** From 1.3234 to 1.3132 — a span of 0.0102, or
**0.77 %**. A correlation between quantities that hardly move states a direction
and says nothing about magnitude. Raising depth from 0 to 50 and treatment
strength from 0 to 0.72 moved divergence by less than one percent.

**(ii) The next link has the wrong sign.** The hypothesis says *less drift, fewer
rounds*, which requires `divergence reduction ↔ rounds saved` to be **positive**.
It is **−0.8820**. The depths whose drift fell the most spent the most **extra**
rounds.

> The first link correlates strongly across a change too small to measure; the
> second link runs backwards. The proposed path is not supported.

### 6.3 Verdict

```
verdict: within_noise

the largest median saving, 0.149 rounds at d = 3, clears its own standard error
at only 26 of 63 targets; no depth resolves a saving at a majority of targets
```

The rule states in one line — **a saving counts as observed only when it exceeds
its own standard error at more than half the targets (32 of 63).** No single point
can overturn it. `depths_with_resolved_saving` is the **empty list**.

---

## 7. Depth against cost, performance and communication

Putting the four axes together shows the shape of the trade-off.

| `d` | FLOPs per query | Seconds per query | Promotion | F1 (vs control) | Saving (median) |
|---|---|---|---|---|---|
| **0** | 0 | 0.000 | 0.0000 | −0.31 | — |
| **3** | 1.63 × 10¹⁰ | 0.147 | 0.0000 | — | +0.149 |
| **7** | 3.81 × 10¹⁰ | 0.344 | 0.3619 | −0.24 | −0.560 |
| **10** | 5.44 × 10¹⁰ | 0.492 | 0.4202 | +0.29 | −0.213 |
| **25** | 1.36 × 10¹¹ | 1.229 | 0.4976 | −0.20 | −0.113 |
| **50** | 2.72 × 10¹¹ | 2.458 | 0.5252 | −0.03 | −0.374 |

**(a) Does increasing depth consistently improve performance — no.** Against the
control, F1 runs −0.24, +0.29, −0.20, −0.03: not even consistent in sign. Every
treatment's median saving is negative.

**(b) Where do diminishing returns appear — in promotion.** There is no
performance return to diminish. Promotion bought per unit of depth falls from
0.0517 at `d` = 7 to 0.0105 at `d` = 50, **4.9-fold**, while cost rises from 0.344
to 2.458 seconds per query, **7.1-fold**. **Cost grows linearly while the dose
saturates.** The knee is between `d` = 10 and `d` = 25.

**(c) Which depth is the best trade-off — `d` = 0.** Every depth spends FLOPs on
every query in perpetuity and buys neither measurable performance nor
communication. Eight of the ten exchange rates are negative, and the break-even
bandwidth is zero at every query volume.

---

## 8. Limitations

The result is **not** "reranking is useless" but "under these conditions reranking
does not buy back communication". The conditions are stated.

| Limitation | Detail |
|---|---|
| **Model scale** | one model, SmolLM2-135M. A model this size may lack the capacity to exploit better passages |
| **Four seeds** | paired standard errors of ±0.16 to ±0.47 F1. Resolving an effect of 0.3 would take sixteen seeds — 64 runs, 90 hours |
| **Fixed budget** | 59 rounds. Most of the 24 curves peak at round 50 or later, so they **have not converged** |
| **Six depths** | `{0, 3, 7, 10, 25, 50}`. 5, 20 and 30 were not measured: every observed treatment effect sits inside the noise floor the control sets (0.31 F1), and a finer ladder cannot read a quantity below the smallest division |
| **Assumed throughput** | 10¹² FLOP/s. Break-even bandwidth scales with it, but since the saving is negative no value changes the sign |
| **Instability at high τ** | interpolation error grows where the curve flattens. Not trimmed; handled with a median and a count |

**The apparatus worked.** Promotion rose monotonically from 0 to 0.525, the
controls held promotion at zero, all 24 runs spent an identical 1.7400 GB, and the
evaluation context was identical to three decimals across every condition. The
link that breaks is **training inputs → what is learned**.

---

## 9. Status

Subtask 1.4 is complete. The ten exchange rates, the break-even surface and the
mechanism verdict are all in `results/logs/analysis/exchange_rate.json`, and the
four figures and the final report are generated from that file. **The numbers are
generated rather than transcribed**, so re-running the generation after any
re-analysis carries them through.

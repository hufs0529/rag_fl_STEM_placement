# Week 1 — Environment, corpus scan and the three pilot gates

**Plan subtasks:** 1.1 (corpus construction, partial) and 1.2 (pipeline, gates)
**Outcome:** validated setup, fixed `K` and `R`, and a recorded go/no-go.

---

## Why gates before anything else

Three free parameters remain after the design is written: whether the base
checkpoint has enough headroom to show *any* retrieval effect, whether reranking
depth actually improves retrieval on this corpus, and what `K` (local steps per
round) should be. Each is cheap to check and expensive to get wrong — a full
20-run experiment built on a checkpoint with 2 F1 of headroom measures nothing.

Each gate has a **predefined failure action**, written down before the
measurement, so the decision is not made after seeing the number.

---

## Step 1 — Scan `psgs_w100` for answer-bearing passages

NQ-open and the DPR Wikipedia split are independent resources, so there are no
gold-passage labels. Questions with no gold passage are discarded; the plan
expects ~60k of 87,925 to survive.

```bash
# Week-1 estimate: sample the corpus, extrapolate the usable-question count
python scripts/scan_corpus.py --sample-fraction 0.05

# Week-2 artefact: the full single pass
python scripts/scan_corpus.py --sample-fraction 1.0
```

### Gold needs two stages, not one

The obvious criterion — **normalised answer-string containment**, the standard
top-k accuracy measure of the dense-retrieval literature [8] — does not survive
being used the way this scan uses it.

DPR applies that criterion to the passages **retrieved for one question**. Used
as a corpus-wide labeller it collapses, because **69% of NQ answers are a single
token** (`1950`, `North`, `25`, `Greek`). Measured on a 20,000-passage slice:

| | |
|---|---|
| passages matching *somebody's* answer string | **19,628 / 20,000 (98%)** |
| questions hitting the 20-gold cap | most of them |

At that rate "gold" means nothing, the answer-free reservoir is starved, and the
corpus would be made of noise.

So a passage is gold for a question only if **both** hold:

1. the normalised answer string occurs in the passage, **and**
2. at least `data.gold_min_question_overlap` (default **2**) distinctive content
   words from the question occur in the passage's title or text.

Content words drop question words and short tokens — leaving `when was puerto
rico added to the usa` as `{puerto, rico, added, usa}` — so the filter cannot be
satisfied by function words alone.

Measured effect of the threshold on the same slice:

| Threshold | False-positive pairs removed |
|---|---|
| `>= 1` | 83.0% |
| **`>= 2`** (default) | **96.8%** |
| `>= 3` | 99.4% |

Concretely: *"The 1950 census of rural Norway recorded growth"* no longer counts
as gold for *"when was puerto rico added to the usa"*, while *"Puerto Rico was
added to the usa territories in 1950"* still does.

Both counts are logged (`gold_pairs` and `rejected_pairs`) so the size of the
correction stays visible, and the threshold is a config value, not a constant.
The full scan put numbers on it: of 21,015,300 passages, **628** contained no
question's answer string at all. Without stage two, 99.997% of Wikipedia would
have been gold for somebody.

### Recall is measured by gold id, not by re-matching strings

Because gold is a two-stage criterion, **scoring retrieval success with stage one
alone does not match it**. Nearly every passage in the corpus — including the
random remainder — contains some question's answer string, so string matching
counts a random passage as a successful retrieval. Measured on the real data:
**1.9%** of three-passage selections score as a hit by accident, **6.6%** for
questions whose answer is a single token. That floor sits under every condition
and shrinks the difference the experiment exists to measure.

`gold_recall_at_k()` therefore takes **passage ids** and checks membership in the
question's own gold list, which already passed both stages. `gold_recall_by_text()`
keeps the loose string version for diagnostics only.

**This buys accuracy at the cost of a bias**, and the bias is stated in the
report: requiring question terms to co-occur skews the retained question set
toward **lexically answerable** questions, and discards questions whose evidence
paraphrases the question rather than repeating it.

The scan is a single streaming pass over 21M passages. It is made tractable by
`AnswerIndex`, which prefilters candidate n-gram start positions by the set of
**answer first-tokens** instead of generating every n-gram of every passage.
The overlap stage costs nothing extra: it reuses the token list the first stage
already normalised.

### Where the passages come from

The canonical `wiki_dpr` dataset is **script-based and cannot be loaded on
`datasets >= 3`** (`RuntimeError: Dataset scripts are no longer supported`).
The default source is therefore a parquet mirror, read directly with
**column projection**:

| | |
|---|---|
| `data.passage_dataset` | `kenhktsui/wiki_dpr_e5` — 157 shards × 133,856 rows ≈ 21.0M, first row `id=1, "Aaron"`, i.e. psgs_w100 |
| `data.passage_columns` | `[id, title, text]` — the mirror's unused e5 embedding column is **90.6% of the bytes** |
| effect | ≈ 86 GB → ≈ 8 GB transferred |

`data.passage_source: datasets` restores the original loader for anyone on
`datasets < 3`. The question set likewise needs its canonical id —
`google-research-datasets/nq_open`; the bare alias `nq_open` no longer resolves.

### The scan survives a dropped connection

The full pass takes hours and the Hub drops connections mid-stream
(`Server disconnected without sending a response`, `read operation timed out`).
`huggingface_hub` retries five times; when it exhausts them the process dies, and
without state on disk **every hour of work goes with it**.

So the parquet path processes **one shard at a time and checkpoints at each shard
boundary** (`data/scan_checkpoint.json`), holding the gold map, the counters, and
the **byte offsets** of both output files. On restart the scan:

1. refuses to resume if the settings changed — the fingerprint covers the
   dataset, shard list, overlap threshold and gold cap, because stitching two
   differently-labelled halves together would be worse than starting over;
2. **truncates both output files to the checkpointed offsets**, discarding the
   half-written records from the shard that died;
3. re-runs that shard and continues.

Two consequences shaped the design:

* **The random reservoir became per-shard.** A single 300k reservoir is ~200MB,
  and writing it at every shard boundary would cost 31GB of I/O. Each shard now
  draws an equal quota instead — stratified sampling, which is uniform across the
  corpus (shards hold equal row counts) and has *lower* variance than a single
  global reservoir.
* **Checkpoints are written atomically** (`.tmp` then `replace`), so dying during
  a checkpoint write leaves the previous one intact.

`tests/test_corpus_scan.py` asserts the property that matters: a scan interrupted
after two shards and resumed produces **byte-identical** output files to an
uninterrupted run. Resume that changes the artefact is worse than no resume.

```bash
python scripts/scan_corpus.py --sample-fraction 1.0      # resumes by default
python scripts/scan_corpus.py --sample-fraction 1.0 --no-resume
python scripts/scan_corpus.py --sample-fraction 1.0 --checkpoint-every 5
```

The checkpoint is deleted on a clean finish, so a completed scan never resumes by
accident.

**Sampling happens at shard granularity, not row granularity.** Striding rows
would download all 157 shards and discard 19 of every 20 — the same hours, none
of the saving. `plan_parquet_shards()` selects every *n*-th shard instead, so
`--sample-fraction 0.05` reads 8 shards (~1.07M passages) and transfers about a
twentieth of the bytes. Shards are ordered by passage id, so an even stride
spans the whole corpus rather than its first slice. Granularity is one shard
(~0.64%), so the **effective** fraction is computed and logged, and the
usable-question extrapolation uses that number rather than the requested one.
Gold passage *bodies* are written out during the same pass, because Week 2's
corpus build and both retrieval gates need them and 21M should not be read twice.

| Output | Contents |
|---|---|
| `data/gold_by_qid.json` | question → gold passage ids (capped at 20/question) |
| `data/gold_passages.jsonl` | the answer-bearing passage bodies |
| `data/random_passages.jsonl` | a uniform random sample of **answer-free** passages |
| `data/questions_usable.jsonl` | questions that have at least one gold passage |
| `results/logs/gates/scan_stats.json` | coverage, histogram, extrapolated usable count |

The 20-gold cap stops questions with very common answer strings (`"1999"`) from
eating Week 2's 200k-passage budget — and it also bounds what gets written.

**Only passages actually recorded are written out.** Measured on a full-question
scan, 66% of passages qualify as gold for *somebody*; writing all of them would
produce ~14M passages (~10GB), which Week 2 then loads into memory whole (~8GB).
A passage rejected by the per-question cap is referenced by nothing downstream,
so the scan writes it nowhere and counts it as `passages_capped_out` instead.
That bounds the file at `questions x max_gold_per_question` pairs rather than at
the size of the corpus. Lower `--max-gold-per-question` to shrink it further.

### The random remainder comes out of the same pass

Week 2's corpus is gold + hard negatives + a **random Wikipedia remainder**. If
that third component were drawn from the passages the scan already kept, every
passage in the 200k corpus would be one that contains *somebody's* answer, and
retrieval difficulty would not be what the design intended.

So the scan also fills a `ReservoirSampler` with passages that matched **no**
question (`data.random_pool_size`, default 300k). Two properties matter:

* **Answer-free by construction.** Only passages with zero matches are offered to
  the reservoir, so the remainder can never contribute a false-positive recall.
* **Uniform, in one pass.** Reservoir sampling (Algorithm R) gives every passage
  in the stream the same chance. Taking "the first 300k" instead would hand the
  corpus a severely skewed topic slice, because a Wikipedia dump is ordered by
  document, not shuffled.

The scan warns if the pool it collected is small relative to the corpus budget —
the remainder would not fill, and the corpus would silently drift back towards
being all-gold.

---

## Gate (i) — Capability headroom · *no training required*

```bash
python scripts/gate1_capability_headroom.py
```

Measures answer F1 with **gold** passages injected versus **random** passages
injected. The gap bounds what any retrieval improvement can possibly deliver.

* **Passes if** gap ≥ ~5 F1.
* **Fails →** escalate to `SmolLM2-360M-Instruct` (`gates.headroom.fallback_model`).

Both conditions place exactly `top_k` passages in the prompt — gold entries are
padded with random passages when a question has fewer than three golds — so
context *length* cannot contribute to the gap.

### The probe needs format demonstrations, or it measures the wrong thing

Run without them, the gate on `SmolLM2-135M-Instruct` produced:

```
gold context   F1  1.19   EM  0.67
random context F1  0.00   EM  0.00
[FAIL] headroom 1.19 F1 < 5.0: escalate to a larger checkpoint
```

Looking at what the model actually emitted explains it — it was not answering at
all:

```
''                                                    <- immediate EOS
'\n\n[2] "International Regulations...": stand on vessel...'   <- continuing the context
```

An untrained 135M model does not know that the expected output is a short
extractive span; it continues the document instead. So the bare probe measures
**whether the model already knows the output format**, not whether it can use the
context — and "escalate to a larger checkpoint" is then the wrong prescription,
because a larger model fails the same way.

The main runs teach the format through answer-token loss masking, so the gate
supplies it instead with `gates.headroom.n_shots` (default **2**) hand-written
demonstrations, each carrying its own one-passage context. Format is held
constant across both arms, leaving the gold-vs-random gap to measure what it was
meant to: **the capacity to ground an answer in the provided context**.

The gate also saves five raw predictions under `sample_predictions`. Numbers
alone cannot distinguish "cannot use context" from "does not know the format";
the samples can.

---

## Gate (ii) — Depth monotonicity · *retrieval only*

```bash
python scripts/gate2_depth_monotonicity.py
```

Verifies gold recall@3 rises monotonically across `d ∈ {0, 3, 10, 50}`.

* **Passes if** monotone *and* strictly increasing overall.
* **Fails →** the treatment axis does not exist; revise the depth set.

A flat curve is a **failure, not a pass** — if depth buys no recall there is
nothing for the main experiment to measure.

The gate also asserts a **control invariant**: `d = 3` must leave recall@3
*exactly* unchanged, because it only reorders the passages dense retrieval
already selected. A difference there is a bug in the selection or cache path,
not a finding, and the script says so explicitly.

### The probe corpus has to look like Week 2's corpus

Because Week 2's Qdrant collections do not exist yet, the gate builds its own
small index. **What goes into it decides what the gate measures**, and the first
version got this wrong in a way worth recording.

It put *every* gold passage of each sampled question into the corpus (the scan
keeps up to 20; the mean is 13.4) and drew the filler from the gold pile as well,
because that was the only passage file around. The result on a 5,000-passage
probe:

```
2,689 of 5,000 passages were gold for the 200 sampled questions   (54%)

already in dense top-3  : 0.9100     <- dense already found everything
recoverable             : 0.0750     <- nothing left for depth to do
[FAIL] corpus is too easy
```

The verdict was right about that corpus and useless about the experiment: **no
one will ever train on a corpus that is half gold.** For a single question,
Week 2 hides **one** gold passage among ~25,000.

The probe now mirrors Week 2's composition instead:

| | first version | now |
|---|---|---|
| gold per question | all of them (13.4) | **1** (`data.gold_per_question_in_corpus`) |
| filler | other questions' gold | **`data/random_passages.jsonl`** |
| size | 5,000 | **25,000** (one client's index) |

That fixed the gold *count* and left the gold *share* wrong, which the first
re-run then measured. With one gold per question and random Wikipedia filler,
only **8%** of a 25,000-passage probe is anyone's answer. Week 2's corpus is
**40.4%** (80,720 gold in 200,000). The difference matters because random
passages are easy to outrank and **another question's gold is not** — it contains
an answer, so it shares vocabulary with questions. The probe now tops the corpus
up with gold passages belonging to questions *outside* the evaluated sample,
until the gold share matches Week 2's:

| | first version | after fix 1 | now |
|---|---|---|---|
| gold per question | all (13.4) | 1 | **1** |
| gold share of corpus | 54% | 8% | **40.4%** (derived, not hardcoded) |
| filler | other questions' gold | random only | **other questions' gold + random** |
| size | 5,000 | 25,000 | **25,000** |

The share comes from `week2_gold_share(n_questions, corpus_size, gold_per_q)`, so
a different scan result moves it automatically.

One invariant carries the correctness of this: **no gold passage of an evaluated
question may enter as filler.** Only one gold per question is kept as the label,
and if another of that question's gold passages sits in the corpus unlabelled,
the model effectively receives a correct passage while recall scores it wrong.
`select_question_gold` therefore forbids the *untruncated* gold list of every
sampled question, not just the one it kept. Because this construction has now
been wrong twice, it lives in `src/probe_corpus.py` with tests rather than inline
in the script.

Measured effect of matching the share, on a 600-passage probe:

| | random filler only | Week-2 gold share |
|---|---|---|
| `already` | 0.8250 | **0.7750** |
| `recoverable` | 0.1500 | **0.2000** |

Harder retrieval, more room for the treatment — which is the point.

BM25 hard negatives are still deliberately **left out**. They push gold further
down the dense ranking, which *increases* `recoverable` — so omitting them keeps
the probe conservative: whatever passes here will do at least as well in Week 2.

The script refuses to run without the answer-free reservoir, because falling back
to the gold pile is exactly the failure above. It uses
the same `search()` signature as the Week-2 `QdrantRetriever`, so the gate code
and the cache-building code are the same code. The gate writes its candidates
and cross-encoder scores to `results/cache/gate_scratch.json` so gate (iii) does
not repeat the embedding and rescoring work.

### What the fixed probe measured, and why passing is not enough

On 300 questions against the 25,000-passage probe, the gate passed:

```
already in dense top-3       : 0.7000   (210 questions)   <- was 0.9100
recoverable by reranking     : 0.1100   ( 33 questions)   threshold 0.10
gold outside the pool        : 0.1900   ( 57 questions)   <- was 0.0150
ceiling                      : 0.8100

d= 0  recall@3 0.7000  rank-change 0.000  promotion 0.000
d= 3  recall@3 0.7000  rank-change 0.380  promotion 0.000
d=10  recall@3 0.7100  rank-change 0.628  promotion 0.394
d=50  recall@3 0.7133  rank-change 0.680  promotion 0.487
```

Two things to read off this.

**The ordering control behaves exactly as designed.** At `d=3` the reranker
changed the order of the selected passages in 38% of questions and promoted a
passage in **0%** of them — recall is identical to `d=0` to four decimals. That
is the property the whole comparison rests on: any later gain at `d=10` or
`d=50` cannot be attributed to reordering, because reordering alone is measured
here to do nothing.

**The effect size is marginal, and the gate's threshold does not catch that.**
The `d=0 → d=50` gain is 0.0133, which on 300 questions is **four questions**.
Treating those four as binomial successes puts the rough standard error at
0.0067, so the gain sits at the 2 s.e. boundary. `recoverable_captured` is
0.121: of the 33 questions reranking *could* have rescued, it rescued 4.

A 1.33 pp retrieval gain is a thin foundation for a downstream claim about
rounds-to-target, because the fine-tuning signal difference will be smaller
still and five seeds will not separate it from noise.

The gate now reports this itself rather than leaving it to be noticed. The
`d=0 → d=max` comparison is **paired** — both depths see the same questions, so
only the questions whose verdict flipped carry information about the difference:

```
d=0 -> d=50 paired: +4 / -0 questions of 300  delta +0.0133 +- 0.0067  z 2.00
```

`paired_recall_delta` counts `gained` and `lost` and reports
`sqrt(gained + lost) / n` as the standard error (the McNemar structure). When
`|z| < 2` the script prints a warning, because monotonicity alone would pass the
gate on an effect indistinguishable from zero. The pass criterion is unchanged —
whether to tighten it is a design decision, not something a diagnostic should
make silently. Two parameters in this
run understate the real headroom, and both are config, not design:

| | this run | should be | effect |
|---|---|---|---|
| `data.gold_per_question_in_corpus` | 2 | **1** | two golds give two chances at the dense top-3, inflating `already` |
| gold share of the probe | 8% | **40.4%** | random filler is easy to outrank, so `already` is overstated |
| `gates.monotonicity.n_questions` | 300 | **2000** | 300 questions resolve the gain only to ±0.007 |

The full record of that run is kept at
`results/logs/gates/gate2_monotonicity_n300_gold2_superseded.json`.

The gold-per-question value was `2` on this branch and `1` from Week 2 onward —
a config that differed by branch while the gate on *this* branch reads it. It is
now `1` everywhere. Note the direction is not one-sided: dropping to one gold
lowers `already` but also raises `out_of_pool`, and `recoverable` is what is left
over, so it has to be re-measured rather than argued.

`out_of_pool = 0.19` is the number to watch. For 57 of 300 questions no gold
passage is in the top-50 at all, so no reranking depth can reach them. If the
re-run keeps it that high, `retrieval.candidate_pool` has to rise above 50 — but
50 is also the largest treatment depth, so that changes the treatment grid and
the cost model together, and is a design decision rather than a tuning one.

To make that decision on evidence, the gate retrieves a **deeper pool than the
treatment uses** and reports what each pool size would buy. The treatment still
sees only the first `candidate_pool` candidates, and no extra reranking happens,
so the diagnostic is nearly free:

```
pool=  50  recoverable 0.2000  out_of_pool 0.0250  ceiling 0.9750  <- in use
pool= 100  ...
pool= 200  ...
```

### Re-running is cheap now

Embedding 25,000 passages costs 37 minutes on 12 CPU cores, and a gate gets
re-run every time a config value is corrected — which has happened repeatedly.
`src/embedding_cache.py` keys corpus vectors on the model name, the passage id
order **and a digest of the passage texts**, so an identical probe reuses them
and any change re-encodes. Silently reusing stale vectors is worse than paying
the 37 minutes, because a wrong answer looks exactly like a right one.

---

## Gate (iii) — Local-step calibration · *short FL probe*

```bash
python scripts/gate3_local_step_calibration.py
```

`R = S / K`. The round count is **not a free parameter** — it is total local
steps divided by local steps per round, and `K` governs how far clients drift
before averaging.

The gate sweeps `K ∈ {4, 8, 16, 32, 64}`, runs a few probe rounds at each, and
measures client-update divergence:

```
div = mean_i ‖ δ_i − δ̄ ‖ / ‖ δ̄ ‖
```

normalised by the mean update so it is comparable across `K`. It selects the
**smallest** `K` whose divergence clears the threshold:

* too small a `K` averages the clients before they diverge at all, so a
  treatment that acts *by reducing drift* would be invisible;
* too large a `K` shrinks `R = S/K` until there is no resolution left to
  interpolate a treatment effect out of.

`S` is then set to 1–2 passes over each client's data, and `R = S / K` follows.
The result is written to **`results/logs/gates/calibrated_schedule.json`**, which
the Week 3 and Week 4 run scripts read directly — so "a pilot fixed these, not
convention" is enforced by the code, not just claimed in the report.

**Measured outcome:** `K = 32`, `S = 1,892`, `R = 59`.

Two details that keep the calibration honest:

* The probe runs at **`d = 0`** (baseline). We need a `K` where drift is
  observable *without* the treatment.
* Optimiser state is carried across rounds (`LocalTrainer`), so small `K` does
  not leave AdamW permanently in warm-up and confuse the effect of `K` with the
  effect of restarting the optimiser. The probe and the Week-3 client share the
  same loop for exactly this reason.

The provisional partition (group by gold-passage title, greedily fill the
smallest client) stands in for Week 2's topic-cluster partition. What the gate
needs is a structure that *produces* drift, not the final allocation.

### The threshold rule was degenerate, so the choice is declared and verified

The gate's stated rule was "the smallest `K` whose divergence clears
`min_divergence`". On the measured data that rule carries no information: all
five `K` land between 0.3343 and 0.5097, which is 17-25x the 0.02 threshold, so
the rule returns the smallest candidate (`K = 4`) whatever the data say. The
output was a PASS that had chosen nothing.

The cause is the normalisation. `divergence = spread / ||mean update||`, and as
`K` grows the denominator grows faster than the numerator, so the normalised
value *falls* while the absolute spread *rises* monotonically. A single
threshold cannot express "the smallest `K` at which drift is real".

Rather than reverse-engineer a rule that happens to output the right answer, the
choice and the check are now separate. `gates.local_steps.chosen_K` declares the
value with its evidence in the config, and `local_step_decision` **verifies** it:

| Check | At `K = 32` |
|---|---|
| divergence ≥ `min_divergence` | 0.3343 ≥ 0.02 |
| absolute spread ≥ 2x that of the smallest `K` | 0.9170 / 0.1762 = **5.2x** |
| `R = S/K` ≥ `min_rounds` | 1,892 / 32 = **59** ≥ 20 |

The spread check is what rules out the small-`K` end: at `K = 4` the normalised
divergence is the highest of the sweep, but the updates are the smallest
(0.4925), so what looks like drift is relative noise. Requiring the *absolute*
spread to be at least twice the smallest `K`'s makes that explicit, and
`K = 8` fails it (1.74x).

With no `chosen_K` declared and a threshold every candidate clears, the gate now
**fails** and says so, instead of returning the smallest value.

`K = 32` over `K = 64`: the round-1 curve turns upward at 64 (0.4334), which is
the clearest sign of task divergence, but that turn rests on a single point and
`R` halves to 29. `K = 32` sits at the flat bottom of the basin (`K = 16` is
0.3376, `K = 32` is 0.3343) with twice the round resolution. Switching to 64 is
a config change plus an update to the evidence comment.

### The schedule is written at main-experiment scale

`S` is `(questions per client / effective batch) x passes`. Two scales exist and
must not be mixed:

| | Questions per client | `S` | `R` at `K = 32` |
|---|---|---|---|
| Gate probe (`--max-questions-per-client`) | 250 | 46 | 1.4 |
| **Main experiment** | **10,090** | **1,892** | **59** |

The probe caps questions per client to keep the gate affordable, so its `S` is
not the experiment's `S`. `calibrated_schedule.json` records the full-scale value
and keeps the probe value beside it under `scale`, and the gate's `min_rounds`
check runs against the full-scale `R` — checking the probe's `R = 1.4` would
reject every `K`.

The effective batch matters here too: with `grad_accumulation = 4`, one step
consumes 8 examples, not 2. Dividing by the minibatch inflated `S` fourfold and
labelled 6 passes as 1.5.


### Snapshots make the sweep cheap, and the nesting is only exact in round 1

The first version restarted from the initial parameters for every candidate `K`,
so a client ran `sum(K) = 124` steps per round. But **within a round every client
starts from the same global parameters**, so a run heading for `K = 64` passes
through step 4, 8, 16 and 32 on the way. Snapshotting the trainable tensors at
those points yields the smaller `K` values for free:

```
0 ----4----8------16----------32------------------64
     snap snap    snap        snap                snap
```

Cost per client per round falls from `sum(K) = 124` to `max(K) = 64`. At the
measured 31 s per step on this CPU that is 25.6 h → 13.2 h. Measurement quality
improves too: all five `K` are read off **one trajectory**, so comparisons
between them are not confounded by different random trajectories.

**The nesting is exact only in round 1.** From round 2 the starting point is a
FedAvg result, and that result depends on `K`, so the sharing breaks. The gate
advances the single global trajectory at `K_max` and reports, for each round,
the divergence reached `K` steps from that round's common start. Round 1 is
therefore the unconfounded per-`K` measurement (recorded separately as
`divergence_round1_only`), and rounds 2–3 answer the question they were there
for in the first place: does the drift survive once averaging has pulled the
clients back together? Running five independent federated trajectories for three
rounds would cost 25.6 h to answer a robustness question — the trade is recorded
here rather than hidden.

### Divergence was structurally pinned at exactly zero

The probe reported `divergence 0.0000` and `cos +1.000` for every `K` in every
round. Exact values like that are a defect, not a small-`K` effect, and the
defect was in the snapshot helpers rather than in the gate:

```python
v.detach().cpu()           # on CPU, .cpu() is a no-op — same tensor
v.to(dtype=torch.float32)  # already float32 — same tensor
.numpy()                   # shares memory with the tensor
```

Every "snapshot" was a **view onto the model's live parameters**. Gate (iii)
reuses one model object and swaps parameters per client, so by the time the loop
finished, all clients' stored arrays showed the *last* client's weights.
Identical vectors give cosine exactly 1 and divergence exactly 0.

`get_trainable_state_dict` now clones and `state_dict_to_numpy` /
`state_dict_to_ndarrays` copy. `tests/test_communication_snapshot.py` pins the
property directly — take a snapshot, train, assert the snapshot did not move —
including the two-clients-sharing-one-model pattern the gate actually uses.

This one is worth dwelling on: **on a GPU the bug is invisible**, because `.cpu()
` copies across devices. The whole Week-5 mechanism check (client-update
divergence as the proposed mechanism) reads these helpers, so it would have
reported a clean zero on CPU and plausible numbers on GPU, with nothing in the
output to say which. A test that asserts the snapshot is detached is the only
thing that catches it in either place.

---

## Memory: why `batch_size` is 2 and not 8

CPU training cannot use bfloat16, so `resolve_dtype` falls back to float32 — a
correct fallback with an expensive consequence. The dominant tensor is not the
model but the **logits**:

```
batch 8 x seq 1024 x vocab 49,152 x 4 bytes = 1.61 GB
```

The stored forward copy, the loss intermediate and the gradient are all live at
once, so the peak is roughly 4.8 GB on top of weights and activations. Measured:
the process reached **7.2 GB** and the kernel's OOM killer took it:

```
Out of memory: Killed process 2073 (python) anon-rss: 7,186,104 kB
```

On WSL2 that is worse than a crashed job — the default memory allocation is half
of host RAM (7.6 GB of 15.7 GB) with 2 GB of swap, and exhausting it destabilised
the distribution enough to drop the editor connection too.

Two fixes, at different levels:

| Level | Change | Effect |
|---|---|---|
| Host | `.wslconfig`: `memory=11GB`, `swap=16GB` | OOM becomes *slow* rather than *fatal* |
| Config | `batch_size: 2`, `grad_accumulation: 4` | peak 4.8 GB → 1.2 GB, **same effective batch** |

`grad_accumulation` was in the config but **`LocalTrainer` never read it**, so
setting it did nothing. It now accumulates, and one *step* still means one
optimiser update, so `K` keeps its meaning. `tests/test_local_train.py` proves
`batch 8 x accum 1` and `batch 2 x accum 4` produce the identical weight update —
without that equivalence, lowering the batch would have changed the experiment
rather than just its memory profile.

Gate (iii) now estimates the peak before training and refuses to start if it
exceeds 75% of available memory, naming the knob to turn. A 13-hour job that
dies at hour 9 costs more than one that refuses in the first second.

Sequence length was checked and is **not** a lever: padding is already dynamic
(longest-in-batch), and the measured prompt length is 515 tokens at the median,
622 at the maximum, so `max_seq_length: 1024` never binds. Thread count is not a
lever either — the i7-1355U has 6 physical cores, and PyTorch's default of 6
threads already matches.

---

## Run all three

```bash
python scripts/run_week1_gates.py               # full scale
python scripts/run_week1_gates.py --dev         # reduced scale, for a smoke test
python scripts/run_week1_gates.py --stop-on-fail
```

Writes `results/logs/gates/week1_summary.json` with the combined **GO / NO-GO**
verdict and each gate's stated action. Exit code is non-zero on NO-GO so the
week's outcome is visible to CI or a shell loop.

---

## Modules introduced this week

| Module | Role |
|---|---|
| `src/config.py` | YAML config, dev overlay, `--set` overrides |
| `src/metrics.py` | NQ normalisation, F1/EM, answer-string containment, recall@3 |
| `src/selection.py` | **the treatment**: `select_passages(candidates, scores, depth)` |
| `src/nq_data.py` | NQ-open loading, `AnswerIndex`, fixed held-out set |
| `src/corpus_scan.py` | streaming `psgs_w100` scan and usable-count extrapolation |
| `src/probe_corpus.py` | gate (ii)'s probe corpus, shaped like Week 2's |
| `src/embedding_cache.py` | content-addressed corpus-embedding cache |
| `src/models.py` | model/dtype resolution **and training-peak memory estimate** |
| `src/prompting.py` | prompt template, answer-token loss masking, left truncation |
| `src/embedding.py`, `src/retrieval.py`, `src/rerank.py` | frozen retrieval stack |
| `src/models.py`, `src/generate.py`, `src/local_train.py` | model, greedy decoding, local loop |
| `src/communication.py`, `src/aggregate.py`, `src/divergence.py` | payload, FedAvg, drift |
| `src/gates.py`, `src/logging_utils.py` | gate decisions and on-disk records |

```bash
pytest -q          # 159 tests, no GPU and no network required
```

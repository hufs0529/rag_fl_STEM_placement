# Research records and data management plan

## Scope of this document

This document is the experimental record for the placement reported in *Buying
back communication with retrieval compute in federated fine-tuning*. It
describes what was collected, when and how; how the data are organised,
verified and archived; what is kept and what is deliberately not kept; and where
the reasoning behind each decision is preserved. It is written so that a reader
who has never seen the project can locate any number in the report and
regenerate it.

Placement period: **16 September 2026 – 28 October 2026** (six weeks).
All work was performed by the author unless stated otherwise.

---

## 1. Overview of the record

| Item | Value |
|---|---|
| Experimental runs | 24 (6 reranking depths × 4 seeds) |
| Main sweep executed | 12 October 2026 – 20 October 2026 |
| Wall-clock compute | 33.9 h total, 1.41 h per run (mean) |
| Per-round logs | 24 JSONL files, 24 JSON summaries, 868 KB |
| Pilot-gate artefacts | 6 JSON files |
| Frozen input artefacts | 7 JSON files (corpus, partitions, indexes, retrieval caches) |
| Analysis artefacts | 1 JSON (`exchange_rate.json`, 300 KB) |
| Figures | 4 PNG |
| Source modules | 37 Python modules under `src/` |
| Analysis and utility scripts | 29 under `scripts/` |
| Automated tests | 333, all passing |
| Commits on the final branch | 51 |

### 1.1 Nothing in this record is simulated or imputed

Every figure and table in the report is read directly from the archived run
logs by a script in the archive. No value was transcribed by hand into the
report without a script that reproduces it, and no missing value was imputed. A
condition that failed to reach a target threshold is recorded as not reaching
it, never extrapolated.

---

## 2. What was collected

### 2.1 Per-round experiment logs — the primary record

`results/logs/runs/d{depth}_s{seed}.jsonl`, one line per event.

The first line is a **manifest** fixing the conditions of that run: depth, seed,
partition seed, rounds, local steps per round, total local steps, client count,
batch size, gradient accumulation, examples per step, the evaluation round
schedule, the early-stopping flag, the backend, the base model, trainable
parameter count, payload bytes per client per direction, and the total
communication budget with the formula that produced it.

Subsequent lines are **round records**: round index, whether the round was
evaluated, held-out F1 and EM, evaluation-set size, gold recall at 3, cumulative
bytes, wall-clock seconds, and mean client-update divergence.

`results/logs/runs/d{depth}_s{seed}_summary.json` holds the manifest plus final
scores and treatment diagnostics (promotion rate, rank-change rate).

**Logs are written fresh, not appended.** An earlier version appended on retry,
which produced curves that walked backwards in round order; the writer now
truncates and the curve loader additionally de-duplicates by round, last write
winning. The defect and its fix are preserved in the commit history.

### 2.2 Frozen input artefacts

Computed once and reused by every run, so that depth selects a different slice
of identical inputs:

| File | Contents |
|---|---|
| `results/logs/week2/corpus.json` | 200,000-passage working corpus, build statistics |
| `results/logs/week2/partition_{1001,1002}_summary.json` | client allocation statistics for both partition seeds |
| `results/logs/week2/indexes_{1001,1002}.json` | dense index build records |
| `results/logs/week2/retrieval_cache_{1001,1002}.json` | frozen dense retrieval and cross-encoder scores |

Freezing the retrieval cache is what makes the comparison clean: the
cross-encoder scored each question once per depth rather than once per round, so
no run differs from another by retrieval stochasticity.

### 2.3 Pilot-gate artefacts

`results/logs/gates/`:

| File | Decision recorded |
|---|---|
| `gate1_headroom.json` | headroom 18.40 F1 (gold 19.57 vs random 1.17) ≥ 5.0 → proceed |
| `gate2_monotonicity.json` | gold recall at 3 rose 0.467 → 0.5095; recoverable share 0.202–0.307 ≥ 0.10 → proceed |
| `gate2_monotonicity_n300_gold2_superseded.json` | superseded 300-question version, retained |
| `gate3_local_steps.json` | divergence by K ∈ {4, 8, 16, 32, 64}; probe at 250 questions per client |
| `calibrated_schedule.json` | K = 32, S = 1,892, R = 59 re-derived at full scale |
| `scan_stats.json` | corpus scan: 84,076 of 91,535 questions covered |

The superseded gate-2 file is kept on purpose. The first version used 300
questions and a weaker gold-matching rule; replacing it was a methodological
decision, and deleting the earlier evidence would hide that the decision was
made.

### 2.4 Derived artefacts

`results/logs/week4/flops.json` records the measured FLOPs per cross-encoder
pair (5.437 × 10⁹, 256 tokens) and per embedded passage (5.437 × 10⁹, 128
tokens), each alongside its analytic estimate and the measured/analytic ratio,
plus measured CPU throughput of 20.34 pairs per second. Recording both the
measurement and the estimate means a reader can see that the cost model's
denominator is not an unchecked assumption.

`results/logs/week4/main_runs.json` and `run_checks.json` record the sweep
execution and the post-hoc integrity verdict.
`results/logs/analysis/exchange_rate.json` is the single analysis artefact from
which every table and figure in the report is produced.

### 2.5 Deliberately not retained

**Per-round model checkpoints.** They are large, fully derived from the logged
configuration and the frozen inputs, and reproducible by re-running. Retaining
them would multiply the archive size without adding evidence.

**The base model and source datasets.** `SmolLM2-135M-Instruct`, `nq_open` and
`wiki_dpr_e5` are fetched from the Hugging Face Hub at run time and are not
redistributed. Their identifiers and the exact column projection used are
recorded in `configs/experiment_config.yaml`.

**Intermediate embedding tensors.** Regenerable from the corpus and the recorded
embedding model.

---

## 3. Provenance of inputs

| Input | Source | Licence |
|---|---|---|
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` | Apache-2.0 |
| Questions | `google-research-datasets/nq_open` | CC BY-SA 3.0 |
| Passages | `kenhktsui/wiki_dpr_e5` (mirror of DPR `psgs_w100`) | CC BY-SA 3.0 (Wikipedia) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache-2.0 |
| Embedder | `BAAI/bge-small-en-v1.5` | MIT |

The CC BY-SA 3.0 provenance of the question and passage sources is the reason
experiment logs are released under CC BY-SA 4.0 rather than Apache-2.0:
generated text in the logs reproduces material derived from those corpora.

**Ethics and permits.** No human participants, animals, personal data or
physical specimens were involved. The corpora are public research datasets used
under their stated licences. No ethics approval was required, and none was
sought.

---

## 4. Inclusion and exclusion criteria

Recording these explicitly matters, because each one would otherwise be an
invisible degree of freedom.

**Gold passages.** Included only when a normalised answer string was contained
in the passage *and* the passage shared at least two discriminative content
words with the question. Containment alone marks 98 % of the corpus as gold for
some question, because 69 % of Natural Questions answers are a single token; the
overlap requirement removed 96.8 % of those false positives.

**Hard negatives.** Up to five BM25 negatives per question, with answer-bearing
candidates excluded so a distractor could not be accidentally correct.

**Target thresholds.** Of 80 candidate target values drawn from the baseline
curves' own evaluated scores, 3 fell below the derived lower bound (18.13, the
maximum over seeds of the baseline's round-one score) and 14 above the derived
upper bound (28.14, the minimum over seeds of the baseline's peak), leaving 63.
Both bounds are derived from the data, not declared.

**Seeds.** Four, not five. Two partition assignments are alternated across
seeds, so only an even seed count balances the allocation; five seeds would give
a 3:2 imbalance.

**Runs excluded from analysis.** None. All 24 completed and all 24 were used.

**Outlier handling.** None. The high-target tail of the saving curve swings by
more than ±150 MB (Figure 2 of the report) and was **not** trimmed, because
choosing a cut point would be a further arbitrary decision. Extreme-insensitive
statistics — medians and counts of targets — were used instead.

---

## 5. Derived measures and transformations

| Measure | Definition |
|---|---|
| Promotion rate | share of the three prompt slots holding a passage dense retrieval alone would not have selected |
| Rank-change rate | share of slots holding a different passage than the baseline, reordering included |
| Rounds-to-target | first round reaching target F1, by linear interpolation between evaluated rounds, computed **per seed then averaged** |
| Rounds saved | baseline rounds minus condition rounds, paired by seed |
| Bytes saved | rounds saved × 29.49 MB per round |
| Exchange rate | bytes saved ÷ reranking FLOPs |
| Break-even bandwidth | `saved × 8 × throughput ÷ FLOPs`, clamped to 0 when the numerator is non-positive |
| Client-update divergence | mean pairwise divergence of the eight client updates, averaged over the first ten rounds |

Two ordering choices in this table are load-bearing and were both corrected
during the placement after review. Interpolating per seed and then averaging —
rather than averaging the curves first — preserves the between-seed variance the
interval depends on. Taking the median of rounds saved across all 63 targets —
rather than reading one representative target — prevents a single threshold from
determining the result; an earlier draft read the exchange-rate numerator at one
target, which gave two depths the opposite sign to the rounds table computed
over all 63.

---

## 6. Statistical decisions, and the ones that were revised

**Pairing.** Depth is compared with baseline seed by seed. Seed identity shifts a
whole run's round count, and pairing cancels that component: 0.16–0.47 rounds
paired against 0.879 unpaired.

**Critical value.** Four seeds give three degrees of freedom and a two-tailed
95 % critical value of **t = 3.182**. An earlier version of the analysis used
one standard error in code and "twice the standard error" in prose — two
different rules, both too lenient. Under the corrected rule the count of targets
at which a saving is resolved falls from 26/10/16/9/7 to **5/0/1/1/5**, and one
contrast that had read as significant (`d` = 10 vs `d` = 0) does not
(p = 0.127). The direction of the conclusion was unchanged; the rule was wrong
and is now right.

**Resolution ceiling.** A paired permutation test over four seeds has 2⁴ = 16
sign arrangements and therefore a smallest attainable two-tailed p of 0.125.
This is recorded in the analysis artefact under `resolution_limit`, so that the
`within_noise` verdict is read as a property of the design rather than as a
failure to find something.

**Verdict rule.** A saving counts as observed only when it clears the decision
threshold at a majority of the 63 targets. This rule replaced an earlier one
under which any positive value counted; the earlier rule allowed a single
+0.0988-round value with a standard error of 0.515 to flip the verdict from
`no_effect` to `consistent`.

---

## 7. How the record is organised

### 7.1 Weekly branches are the laboratory notebook

The repository carries linear branches `week1` … `week6`. Each preserves the
state of the project at the end of that week, and commit messages carry the
reasoning — what was tried, what failed, and why a choice was made — rather than
only what changed.

| Branch | Period | Content |
|---|---|---|
| `week1` | 16 Sep 2026 | plan, scaffolding, pilot gate 1 |
| `week2` | Sep 2026 | corpus construction, partitions, indexes, retrieval caches |
| `week3` | Sep–Oct 2026 | gates 2 and 3, schedule calibration, pilot runs |
| `week4` | Oct 2026 | main sweep (24 runs), FLOP measurement, integrity checks |
| `week5` | Oct 2026 | analysis, cost model, mechanism check, report |
| `week6` | Oct 2026 | review responses, corrections, final deliverables |

This is the part of the record that cannot be reconstructed after the fact, and
it is why branches are retained rather than squashed. Several decisions reported
as single lines in the report — the gate-2 replacement, the interpolation
ordering, the critical-value correction — are legible only in that history.

### 7.2 Repository layout

```
configs/     experiment configuration (single YAML, dev overlay)
src/         library code: data, retrieval, federated training, analysis
scripts/     runnable entry points and report conversion
tests/       333 automated tests
results/
  logs/
    gates/     pilot-gate artefacts
    week2/     frozen corpus, partitions, indexes, retrieval caches
    week3/     pilot run logs
    week4/     FLOP measurement, sweep record, integrity checks
    runs/      24 per-round JSONL logs + 24 summaries
    analysis/  exchange_rate.json
  figures/   4 PNG figures
docs/        weekly reports, final report, research records
```

### 7.3 Integrity verification

`MANIFEST.sha256` lists a SHA-256 for every archived file. To verify:

```
sha256sum -c MANIFEST.sha256
```

Separately, `scripts/check_runs.py` re-reads the 24 run manifests and asserts
that rounds, client count, payload size, local steps, total bytes, batch size,
gradient accumulation, examples per step and base model are identical across
runs, and that the treatment diagnostic differs between conditions as it must.
It reports **24/24 complete, identical budget, one axis varied**. A field
recorded for some runs but not all is itself flagged, because a setting that was
not recorded cannot be verified.

---

## 8. Reproducing the analysis

Analysis runs from the archived logs alone and needs no GPU:

```
pip install -r requirements.txt
sha256sum -c MANIFEST.sha256
python scripts/check_runs.py
python scripts/analyze_exchange_rate.py
python scripts/plot_results.py
pytest tests/
```

`analyze_exchange_rate.py` rewrites `results/logs/analysis/exchange_rate.json`;
`plot_results.py` rewrites the four figures from it. Every table in the report
is derived from that artefact.

Reproducing the **runs** requires a GPU and roughly 34 hours:

```
python scripts/run_main_runs.py
```

A single run can be targeted with `--only-depth` and `--only-seed`. A changed
base model requires `--set treatment.variant=<name>` so that its logs are
written under distinct names; the runner refuses to proceed otherwise, because
run names are built from depth and seed alone and would otherwise overwrite the
main matrix.

**Environment.** Python 3.11 on Linux; training on an NVIDIA A10G GPU in
bfloat16. The pinned package versions are in `requirements.txt` and a frozen
environment listing accompanies the archive. Versions are pinned rather than
ranged because `transformers` 5.x removed dataset script loading — which is why
a parquet mirror of the passage corpus is used — and `sentence-transformers` 6.x
renamed the embedding-dimension API; a ranged specification would silently run
different code a few months later.

---

## 9. Archiving and long-term storage

**Citable snapshot.** The repository is archived on Zenodo as a versioned
release with a DOI, under the same arrangement used for the author's previous
placement (Zenodo record 22781474). The Zenodo deposit is a single zip of the
`main` branch snapshot, including `MANIFEST.sha256`, and is linked to the GitHub
repository through an "is supplement to" relation.

**Working repository.** GitHub, `hufs0529/rag-fl`, with the week branches
retained.

**Retention.** Zenodo provides long-term preservation with a persistent DOI;
nothing in this project depends on GitHub remaining available. The archive is
self-contained for analysis: the logs, the scripts that read them and the tests
that constrain those scripts are all inside it.

**Licensing.** Source code, scripts and tests under Apache-2.0
(`LICENSE`). Experiment logs under `results/logs/` under CC BY-SA 4.0
(`LICENSE-DATA`), for the provenance reason in Section 3. Neither the base model
nor the source corpora are redistributed.

**Citation.** `CITATION.cff` in the repository root carries the citation
metadata; `.zenodo.json` carries the deposit metadata (title, creator, ORCID,
affiliation, keywords, licence, related identifiers).

**Analytical scripts.** Archived with the data rather than separately, so that
the artefact and the code that produced it cannot drift apart. The 333 tests are
included for the same reason: they record what the analysis is supposed to do,
and several encode specific defects that were found and fixed so that they
cannot silently return.

---

## 10. Deviations from the original plan

| Planned | Actual | Reason |
|---|---|---|
| 4 depths × 5 seeds = 20 runs | 6 depths × 4 seeds = 24 runs | five seeds give a 3:2 partition imbalance; depths 7 and 25 were added on mentor feedback that the ladder was too coarse to show a trend |
| Target accuracy fixed at one value | 63 targets swept | a single target lets the choice determine the result; bounds are derived from the baseline curves |
| Uniform grid of target values | the baseline curves' own evaluated values | on a uniform grid, 15 of 25 points fell inside the round 0→1 interval and the measured saving came out exactly zero |
| K = 4 from the pilot probe | K = 32 | the probe ran at 250 questions per client; the decision was re-derived at full scale from round-one divergence |
| Gate 2 on 300 questions | re-run on 2,000 with a stricter gold rule | containment-only matching marked 98 % of the corpus as gold |

An additional 8-run comparison at a larger model scale was scoped and the
tooling for it was built and tested, but the runs were not executed: the
estimated 28–40 hours of GPU time fell outside the placement budget. The
capability is retained in the archive so the comparison can be run later without
rework.

---

## 11. Known defects found and corrected during the placement

Recorded because a record that lists only successes is not a record.

| Defect | Consequence had it not been found | Status |
|---|---|---|
| Logger appended on retry | curves walked backwards in round order | fixed; writer truncates, loader de-duplicates |
| Sweep launched twice concurrently | two processes writing the same logs | fixed; PID lock, verified to refuse a second launch |
| `--only-depth` accepted values absent from the matrix | 4 runs executed silently where 8 were intended | fixed; the runner exits with the available values |
| Verdict rule accepted any positive saving | one +0.0988-round value (s.e. 0.515) flipped the verdict | fixed; majority-of-targets rule |
| Exchange rate read at one target | two depths carried the opposite sign to the rounds table | fixed; median over all 63 targets |
| Decision threshold used 1–2 × s.e. | one contrast read as significant that is not | fixed; t(0.975, 3) = 3.182 |
| Mechanism figure used a dual axis | a 0.77 % change rendered as a collapse, contradicting the text | fixed; two panels, divergence axis anchored at zero |
| Manifest did not record the base model | a re-run under a different model would have overwritten the main matrix | fixed; recorded, checked, and guarded |
| Batch split not recorded in the manifest | identical-budget claim unverifiable | fixed; three fields added and compared |

---

## 12. Use of generative AI

An AI coding assistant was used throughout, consistent with Macquarie University
policy. It wrote and refactored analysis code, drafted and revised report text,
and reviewed statistical reasoning; several of the defects in Section 11 were
identified in that review. All design decisions and all interpretation were made
and approved by the author.

No experimental data were generated, altered or synthesised by an AI system.
Every number in the report and in this document is produced by an archived
script reading an archived log, and the full development history — including the
reasoning behind each change and each correction — is preserved in the
repository's commit record and weekly branches.

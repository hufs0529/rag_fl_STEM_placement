# Buying Back Communication with Retrieval Compute

**The Exchange Rate of Client-Side Reranking in Federated LLM Fine-Tuning**

A 6-week STEM research placement project. This repository contains the full
experimental pipeline, the analysis code and the reporting artefacts.

> **You are on `week6`, week 6 of 6.** The final deliverables, and the full pipeline in one command. This branch carries all six weeks.
> The [branch map](#4-branch-map-one-branch-per-week) shows what each branch adds;
> `week6` carries everything.

---

## 1. The question

Federated Learning (FL) lets several machines jointly fine-tune an LLM without
sharing raw data, and its dominant cost is communication. Total communication is

```
total bytes  =  R  x  N  x  |AB|  x  2
                ^     ^      ^      ^
                |     |      |      upload + download
                |     |      LoRA adapter payload
                |     number of clients
                rounds required
```

The established levers — PEFT, quantisation, sparsification — all shrink `|AB|`,
the **bytes per round**. This project attacks the other factor, `R`, the
**rounds required**, from an unexpected direction: the quality of the retrieval
that constructs each client's training inputs.

> **Aim.** Measure whether raising retrieval quality through client-local
> reranking reduces the communication required to reach a target accuracy in
> federated LLM fine-tuning, and quantify the **exchange rate** between
> reranking compute and communication bytes.

### Why retrieval could move the round count

Institutions that cannot centralise data do not deploy a bare fine-tuned model;
they deploy RAG. Aligning training with deployment therefore means retrieving
*inside* the training loop (retrieval-augmented fine-tuning, RAFT [4]), so each
client trains on prompts assembled from its own private corpus via its own index.

That creates a coupling nobody has measured:

* when retrieval **succeeds**, every client is learning the same task —
  grounding an answer in supporting evidence;
* when retrieval **fails**, each client instead adapts to its own pattern of
  irrelevant context.

Non-IID client updates are known to diverge and slow convergence [9, 10]. Poor
and unevenly distributed retrieval is therefore a plausible, previously untested
contributor to round count. Reranking [5] is the standard remedy for retrieval
quality — but inside a federated training loop its cost is a *recurring charge
borne by every client*. So the question is a **trade**, not an improvement:
local compute spent in exchange for less network transfer.

---

## 2. Experimental design in one page

### The single treatment axis

`d` = **reranking depth** = the number of dense candidates rescored by the
cross-encoder before the top 3 are selected.

| `d`    | What happens                                                        | Role                     |
|--------|---------------------------------------------------------------------|--------------------------|
| **0**  | Dense retrieval only, top 3 taken as-is                             | Baseline                 |
| **3**  | The 3 passages dense already selected are *reordered*, none promoted | **Control** for ordering |
| **10** | Top 10 rescored, best 3 kept                                        | Treatment                |
| **50** | Full candidate pool rescored, best 3 kept                           | Treatment (max dose)     |

Every condition puts **exactly three passages** in the prompt, drawn from **one
common top-50 candidate pool**. Conditions differ *only* in how much compute was
spent selecting them. `d = 3` separates a pure ordering effect from a genuine
retrieval-quality effect.

Depth is the axis rather than retriever capacity because depth is a **recurring
per-query cost**, commensurable with the per-round communication traded against
it, whereas a larger encoder is a one-off indexing cost. Depth also spans an
order of magnitude of compute on a single fixed index.

Everything else — base model, embedding model, chunking, top-k, prompt template,
client partition, local and total steps — is held identical.

### Fixed background conditions

| Component        | Choice                                        | Why                                                               |
|------------------|-----------------------------------------------|-------------------------------------------------------------------|
| Base model       | `SmolLM2-135M-Instruct` [15]                  | Task is grounding, not new world knowledge                        |
| Adapter          | LoRA `r=8`, attention projections only [2]    | 0.92M trainable params = **1.84 MB / client / direction** at fp16 |
| Clients          | 8, full participation, FedAvg [1]             | Cross-silo scale                                                  |
| Partition        | Equal-size, **topic-disjoint** (not Dirichlet)| Keeps the design one-dimensional (see below)                      |
| Questions        | NQ-open [7] (87,925 train / 3,610 val)        | Short extractive answers                                          |
| Corpus           | 200k passages from DPR `psgs_w100` [8]        | Gold + BM25 hard negatives + random remainder                     |
| Embedder         | `BAAI/bge-small-en-v1.5` [12]                 | Frozen; per-client Qdrant collections                             |
| Reranker         | `cross-encoder/ms-marco-MiniLM-L-6-v2` [6]    | Distilled 6-layer — plausible on constrained hardware             |
| Framework        | Flower simulation [14] + HuggingFace PEFT     | 8 simulated clients on one machine                                |

**Why not a Dirichlet partition?** It would vary two things at once — tightening
topic range (intended) *and* making client dataset sizes unequal (an unintended
source of averaging instability) — and it redraws under each seed, injecting
partition variance into the very estimate the seeds are meant to stabilise.
Client heterogeneity here is a **fixed background condition, not a factor**. Two
equal-size topic-disjoint partitions are assigned across the 4 seeds so the
result is not tied to one allocation.

### Two measurement problems, and how the design answers them

1. **Round count is too coarse a ruler.** LoRA adapters train few parameters and
   converge in few rounds, so a convergence-stopped experiment cannot resolve a
   treatment effect at this scale. → Every run consumes an **identical
   communication budget with no early stopping**, and rounds-to-target are
   recovered by **linear interpolation of the learning curves**. The result never
   depends on a stopping rule.

2. **Rounds are not a free parameter.** `R = S / K` where `S` is total local
   steps and `K` is local steps per round, and `K` governs how far clients drift
   before averaging. → Both are fixed by a **documented pilot** (gate 3), not by
   convention.

### The three Week-1 pilot gates

Cheap, run before any full experiment, each with an explicit failure action.

| Gate                        | Needs          | Passes if                                        | If it fails                       |
|-----------------------------|----------------|--------------------------------------------------|-----------------------------------|
| (i) Capability headroom     | No training    | F1(gold ctx) − F1(random ctx) ≥ ~5                | Escalate to a larger checkpoint   |
| (ii) Depth monotonicity     | Retrieval only | gold recall@3 rises monotonically in `d`          | The treatment axis does not exist; revise depths |
| (iii) Local-step calibration| Short FL probe | smallest `K` with measurable client drift         | Revise `K` range; `S` then sets `R = S / K` |

Gate (iii) exists because **a treatment that acts by reducing drift cannot be
observed where drift is absent.**

### Measurement protocol (24 main runs = 6 depths × 4 seeds)

* Identical communication budget per run, no early stopping.
* **Answer F1** primary (continuous, interpolates stably), **EM** alongside.
* Fixed **1,000-question** held-out set, identical across every run, scored
  centrally on the global model, greedy decoding, official NQ normalisation.
* **Evaluation context held at `d = 0` for all conditions** — this isolates what
  the model *learned* from reranking rather than what better inference-time
  context supplies.
* Curves evaluated on a fixed round schedule, dense early and sparse late.
* Also logged: gold recall@3 and rank-change rate (confirming the treatment took
  effect), cumulative bytes, reranker forward passes at measured FLOPs, indexing
  cost, client-update divergence, wall-clock.

### From curves to an exchange rate

For a target accuracy `τ`, read off the round at which each depth first reaches
`τ` by linear interpolation, and convert the difference into saved bytes via
`R x N x |AB| x 2`. Because the verdict would otherwise depend on the choice of
`τ`, **`τ` is swept** across the range the baseline attains and the saving is
reported as a curve; regions where a condition never reaches `τ` are reported as
such rather than extrapolated.

Reranking cost is charged against the saving under a **two-term model**:

```
one-off      C_index  = |corpus| x FLOPs_embed(passage)
recurring    C_rerank = Q x d x FLOPs_ce(query, passage)
```

Both are converted into the same unit as saved bytes (seconds, via link
bandwidth `B` and client throughput `T`), and the verdict is reported as a
**surface over bandwidth × query volume with the break-even contour marked** —
an operating regime, not a single verdict.

---

## 3. Repository layout

```
rag_fl_STEM_placement/
├── configs/
│   ├── experiment_config.yaml   # single source of truth for the whole plan
│   └── dev_config.yaml          # reduced-scale overlay for rehearsal/tests
├── src/                         # library code (imported by scripts and tests)
├── scripts/                     # CLI entry points, one per plan step
├── tests/                       # pytest; network/slow tests are marked
├── results/
│   ├── cache/                   # frozen retrieval artefacts (gitignored)
│   ├── logs/                    # per-run JSONL (gitignored)
│   └── figures/                 # generated plots (gitignored)
├── reports/                     # final report, presentation, records
└── data/                        # corpus, indexes, partitions (gitignored)
```

Every module carries a bilingual docstring naming the **subtask number** from
the plan that it implements, so the code and the proposal stay traceable to each
other.

### Where the design lives in the code

The parts of the design that are easiest to get wrong are enforced rather than
documented. On this branch:

| Design commitment | Enforced by |
|---|---|
| Exactly three passages, one common pool; depth is the only difference | `src/selection.py` + `tests/test_selection.py` |
| `d = 3` reorders but never promotes | `promotion_rate == 0` asserted in `tests/test_selection.py` and in gate (ii) |
| A flat recall curve is a failure, not a pass | `monotonicity_decision()` — no treatment axis means no experiment |
| Answer-bearing passages are never used as hard negatives | `verify_no_answer_bearing_negatives()`, non-zero exit on violation |
| One common top-50 pool serves every depth | `src/retrieval_cache.py` — per-condition caches could drift apart |
| Rounds come from a pilot, not a convention | `resolve_schedule()` raises without gate 3's artefact |
| No early stopping | there is no stopping branch in `src/fl_runner.py` |
| Evaluation context fixed at `d = 0` | `eval.eval_depth`, asserted equal across depths by the rehearsal |
| All 24 runs spent an identical budget | `check_runs.py` compares every budget field post hoc |
| τ is swept; conditions that never reach it are not extrapolated | `tau_grid()` + `rounds_to_target()` returning `None` |
| Report numbers match the analysis | `reports/` is generated from the analysis JSON, never transcribed |

---

## 4. Branch map (one branch per week)

The work is delivered as six stacked branches off `main`. Each branch is
self-contained and builds on the previous one; `week6` contains everything.

| Branch | Plan subtask | Deliverable | On this branch |
|---|---|---|---|
| `main` | — | Scaffolding: config system, layout, requirements, this README | included |
| `week1` | 1.1 / 1.2 | NQ + `psgs_w100` scan, metrics, prompting, **the three pilot gates** — [runbook](docs/week1.md) | included |
| `week2` | 1.1 / 1.2 | 200k corpus with hard negatives, topic partition, Qdrant, **retrieval cache** — [runbook](docs/week2.md) | included |
| `week3` | 1.2 | Flower + PEFT federated pipeline, loss masking, cost/recall/drift logging — [runbook](docs/week3.md) | included |
| `week4` | 1.3 | The 24 main runs at a fixed communication budget — [runbook](docs/week4.md) | included |
| `week5` | 1.4 | τ-sweep interpolation, two-term cost model, break-even surface, diagnostics — [runbook](docs/week5.md) | included |
| **`week6`** | 1.4 | Final report, presentation, limitations, release notes — [runbook](docs/week6.md) | **you are here** |

```bash
git log --oneline --graph main week1 week2 week3 week4 week5 week6
```

Every runbook is present here: `docs/week1.md` … `docs/week6.md`.

---

## 5. Getting started

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Heavy dependencies (`torch`, `transformers`, `peft`, `flwr`, `sentence-transformers`, `qdrant-client`) are needed from Week 1's gates onward. The pure-logic modules — metrics, passage selection, partitioning, the
cost model, curve interpolation — depend only on `numpy`/`pyyaml` and their
tests run without a GPU:

```bash
pytest -q                                # 333 tests on this branch
pytest -q -m "not network and not slow"
```

Every script shares the same config interface:

```bash
python scripts/<any_script>.py --dev                       # reduced scale
python scripts/<any_script>.py --set train.rounds=20       # override one value
python scripts/<any_script>.py --config configs/experiment_config.yaml
```

---

## 6. Reproducing the whole thing

```bash
bash scripts/run_all.sh            # weeks 1-6 end to end
bash scripts/run_all.sh --dev      # reduced-scale smoke run
```

Or one week at a time:

```bash
python scripts/scan_corpus.py --sample-fraction 1.0   # week 1
python scripts/run_week1_gates.py                     # week 1  -> GO / NO-GO
python scripts/run_week2_pipeline.py                  # week 2  -> frozen artefact
python scripts/run_rehearsal.py --dev --synthetic     # week 3  -> rehearsed pipeline
python scripts/run_main_runs.py                       # week 4  -> 24 runs
python scripts/check_runs.py                          # week 4  -> did only one axis vary?
python scripts/measure_flops.py                       # week 4  -> cost inputs
python scripts/analyze_exchange_rate.py               # week 5  -> exchange rate
python scripts/plot_results.py                        # week 5  -> figures
python scripts/make_report.py                         # week 6  -> report
```

`reports/final_report.md` and `reports/presentation.md` are **generated** from
`results/logs/analysis/exchange_rate.json`, never transcribed — re-run
`make_report.py` after any re-analysis.

---

## 7. Deliverables and outputs

| Artefact | Contents |
|---|---|
| [`docs/final_report.pdf`](docs/final_report.pdf) | **placement deliverable — report.** Question, methods, statistics, results, discussion, limitations, references |
| [`docs/research_records.pdf`](docs/research_records.pdf) | **placement deliverable — research records.** Data management plan, run inventory, provenance, deviations, corrected defects |
| `docs/week{1,23,4,5}_report_{ko,en}.{md,docx}` | weekly reports in Korean and English |
| `results/logs/runs/` | 24 per-round JSONL logs and summaries — the primary record |
| `results/logs/analysis/exchange_rate.json` | the single artefact every table and figure derives from |
| `results/figures/` | four figures, regenerated by `plot_results.py` |
| `MANIFEST.sha256` | SHA-256 for every archived file; `sha256sum -c MANIFEST.sha256` |

`data/` and `results/cache/` are gitignored and rebuilt by the scripts. The
per-round logs under `results/logs/` are **tracked on purpose** — they are the
evidence, and the analysis runs from them without a GPU.

### Key findings

* **The treatment worked.** New-document share rose monotonically from 0 % to
  52.5 % with reranking depth, and all 24 runs spent an identical 1.74 GB.
* **The learning did not follow.** Across 63 derived target thresholds, no depth
  resolved a round saving at more than 5 of them, and the largest median saving
  belonged to `d` = 3 — an ordering control that cannot change which documents
  appear.
* **The chain breaks at one identified link.** Mean client-update divergence
  moved 0.77 % while treatment strength rose from 0 to 0.72: the better
  documents reached the clients and the clients did not use them.
* **Eight of ten exchange rates are negative**, and both positive ones belong to
  the control. The scoped conclusion is that the fewer-rounds path alone cannot
  buy back the reranking compute at this model scale.

## 8. Stated limitations

Carried through to the final report, not discovered at the end:

* 4 seeds — a paired permutation test over four seeds has a floor of `p` = 0.125,
  so **no effect of any size could have been declared significant**. Effect sizes,
  standard errors and the design's own resolution limit are reported.
* A single model scale (135M) and a single adapter rank.
* A fixed round horizon; behaviour beyond the budget is not observed.
* A fixed client partition (two allocations across the seeds).
* Simulated clients — wall-clock and network behaviour are modelled, not measured on real links.
* **Answer-string matching as a proxy for gold labels** [8]: a passage containing
  the answer string is not necessarily supporting evidence.
* The cost model's assumptions (effective device throughput, amortisation of
  indexing) are stated explicitly and swept where they matter.

---

## 9. References

1. H. B. McMahan et al., "Communication-efficient learning of deep networks from decentralized data," AISTATS, 2017.
2. E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," arXiv:2106.09685, 2021.
3. P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," NeurIPS, 2020.
4. T. Zhang et al., "RAFT: Adapting language model to domain specific RAG," arXiv:2403.10131, 2024.
5. R. Nogueira and K. Cho, "Passage re-ranking with BERT," arXiv:1901.04085, 2019.
6. W. Wang et al., "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers," NeurIPS, 2020.
7. T. Kwiatkowski et al., "Natural Questions: A benchmark for question answering research," TACL, vol. 7, pp. 453–466, 2019.
8. V. Karpukhin et al., "Dense passage retrieval for open-domain question answering," EMNLP, 2020.
9. T. Li et al., "Federated optimization in heterogeneous networks," MLSys, 2020.
10. Q. Li, Y. Diao, Q. Chen, and B. He, "Federated learning on non-IID data silos: An experimental study," IEEE ICDE, 2022.
11. S. P. Karimireddy et al., "SCAFFOLD: Stochastic controlled averaging for federated learning," ICML, 2020.
12. S. Xiao et al., "C-Pack: Packed resources for general Chinese embeddings," arXiv:2309.07597, 2023.
13. K. Lee, M.-W. Chang, and K. Toutanova, "Latent retrieval for weakly supervised open domain question answering," ACL, 2019.
14. D. J. Beutel et al., "Flower: A friendly federated learning research framework," arXiv:2007.14390, 2020.
15. L. Ben Allal et al., "SmolLM2: When Smol goes big — data-centric training of a small language model," arXiv:2502.02737, 2025.
16. P. Kairouz et al., "Advances and open problems in federated learning," FnT in ML, vol. 14, no. 1–2, pp. 1–210, 2021.

---

## 10. Licence

Dual-licensed, because the two kinds of content have different provenance.

* **Source code, tests, scripts and configuration** — Apache-2.0, see [`LICENSE`](LICENSE).
* **Experiment logs under `results/logs/`** — CC BY-SA 4.0, see
  [`LICENSE-DATA`](LICENSE-DATA). Generated text in the logs reproduces material
  derived from Natural Questions and the DPR Wikipedia split, both CC BY-SA 3.0.

The base model and the source corpora are **not redistributed**; they are fetched
from the Hugging Face Hub at run time, and their identifiers are recorded in
`configs/experiment_config.yaml`.

## 11. Archive and citation

The citable snapshot is deposited on Zenodo and linked to this repository through
an *is supplement to* relation. Deposit metadata is in
[`.zenodo.json`](.zenodo.json) and citation metadata in
[`CITATION.cff`](CITATION.cff); the DOI is added to `CITATION.cff` once Zenodo
issues it for the release.

Verify an archive copy before using it:

```bash
sha256sum -c MANIFEST.sha256     # regenerate with scripts/make_manifest.py
```

The frozen environment used for the runs is recorded in `env_a10g.txt`; the
pinned package set is in `requirements.txt`. Versions are pinned rather than
ranged because `transformers` 5.x removed dataset script loading and
`sentence-transformers` 6.x renamed the embedding-dimension API — a ranged
specification would silently run different code a few months later.

---

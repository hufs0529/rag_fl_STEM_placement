# Buying Back Communication with Retrieval Compute

**The Exchange Rate of Client-Side Reranking in Federated LLM Fine-Tuning**

A 6-week STEM research placement project. This repository contains the full
experimental pipeline, the analysis code and the reporting artefacts.

> **You are on `week3`, week 3 of 6.** The federated training loop (Flower + PEFT) and the rehearsal that exercises every path from cached retrieval to a logged learning curve.
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
equal-size topic-disjoint partitions are assigned across the 5 seeds so the
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

### Measurement protocol (20 main runs = 4 depths × 5 seeds)

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
rag-fl/
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

The remaining commitments are enforced by code that arrives on later branches — see the branch map.

---

## 4. Branch map (one branch per week)

The work is delivered as six stacked branches off `main`. Each branch is
self-contained and builds on the previous one; `week6` contains everything.

| Branch | Plan subtask | Deliverable | On this branch |
|---|---|---|---|
| `main` | — | Scaffolding: config system, layout, requirements, this README | included |
| `week1` | 1.1 / 1.2 | NQ + `psgs_w100` scan, metrics, prompting, **the three pilot gates** — [runbook](docs/week1.md) | included |
| `week2` | 1.1 / 1.2 | 200k corpus with hard negatives, topic partition, Qdrant, **retrieval cache** — [runbook](docs/week2.md) | included |
| **`week3`** | 1.2 | Flower + PEFT federated pipeline, loss masking, cost/recall/drift logging — [runbook](docs/week3.md) | **you are here** |
| `week4` | 1.3 | The 20 main runs at a fixed communication budget | later |
| `week5` | 1.4 | τ-sweep interpolation, two-term cost model, break-even surface, diagnostics | later |
| `week6` | 1.4 | Final report, presentation, limitations, release notes | later |

```bash
git log --oneline --graph main week1 week2 week3 week4 week5 week6
```

Runbooks present on this branch: `docs/week1.md`, `docs/week2.md`, `docs/week3.md`. The later ones arrive with their branches.

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
pytest -q                                # 235 tests on this branch
pytest -q -m "not network and not slow"
```

Every script shares the same config interface:

```bash
python scripts/<any_script>.py --dev                       # reduced scale
python scripts/<any_script>.py --set train.rounds=20       # override one value
python scripts/<any_script>.py --config configs/experiment_config.yaml
```

---

## 6. What you can run here

```bash
# rehearse the wiring offline, no 21M-passage download needed
python scripts/run_rehearsal.py --dev --synthetic

# one real run, once week2's artefacts and gate 3's schedule exist
python scripts/run_single_run.py --depth 10 --seed 1
python scripts/run_single_run.py --depth 0 --seed 1 --backend inprocess
```

The rehearsal asserts the two failure modes that would otherwise yield a
plausible-looking result: the **treatment silently not applying**
(`promotion_rate > 0` required at `d > 3`, `== 0` at `d ∈ {0, 3}`) and the
**treatment leaking into evaluation** (eval recall identical across depths).

---

## 7. Outputs

| Artefact | Contents |
|---|---|
| `results/logs/runs/d{d}_s{s}.jsonl` | per-round curve, cumulative bytes, drift, treatment diagnostics |
| `results/logs/runs/d{d}_s{s}_summary.json` | budget, final scores, completion flag |
| `results/logs/week3/rehearsal.json` | the rehearsal's pass/fail checks |

Everything under `data/`, `results/` and generated reports is gitignored: the scripts rebuild it.

---

## 8. Stated limitations

Carried through to the final report, not discovered at the end:

* 5 seeds — effect sizes and standard errors are reported, **not significance tests**.
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

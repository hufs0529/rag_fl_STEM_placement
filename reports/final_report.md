# Buying Back Communication with Retrieval Compute

**The Exchange Rate of Client-Side Reranking in Federated LLM Fine-Tuning**

*Generated from `results/logs/analysis/exchange_rate.json` on 2026-10-23.*

---

## 1. Summary

Across all 63 targets the baseline attains, reranking at d = 50 reached them 0.37 rounds SLOWER than dense retrieval alone at the median, beating the baseline at 15 of 63 targets and clearing its own standard error at 7. No single target decides this: every score the baseline recorded inside the comparable range is used as one. No depth resolves a saving at a majority of targets, so this measurement resolves no saving and there is no mechanism to explain.

## 2. What the question was

Total federated communication is `R x N x |AB| x 2`. PEFT, quantisation and
sparsification all shrink `|AB|`. This project asked whether `R` can be reduced
from a different direction — by improving the retrieval that constructs each
client's training inputs — and at what exchange rate against the local compute
that improvement costs.

The treatment was **reranking depth** `d`, the number of dense candidates
rescored by a distilled cross-encoder before the top 3 are selected, at
`d ∈ [0, 3, 7, 10, 25, 50]`. Every condition drew from one common top-50 pool
and placed exactly three passages in every prompt; `d = 3` was a control that
reorders without promoting. Everything else was held identical.

## 3. Protocol

* 8 clients, FedAvg, full participation, Flower simulation.
* LoRA payload **1.84 MB** per client per direction.
* Identical communication budget per run, **no early stopping**; `K`, `S` and `R`
  fixed by the Week-1 pilot rather than by convention.
* A fixed 1000-question held-out set, identical across runs, scored
  centrally on the global model with greedy decoding and the official NQ
  normalisation.
* **Evaluation context held at d = 0 for every condition**, isolating what the
  model learned from reranking rather than what better inference-time context
  supplies.
* 63 τ points swept across the range the baseline attains.

## 4. What the fixed budget bought

| Condition | Held-out F1 at the fixed budget | Runs |
|---|---|---|
| d = 0 (baseline) | 29.13 ± 0.69 (n=4) | 4 |
| d = 3 (control) | 29.44 ± 0.76 (n=4) | 4 |
| d = 7 | 29.20 ± 0.76 (n=4) | 4 |
| d = 10 | 29.72 ± 0.55 (n=4) | 4 |
| d = 25 | 29.24 ± 0.39 (n=4) | 4 |
| d = 50 | 29.41 ± 0.48 (n=4) | 4 |

## 5. Rounds to target and communication saved

Read at a representative τ = 25.02 F1; the full sweep is in
`saving_vs_tau.png` and in the analysis JSON. Conditions that never reach τ are
reported as such and are not extrapolated.

| Condition | Rounds to τ | Rounds saved | Communication saved (MB) |
|---|---|---|---|
| d = 0 (baseline) | 9.02 ± 0.17 (n=4) | 0.00 | 0.0 |
| d = 3 (control) | 9.63 ± 0.28 (n=4) | -0.61 | -17.8 |
| d = 7 | 11.08 ± 0.83 (n=4) | -2.05 | -60.6 |
| d = 10 | 10.05 ± 0.71 (n=4) | -1.03 | -30.4 |
| d = 25 | 8.92 ± 0.51 (n=4) | 0.10 | 2.9 |
| d = 50 | 9.17 ± 0.86 (n=4) | -0.15 | -4.3 |

## 6. The exchange rate

FLOPs per reranked pair: 5.437e+09 (measured).
FLOPs per indexed passage: 5.437e+09 (measured).
Device throughput assumed: 1.000e+12 FLOP/s.

| Condition | Assumption | Bytes saved per extra FLOP | Total FLOPs charged |
|---|---|---|---|
| d = 10 | index amortised | -5.593e-09 | 5.437e+15 |
| d = 10 | indexing charged | -4.661e-09 | 6.525e+15 |
| d = 25 | index amortised | 2.143e-10 | 1.359e+16 |
| d = 25 | indexing charged | 1.984e-10 | 1.468e+16 |
| d = 3 | index amortised | -1.094e-08 | 1.631e+15 |
| d = 3 | indexing charged | -6.565e-09 | 2.719e+15 |
| d = 50 | index amortised | -1.589e-10 | 2.719e+16 |
| d = 50 | indexing charged | -1.528e-10 | 2.827e+16 |
| d = 7 | index amortised | -1.591e-08 | 3.806e+15 |
| d = 7 | indexing charged | -1.238e-08 | 4.894e+15 |

The bytes-per-FLOP figure is the only assumption-free number here: it carries no
bandwidth and no device. Everything below is that number multiplied by a
deployment's parameters.

## 7. Break-even regime

Reranking buys back communication on links **slower** than the contour below,
and the contour falls as query volume rises, because the recurring per-query cost
accumulates while the one-off saving does not.

| Condition | Query volume | Break-even bandwidth (Mbps) — pays off below this |
|---|---|---|
| d = 10 | 1,000 | 0.000 |
| d = 10 | 10,000 | 0.000 |
| d = 10 | 100,000 | 0.000 |
| d = 10 | 1,000,000 | 0.000 |
| d = 10 | 10,000,000 | 0.000 |
| d = 25 | 1,000 | 0.171 |
| d = 25 | 10,000 | 0.017 |
| d = 25 | 100,000 | 0.002 |
| d = 25 | 1,000,000 | 0.000 |
| d = 25 | 10,000,000 | 0.000 |
| d = 3 | 1,000 | 0.000 |
| d = 3 | 10,000 | 0.000 |
| d = 3 | 100,000 | 0.000 |
| d = 3 | 1,000,000 | 0.000 |
| d = 3 | 10,000,000 | 0.000 |
| d = 50 | 1,000 | 0.000 |
| d = 50 | 10,000 | 0.000 |
| d = 50 | 100,000 | 0.000 |
| d = 50 | 1,000,000 | 0.000 |
| d = 50 | 10,000,000 | 0.000 |
| d = 7 | 1,000 | 0.000 |
| d = 7 | 10,000 | 0.000 |
| d = 7 | 100,000 | 0.000 |
| d = 7 | 1,000,000 | 0.000 |
| d = 7 | 10,000,000 | 0.000 |

## 8. Mechanism

The proposed path was: depth ↑ → retrieval quality ↑ → clients learn the same
task → update divergence ↓ → rounds ↓.

| Condition | Treatment strength | Gain vs baseline | Divergence | Reduction vs baseline | Rounds saved |
|---|---|---|---|---|---|
| d = 0 | 0.0000 | 0.0000 | 1.3234 | 0.0000 | 0.00 |
| d = 3 | 0.4505 | 0.4505 | 1.3218 | 0.0015 | 0.15 |
| d = 7 | 0.6484 | 0.6484 | 1.3143 | 0.0091 | -0.56 |
| d = 10 | 0.6764 | 0.6764 | 1.3151 | 0.0083 | -0.21 |
| d = 25 | 0.7120 | 0.7120 | 1.3162 | 0.0071 | -0.11 |
| d = 50 | 0.7244 | 0.7244 | 1.3132 | 0.0101 | -0.37 |

**Verdict: within_noise** — the largest median saving, 0.149 rounds at d=3, clears its own standard error at only 26 of 63 targets; no depth resolves a saving at a majority of targets

Correlations across depths: recall gain vs divergence reduction
r = +0.914, recall gain vs
rounds saved r = -0.652, divergence
reduction vs rounds saved
r = -0.882.

## 9. Limitations

* **Five seeds.** Effect sizes and standard errors are reported; no significance
  tests are performed. Five seeds carry no power, and a p-value computed without
  power misleads more than it informs.
* **A single model scale.** SmolLM2-135M-Instruct with LoRA r = 8 on attention
  projections only. Whether the exchange rate holds at other scales is untested.
* **A fixed round horizon.** Every run spent the same budget; behaviour beyond
  that horizon is not observed, and the τ sweep is bounded by what the baseline
  attained within it.
* **A fixed client partition.** Equal-size and topic-disjoint, with two
  allocations across the five seeds. Client heterogeneity was a background
  condition, not a factor, so nothing here speaks to other heterogeneity regimes.
* **Simulated clients.** Communication is accounted analytically from payload
  size; no real network link was measured. Wall-clock is single-machine.
* **Answer-string matching as a proxy for gold labels.** A passage containing the
  normalised answer string is not necessarily supporting evidence, so both gold
  recall and the corpus construction inherit that approximation.
* **Cost-model assumptions.** Effective device throughput, the amortisation of
  indexing, and measured-versus-analytic FLOPs all shift the break-even contour.
  Each is reported explicitly and swept where it matters, but the verdict is a
  regime under stated assumptions, not a universal constant.
* **Cached retrieval.** All runs shared one retrieval pass because the retrieval
  components are frozen. The cost model charges the full deployment-time
  per-query cost regardless, but no measurement of retrieval *inside* a live
  training loop was taken.

## 10. Figures

| Figure | Shows |
|---|---|
| `learning_curves.png` | mean ± SE curves per depth, in rounds and in cumulative MB |
| `saving_vs_tau.png` | saving as a curve over τ |
| `break_even.png` | break-even bandwidth vs query volume |
| `mechanism.png` | treatment strength vs client-update divergence |

## 11. Reproducing this

```bash
pip install -r requirements.txt
python scripts/scan_corpus.py --sample-fraction 1.0
python scripts/run_week1_gates.py
python scripts/run_week2_pipeline.py
python scripts/run_rehearsal.py --dev --synthetic
python scripts/run_main_runs.py
python scripts/check_runs.py
python scripts/measure_flops.py
python scripts/analyze_exchange_rate.py
python scripts/plot_results.py
python scripts/make_report.py
```

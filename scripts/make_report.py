#!/usr/bin/env python3
"""Week 6: 최종 보고서와 발표 개요를 분석 산출물에서 생성 (Subtask 1.4).

숫자를 손으로 옮기지 않는다. 분석을 다시 돌리면 보고서도 다시 생성하면 된다.
한계(limitations)는 하드코딩되어 있는데, 이는 결과에 따라 달라지는 것이 아니라
설계에서 이미 정해진 것이기 때문이다 - 마지막에 발견되는 것이 아니라 처음부터
명시된 것이어야 한다.

Generates the final report and the presentation outline from the analysis JSON.
Limitations are fixed text because they follow from the design, not from the
result: they are stated up front, not discovered at the end.

Usage:
  python scripts/make_report.py
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args
from src.logging_utils import banner, load_json
from src.reporting import (
    break_even_table, exchange_rate_table, fixed_budget_table, headline,
    mechanism_table_markdown, rounds_table,
)

LIMITATIONS = """
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
"""


def build_report(cfg, report: dict) -> str:
    metric = report["metric"].upper()
    entry_tau = report["representative_tau"]
    flops = report["flops"]
    return f"""# Buying Back Communication with Retrieval Compute

**The Exchange Rate of Client-Side Reranking in Federated LLM Fine-Tuning**

*Generated from `results/logs/analysis/exchange_rate.json` on {date.today().isoformat()}.*

---

## 1. Summary

{headline(report)}

## 2. What the question was

Total federated communication is `R x N x |AB| x 2`. PEFT, quantisation and
sparsification all shrink `|AB|`. This project asked whether `R` can be reduced
from a different direction — by improving the retrieval that constructs each
client's training inputs — and at what exchange rate against the local compute
that improvement costs.

The treatment was **reranking depth** `d`, the number of dense candidates
rescored by a distilled cross-encoder before the top 3 are selected, at
`d ∈ {cfg.treatment.depths}`. Every condition drew from one common top-50 pool
and placed exactly three passages in every prompt; `d = 3` was a control that
reorders without promoting. Everything else was held identical.

## 3. Protocol

* {report['num_clients']} clients, FedAvg, full participation, Flower simulation.
* LoRA payload **{report['payload_bytes_per_client_per_direction'] / 1e6:.2f} MB** per client per direction.
* Identical communication budget per run, **no early stopping**; `K`, `S` and `R`
  fixed by the Week-1 pilot rather than by convention.
* A fixed {cfg.data.eval_set_size}-question held-out set, identical across runs, scored
  centrally on the global model with greedy decoding and the official NQ
  normalisation.
* **Evaluation context held at d = 0 for every condition**, isolating what the
  model learned from reranking rather than what better inference-time context
  supplies.
* {len(report['tau_sweep'])} τ points swept across the range the baseline attains.

## 4. What the fixed budget bought

{fixed_budget_table(report)}

## 5. Rounds to target and communication saved

Read at a representative τ = {entry_tau:.2f} {metric}; the full sweep is in
`saving_vs_tau.png` and in the analysis JSON. Conditions that never reach τ are
reported as such and are not extrapolated.

{rounds_table(report)}

## 6. The exchange rate

FLOPs per reranked pair: {flops['per_pair']:.3e} ({flops['source']['rerank']}).
FLOPs per indexed passage: {flops['per_passage']:.3e} ({flops['source']['embed']}).
Device throughput assumed: {report['device_throughput_flops']:.3e} FLOP/s.

{exchange_rate_table(report)}

The bytes-per-FLOP figure is the only assumption-free number here: it carries no
bandwidth and no device. Everything below is that number multiplied by a
deployment's parameters.

## 7. Break-even regime

Reranking buys back communication on links **slower** than the contour below,
and the contour falls as query volume rises, because the recurring per-query cost
accumulates while the one-off saving does not.

{break_even_table(report)}

## 8. Mechanism

The proposed path was: depth ↑ → retrieval quality ↑ → clients learn the same
task → update divergence ↓ → rounds ↓.

{mechanism_table_markdown(report)}

**Verdict: {report['mechanism']['verdict']}** — {report['mechanism']['note']}

Correlations across depths: recall gain vs divergence reduction
{_c(report['mechanism']['corr_recall_vs_divergence_reduction'])}, recall gain vs
rounds saved {_c(report['mechanism']['corr_recall_vs_rounds_saved'])}, divergence
reduction vs rounds saved
{_c(report['mechanism']['corr_divergence_reduction_vs_rounds_saved'])}.

## 9. Limitations
{LIMITATIONS}
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
"""


def build_presentation(cfg, report: dict) -> str:
    entry = min(report["tau_sweep"], key=lambda e: abs(e["tau"] - report["representative_tau"]))
    deepest = max(int(d) for d in entry["depths"])
    info = entry["depths"][str(deepest)]
    saved = "n/a" if info["saved_mb"] is None else f"{info['saved_mb']:.1f} MB"
    return f"""# Presentation outline

*Generated on {date.today().isoformat()}.*

## Slide 1 — The overlooked factor

Federated communication = **bytes per round × rounds required**.
Everyone shrinks the first. This project attacks the second — from retrieval.

## Slide 2 — The coupling nobody measured

Institutions that cannot centralise data deploy RAG, so training should retrieve
too. When retrieval succeeds every client learns the same task; when it fails
each client adapts to its own pattern of irrelevant context — and non-IID updates
diverge and slow convergence.

## Slide 3 — One axis: reranking depth

`d ∈ {cfg.treatment.depths}`, one common top-50 pool, exactly three passages in
every prompt. `d = 3` reorders without promoting — an ordering control.
Depth is a **recurring** per-query cost, so it is commensurable with the
per-round communication it trades against.

## Slide 4 — Measuring without a stopping rule

Fixed budget, no early stopping, curves interpolated. LoRA converges in a few
rounds, so integer round count is too coarse a ruler. `R = S / K`, and `K` was
fixed by a pilot that found the smallest `K` with measurable client drift.

## Slide 5 — Result

{headline(report)}

At τ = {entry['tau']:.2f}: **{saved}** saved at d = {deepest}.

## Slide 6 — The exchange rate, not a verdict

Bytes saved per extra FLOP → break-even bandwidth. Reranking pays on slow links
and loses as query volume grows. The deliverable is an **operating regime**.

## Slide 7 — Honest limits

Five seeds, one model scale, simulated clients, answer-string gold labels, and
a cost model whose assumptions are stated rather than hidden.
"""


def _c(value) -> str:
    return "n/a" if value is None else f"r = {value:+.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--analysis", default=None)
    args = parser.parse_args()
    cfg = config_from_args(args)

    analysis_path = Path(
        args.analysis or Path(cfg.paths.log_dir) / "analysis" / "exchange_rate.json"
    )
    if not analysis_path.exists():
        print(f"{analysis_path} not found; run scripts/analyze_exchange_rate.py first.")
        return 1

    report = load_json(analysis_path)
    report_dir = Path(cfg.paths.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    banner("Week 6 - generating the report from the analysis artefact")
    for name, text in (
        ("final_report.md", build_report(cfg, report)),
        ("presentation.md", build_presentation(cfg, report)),
    ):
        (report_dir / name).write_text(text)
        print(f"  wrote {report_dir / name}")

    print("\n  numbers are generated, not transcribed: re-run this after any re-analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

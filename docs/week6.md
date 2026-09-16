# Week 6 — Report, presentation and release

**Plan subtask:** 1.4
**Outcome:** the final deliverables.

```bash
python scripts/make_report.py     # reports/final_report.md + reports/presentation.md
```

---

## The report is generated, not transcribed

Every number in `reports/final_report.md` is read from
`results/logs/analysis/exchange_rate.json` at generation time. Nothing is copied
by hand.

The reason is practical. Re-run the analysis with a different device-throughput
assumption, or with measured rather than analytic FLOPs, and a hand-written
report silently keeps describing the old numbers. In week 6 of a six-week
placement there is no time left to find that mismatch — so the prose is
regenerated instead of maintained.

What is *not* generated:

* **The limitations section.** It follows from the design, not the result, so it
  is fixed text — stated up front, not discovered at the end.
* **The headline sentence** is generated but branches on the sign of the result:
  a saving, a *loss* ("the treatment cost communication rather than saving it"),
  or a condition that never reached τ. A negative result is stated plainly. The
  plan committed to producing "a clear positive or negative result", and the
  reporting code holds that commitment rather than leaving it to the writer's
  mood on the day.

---

## Report structure

| Section | Source |
|---|---|
| 1. Summary | `headline()` — direction, magnitude, mechanism verdict |
| 2–3. Question and protocol | config values, so the report cannot describe a setup that was not run |
| 4. What the fixed budget bought | final scores per depth, mean ± SE |
| 5. Rounds to target | interpolated rounds and saved MB at a representative τ |
| 6. Exchange rate | bytes saved per extra FLOP, under both indexing assumptions |
| 7. Break-even regime | contour over query volume |
| 8. Mechanism | recall gain vs divergence reduction, with the verdict |
| 9. Limitations | fixed text (below) |
| 10–11. Figures and reproduction | pointers and the exact command sequence |

---

## The limitations, stated up front

These were written into the plan before any measurement and are carried into the
report unchanged:

* **Five seeds** — effect sizes and standard errors, **no significance tests**.
* **A single model scale** (135M, LoRA r = 8).
* **A fixed round horizon** — nothing is observed beyond the budget, and the τ
  sweep is bounded by what the baseline reached within it.
* **A fixed client partition** — heterogeneity was a background condition, not a
  factor, so the result says nothing about other heterogeneity regimes.
* **Simulated clients** — communication is accounted analytically from payload
  size; no real link was measured.
* **Answer-string matching as a proxy for gold labels** — a passage containing
  the answer string is not necessarily supporting evidence, and both the recall
  metric and the corpus construction inherit that approximation.
* **Cost-model assumptions** — throughput, amortisation and measured-vs-analytic
  FLOPs all move the contour; each is reported next to the numbers it produced.
* **Cached retrieval** — retrieval inside a live training loop was never timed;
  the cost model charges the deployment-time cost instead.

---

## Presentation

`reports/presentation.md` is a seven-slide outline built from the same JSON:
the overlooked factor, the unmeasured coupling, the single axis, measuring
without a stopping rule, the result, the exchange rate as a regime, and the
honest limits.

---

## Release

The repository is the second deliverable. What a reader can reuse independently
of this result:

| Artefact | Reusable as |
|---|---|
| `src/selection.py` | reranking depth as a continuous, cost-dialled treatment |
| `src/corpus_scan.py`, `src/corpus_build.py` | NQ + `psgs_w100` gold labelling and hard-negative corpus construction |
| `src/partitioning.py` | equal-size topic-disjoint federated partitioning |
| `src/retrieval_cache.py` | one frozen candidate pool serving many retrieval conditions |
| `src/fl_runner.py` | fixed-budget federated runs with byte, drift and treatment logging |
| `src/analysis/` | τ-swept interpolation and the two-term cost model |

```bash
bash scripts/run_all.sh          # everything, weeks 1-6
pytest -q                        # 153 tests, no GPU or network
```

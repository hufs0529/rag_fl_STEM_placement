# Reports

`final_report.md` and `presentation.md` in this directory are **generated**, not
written by hand:

```bash
python scripts/analyze_exchange_rate.py
python scripts/plot_results.py
python scripts/make_report.py
```

Every number in them is read from
`results/logs/analysis/exchange_rate.json`. Transcribing numbers into prose is
how a report ends up describing an earlier version of the analysis, and there is
no week left at the end of a six-week placement to find that mismatch — so the
prose is regenerated instead.

What is **not** generated:

* the limitations section, which follows from the design rather than the result
  and is therefore fixed text stated up front;
* any narrative interpretation beyond the headline sentence, which states the
  sign and magnitude of the effect and the mechanism verdict.

Figures referenced by the report live in `results/figures/` and are produced by
`scripts/plot_results.py`.

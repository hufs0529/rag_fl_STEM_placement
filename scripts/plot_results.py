#!/usr/bin/env python3
"""Week 5: 보고서용 그림 생성 (Subtask 1.4).

네 장:
  1) learning_curves      - depth 별 시드 평균 곡선 ±표준오차 (x축: 라운드와 누적 MB)
  2) saving_vs_tau        - tau 스윕에 따른 절감 바이트 곡선 (단일 tau 에 의존하지 않음을 보임)
  3) break_even           - (대역폭 x 쿼리 수) 평면의 손익분기 곡선
  4) mechanism            - recall 증가 vs divergence 감소

그림은 analyze_exchange_rate.py 가 남긴 JSON 만 읽는다.

Usage:
  python scripts/plot_results.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import add_config_args, config_from_args
from src.logging_utils import banner, load_json

COLOURS = {0: "#444444", 3: "#1f77b4", 10: "#ff7f0e", 50: "#d62728"}


def _colour(depth):
    return COLOURS.get(int(depth), None)


def plot_learning_curves(report, path, metric):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    payload = report["payload_bytes_per_client_per_direction"]
    clients = report["num_clients"]

    for depth, curve in sorted(report["mean_curves"].items(), key=lambda kv: int(kv[0])):
        rounds = [p["round"] for p in curve]
        means = [p["mean"] for p in curve]
        errs = [p["stderr"] or 0.0 for p in curve]
        megabytes = [r * clients * payload * 2 / 1e6 for r in rounds]
        label = f"d = {depth}" + (" (baseline)" if int(depth) == 0 else
                                  " (control)" if int(depth) == 3 else "")
        for ax, x in zip(axes, (rounds, megabytes)):
            ax.plot(x, means, marker="o", ms=3, label=label, color=_colour(depth))
            ax.fill_between(x, np.array(means) - np.array(errs),
                            np.array(means) + np.array(errs), alpha=0.15, color=_colour(depth))

    axes[0].set_xlabel("federated round")
    axes[1].set_xlabel("cumulative communication (MB)")
    for ax in axes:
        ax.set_ylabel(f"held-out {metric.upper()}")
        ax.grid(alpha=0.3)
    axes[0].set_title("Fixed budget, no early stopping")
    axes[1].set_title("The same curves priced in bytes")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_saving_vs_tau(report, path, metric):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    baseline = str(report["baseline_depth"])
    depths = [d for d in report["mean_curves"] if d != baseline]

    for depth in sorted(depths, key=int):
        taus, saved = [], []
        for entry in report["tau_sweep"]:
            value = entry["depths"][depth]["saved_mb"]
            if value is not None:
                taus.append(entry["tau"])
                saved.append(value)
        if taus:
            ax.plot(taus, saved, marker="o", ms=3, label=f"d = {depth}", color=_colour(depth))

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel(f"target accuracy tau (held-out {metric.upper()})")
    ax.set_ylabel("communication saved vs d=0 (MB)")
    ax.set_title("Saving is reported as a curve, not at one chosen tau")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_break_even(report, path):
    surfaces = report["break_even_surfaces"]
    keys = [k for k in surfaces if k.endswith("amortised")]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for key in sorted(keys, key=lambda k: surfaces[k]["depth"]):
        contour = surfaces[key]["break_even_contour"]
        q = [p["query_volume"] for p in contour]
        b = [p["break_even_bandwidth_mbps"] for p in contour]
        if any(v is None for v in b):
            continue
        depth = surfaces[key]["depth"]
        ax.plot(q, b, marker="o", ms=3, label=f"d = {depth}", color=_colour(depth))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("query volume the corpus must serve")
    ax.set_ylabel("break-even bandwidth (Mbps)")
    ax.set_title("Reranking pays for itself BELOW each contour")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_mechanism(report, path):
    rows = report["mechanism"]["rows"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax2 = ax.twinx()
    depths = [str(r["depth"]) for r in rows]
    recall = [r["gold_recall_at_3"] for r in rows]
    divergence = [r["divergence"] for r in rows]

    ax.bar(depths, [v if v is not None else 0 for v in recall], alpha=0.6,
           color="#1f77b4", label="retrieval quality proxy")
    ax2.plot(depths, divergence, marker="o", color="#d62728", label="client-update divergence")
    ax.set_xlabel("reranking depth d")
    ax.set_ylabel("rank-change rate (treatment strength)", color="#1f77b4")
    ax2.set_ylabel("mean client-update divergence", color="#d62728")
    ax.set_title(f"Mechanism check: {report['mechanism']['verdict']}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    cfg = config_from_args(args)

    report_path = Path(args.report or Path(cfg.paths.log_dir) / "analysis" / "exchange_rate.json")
    if not report_path.exists():
        print(f"{report_path} not found; run scripts/analyze_exchange_rate.py first.")
        return 1

    report = load_json(report_path)
    figure_dir = Path(cfg.paths.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metric = report["metric"]

    banner("Week 5 - figures")
    plots = [
        ("learning_curves.png", plot_learning_curves, (report, figure_dir / "learning_curves.png", metric)),
        ("saving_vs_tau.png", plot_saving_vs_tau, (report, figure_dir / "saving_vs_tau.png", metric)),
        ("break_even.png", plot_break_even, (report, figure_dir / "break_even.png")),
        ("mechanism.png", plot_mechanism, (report, figure_dir / "mechanism.png")),
    ]
    for name, fn, fn_args in plots:
        fn(*fn_args)
        print(f"  wrote {figure_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

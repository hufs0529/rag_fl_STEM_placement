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

# 깊이는 순서가 있지만, 이 그림들이 하는 일은 **정체성**(어느 곡선이 어느 조건인지)
# 이다. 단일 색조 램프로 4 단계를 쓰면 light 표면에서 인접 단계의 normal-vision
# 분리도가 기준 15 를 넘지 못한다(실측 최선 14.5) - 그래서 처리 조건에는 검증된
# 범주형 순서를 쓰고, 순서 정보는 범례 순서가 나른다.
#
# d=0 은 계열이 아니라 **기준선**이므로 중립 회색으로 둔다.
# 처리 5조건은 범주형 슬롯 1-5 를 깊이 오름차순으로 고정 배정한다(값에 묶이므로
# 조건이 늘거나 줄어도 색이 바뀌지 않는다).
#
# 검증(light, 표면 #fcfcfb): 슬롯 1-8 전 항목 통과. 최악 인접 CVD dE 9.1,
# normal-vision dE 19.6. 세 색이 표면 대비 3:1 미만이라 **relief 규칙**이 걸리는데,
# 범례가 항상 있고 make_report.py 가 같은 수치를 표로 내보내므로 충족된다.
COLOURS = {
    0: "#52514e",    # 기준선 - 중립 회색 (계열이 아니라 참조선)
    3: "#2a78d6",    # 순서 통제   | 아래는 범주형 슬롯 1-7 을
    5: "#eb6834",    # 최소 처리   | 깊이 오름차순으로 고정 배정한 것이다.
    7: "#1baf7a",    #             | 색이 깊이 **값**에 묶이므로 조건이
    10: "#eda100",   #             | 늘거나 줄어도 남은 색이 바뀌지 않는다.
    20: "#e87ba4",
    25: "#008300",
    30: "#6250d6",
    50: "#e34948",
}


def _colour(depth):
    """설정에 있는 깊이는 반드시 색이 지정돼 있어야 한다.

    None 을 돌려주면 matplotlib 이 기본 순환 색을 쓰고, 그것이 다른 깊이의 색과
    겹쳐도 그림은 조용히 그려진다. 깊이를 추가했을 때 가장 먼저 깨지는 곳이다.
    """
    d = int(depth)
    if d not in COLOURS:
        raise KeyError(
            f"depth {d} 에 지정된 색이 없다. scripts/plot_results.py 의 COLOURS 에 "
            f"추가하고 dataviz 검증기로 팔레트를 다시 확인할 것 "
            f"(현재 지정: {sorted(COLOURS)})"
        )
    return COLOURS[d]


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
    """처리 강도와 발산을 **두 패널**로 나란히 둔다.

    이중축으로 그리면 안 된다. 이 그림의 요점은 "왼쪽은 크게 움직이고 오른쪽은
    거의 움직이지 않는다" 인데, 발산을 자기 범위(1.313~1.322)에 맞춰 늘리면
    0.77 % 변화가 급락처럼 보여 **본문과 정반대로 읽힌다.** 그래서 두 패널로
    쪼개고, 발산 축은 0 에서 시작해 실제 크기를 보이게 한다. 진짜 범위는
    확대 상자로 따로 보여 준다 - 숨기지 않되 주축을 왜곡하지 않는다.
    """
    rows = report["mechanism"]["rows"]
    depths = [str(r["depth"]) for r in rows]
    strength = [r["gold_recall_at_3"] or 0.0 for r in rows]
    divergence = [r["divergence"] for r in rows]
    lo, hi = min(divergence), max(divergence)
    span = (hi - lo) / hi * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax1.bar(depths, strength, color="#2E86C1", alpha=0.85)
    ax1.set_xlabel("reranking depth d")
    ax1.set_ylabel("rank-change rate")
    ax1.set_ylim(0, 0.8)
    ax1.set_title("Treatment strength rose")
    ax1.grid(axis="y", alpha=0.3)

    # 0 에서 시작하는 축. 변화가 눈에 안 보이는 것이 결과다.
    ax2.plot(depths, divergence, marker="o", color="#C1562B", linewidth=2)
    ax2.set_xlabel("reranking depth d")
    ax2.set_ylabel("mean client-update divergence")
    ax2.set_ylim(0, max(1.4, hi * 1.08))
    ax2.set_title(f"Divergence did not ({span:.2f} % span)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.annotate(f"full range {lo:.4f} - {hi:.4f}",
                 xy=(0.5, 0.12), xycoords="axes fraction", ha="center",
                 fontsize=9, color="#4A5568")

    fig.suptitle(f"Mechanism check: {report['mechanism']['verdict']}", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    cfg = config_from_args(args)

    variant = str(cfg.treatment.get("variant", "") or "")
    stem = f"exchange_rate__{variant}" if variant else "exchange_rate"
    report_path = Path(args.report
                       or Path(cfg.paths.log_dir) / "analysis" / f"{stem}.json")
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

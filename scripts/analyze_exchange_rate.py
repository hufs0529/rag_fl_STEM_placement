#!/usr/bin/env python3
"""Week 5: tau 스윕 보간 -> 절감 바이트 -> 2항 비용 모델 -> 손익분기 (Subtask 1.4).

절차:
  1) 20 run 의 곡선을 읽는다 (results/logs/runs/*.jsonl)
  2) baseline(d=0)이 실제로 도달한 정확도 구간에서 tau 를 스윕
  3) 각 tau 에서 depth 별 rounds-to-tau 를 시드마다 보간하고 평균±표준오차
  4) 라운드 차이를 R x N x |AB| x 2 로 바이트 환산
  5) reranking FLOPs 를 그 절감에 청구 (일회성 indexing / 반복 rerank 2항)
  6) (대역폭 x 쿼리 수) 평면의 순이득 격자와 손익분기 곡선
  7) 기전 점검: recall 증가가 divergence 감소와 함께 오는가

어떤 조건이 tau 에 도달하지 못한 구간은 외삽하지 않고 "도달 못 함"으로 남긴다.

Usage:
  python scripts/analyze_exchange_rate.py
  python scripts/analyze_exchange_rate.py --set cost.device_throughput_flops=2e12
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.cost_model import (
    break_even_bandwidth_mbps, break_even_surface, depth_cost, exchange_rate, saved_bytes,
)
from src.analysis.curves import (
    attainable_range, curve_mean, final_scores, local_slope, matched_budget_table,
    rounds_by_depth, tau_grid,
)
from src.analysis.diagnostics import assess_mechanism, mean_divergence, mechanism_table
from src.config import add_config_args, config_from_args
from src.logging_utils import banner, load_json, read_jsonl, save_json
from src.run_matrix import enumerate_runs, group_by_depth, is_complete, load_curve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--calibrated", default=None)
    parser.add_argument("--flops", default=None, help="results/logs/week4/flops.json")
    parser.add_argument("--metric", default=None, help="기본은 eval.primary_metric (f1)")
    args = parser.parse_args()
    cfg = config_from_args(args)

    metric = args.metric or cfg.eval.primary_metric
    log_dir = Path(cfg.paths.log_dir)
    specs = enumerate_runs(cfg, args.calibrated)

    banner(f"Week 5 / Subtask 1.4 - exchange rate and break-even | metric = {metric}")

    # --- 1) 곡선 수집 -------------------------------------------------------
    curves_by_depth, summaries_by_depth = {}, {}
    for depth, group in sorted(group_by_depth(specs).items()):
        curves, summaries = [], []
        for spec in group:
            if not is_complete(cfg, spec):
                print(f"  skipping {spec.name}: not complete")
                continue
            curves.append(load_curve(cfg, spec))
            summaries.append(load_json(log_dir / "runs" / f"{spec.name}_summary.json"))
        curves_by_depth[depth] = curves
        summaries_by_depth[depth] = summaries
        print(f"  d={depth:>2}: {len(curves)} complete runs")

    baseline_depth = min(curves_by_depth)
    if not curves_by_depth[baseline_depth]:
        print("no baseline runs found; run scripts/run_main_runs.py first.")
        return 1

    num_clients = summaries_by_depth[baseline_depth][0]["num_clients"]
    payload = summaries_by_depth[baseline_depth][0]["payload_bytes_per_client_per_direction"]

    # --- 2-4) tau 스윕과 절감 바이트 ----------------------------------------
    tau_low, tau_high = attainable_range(curves_by_depth[baseline_depth], metric)
    taus = tau_grid(curves_by_depth[baseline_depth], metric=metric)
    print(f"\n  tau swept over the range the baseline attains: "
          f"{taus[0]:.2f} .. {taus[-1]:.2f} {metric.upper()} ({len(taus)} points)")

    sweep = []
    for tau in taus:
        per_depth = rounds_by_depth(curves_by_depth, tau, metric)
        baseline_rounds = per_depth[baseline_depth]["mean"]
        entry = {"tau": tau, "baseline_rounds": baseline_rounds, "depths": {}}
        for depth, stats in per_depth.items():
            saved = saved_bytes(baseline_rounds, stats["mean"], num_clients, payload)
            entry["depths"][str(depth)] = {
                "rounds_mean": stats["mean"],
                "rounds_stderr": stats["stderr"],
                "seeds_reaching_tau": stats["n"],
                "seeds_not_reaching_tau": stats["missing"],
                "rounds_saved": (
                    None if stats["mean"] is None or baseline_rounds is None
                    else baseline_rounds - stats["mean"]
                ),
                "saved_bytes": saved,
                "saved_mb": None if saved is None else saved / 1e6,
            }
        sweep.append(entry)

    # --- 5-6) 비용 모델과 손익분기 -----------------------------------------
    flops_path = Path(args.flops or (log_dir / "week4" / "flops.json"))
    if flops_path.exists():
        flops = load_json(flops_path)
        flops_per_pair = flops["rerank"]["flops"]
        flops_per_passage = flops["embed"]["flops"]
        flops_source = {k: flops[k]["source"] for k in ("rerank", "embed")}
    else:
        from src.flops import cross_encoder_flops_per_pair, embedder_flops_per_passage
        flops_per_pair = cross_encoder_flops_per_pair()
        flops_per_passage = embedder_flops_per_passage()
        flops_source = {"rerank": "analytic", "embed": "analytic"}
        print(f"  {flops_path} not found; using analytic FLOPs estimates")

    # 대표 tau: baseline 이 도달한 구간의 중앙값. 결론 자체는 스윕 전체로 보고하고,
    # 이 지점은 표 하나로 요약하기 위한 예시일 뿐이다.
    midpoint = sweep[len(sweep) // 2]
    throughput = float(cfg.cost.device_throughput_flops)
    surfaces, rates = {}, {}
    for depth in sorted(curves_by_depth):
        if depth == baseline_depth:
            continue
        saved = midpoint["depths"][str(depth)]["saved_bytes"]
        for amortise in (True, False):
            cost = depth_cost(depth, float(cfg.cost.query_volume_sweep[2]),
                              cfg.data.corpus_size, flops_per_pair, flops_per_passage, amortise)
            key = f"d{depth}_{'amortised' if amortise else 'with_indexing'}"
            rates[key] = {
                "saved_bytes": saved,
                "total_flops": cost["total_flops"],
                "bytes_per_flop": exchange_rate(saved, cost["total_flops"]),
                **cost,
            }
            surfaces[key] = break_even_surface(
                saved, depth,
                cfg.cost.bandwidth_sweep_mbps, cfg.cost.query_volume_sweep,
                cfg.data.corpus_size, flops_per_pair, flops_per_passage,
                throughput, amortise,
            )

    # --- 7) 기전 점검 -------------------------------------------------------
    divergence_by_depth, recall_by_depth = {}, {}
    for depth, group in sorted(group_by_depth(specs).items()):
        values, recalls = [], []
        for spec in group:
            path = log_dir / "runs" / f"{spec.name}.jsonl"
            if not path.exists():
                continue
            records = read_jsonl(path)
            value = mean_divergence(records, first_n_rounds=cfg.eval.round_schedule_dense_until)
            if value is not None:
                values.append(value)
            training_recall = [
                r["rank_change_rate"] for r in records
                if r.get("event") == "round" and r.get("rank_change_rate") is not None
            ]
            if training_recall:
                recalls.append(sum(training_recall) / len(training_recall))
        if values:
            divergence_by_depth[depth] = sum(values) / len(values)
        if recalls:
            recall_by_depth[depth] = sum(recalls) / len(recalls)

    rounds_saved = {
        depth: midpoint["depths"][str(depth)]["rounds_saved"] for depth in curves_by_depth
    }
    table = mechanism_table(recall_by_depth, divergence_by_depth, rounds_saved)
    mechanism = assess_mechanism(table)

    report = {
        "metric": metric,
        "baseline_depth": baseline_depth,
        "num_clients": num_clients,
        "payload_bytes_per_client_per_direction": payload,
        "runs_per_depth": {str(d): len(c) for d, c in curves_by_depth.items()},
        "final_scores_at_fixed_budget": {
            str(d): final_scores(c, metric) for d, c in curves_by_depth.items()
        },
        "mean_curves": {str(d): curve_mean(c, metric) for d, c in curves_by_depth.items()},
        # 1 차 결과: 같은 예산에서의 metric 차이. tau 를 고르지 않으므로
        # "왜 그 tau 인가" 가 생기지 않고, 비단조 곡선에도 영향받지 않는다.
        "matched_budget": matched_budget_table(curves_by_depth, baseline_depth, metric),
        "tau_sweep": sweep,
        "tau_range": {"low": tau_low, "high": tau_high,
                      "low_rule": "max over seeds of the baseline at round 1",
                      "high_rule": "min over seeds of the baseline peak",
                      "n_tau": len(taus),
                      "grid_rule": "the baseline curves' own evaluated values"},
        "representative_tau": midpoint["tau"],
        # 세로 차이를 바이트로 환산할 때 쓰는 기울기. **구간에 따라 3 배 이상
        # 달라지고 부호도 뒤집히므로** 단일 값이 아니라 범위로 남긴다.
        "baseline_slope_by_window": {
            f"r{r0}_r{r1}": local_slope(curves_by_depth[baseline_depth], r0, r1, metric)
            for r0, r1 in ((1, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 59),
                           (40, 59), (20, 59))
        },
        "flops": {"per_pair": flops_per_pair, "per_passage": flops_per_passage,
                  "source": flops_source},
        "device_throughput_flops": throughput,
        # throughput 은 배포 장치에 대한 **가정**이며 유도되지 않는다. 환율의
        # 분모이므로 2 배 틀리면 손익분기 대역폭이 2 배 틀린다. 가정 하나로
        # 보고하지 않고 흔들어 함께 남긴다.
        # 환율(bytes/FLOP)은 throughput 과 무관하다 - 분모가 FLOPs 다. throughput 은
        # 손익분기 **대역폭**에만, 그리고 선형으로 들어간다:
        #   B* = saved x 8 x T / flops   ->  T 가 2 배면 B* 도 2 배
        # 그래서 민감도는 재계산이 아니라 그 선형성을 명시해 남긴다.
        "throughput_sensitivity": {
            "note": "break-even bandwidth is linear in device_throughput_flops; "
                    "the exchange rate in bytes/FLOP does not depend on it",
            "assumed": throughput,
            "variants": {
                f"{factor:g}x": {
                    "device_throughput_flops": throughput * factor,
                    "break_even_bandwidth_multiplier": factor,
                    "break_even_bandwidth_mbps": {
                        key: (None if rates[key]["bytes_per_flop"] is None else
                              break_even_bandwidth_mbps(
                                  rates[key]["saved_bytes"],
                                  rates[key]["total_flops"],
                                  throughput * factor))
                        for key in rates
                    },
                }
                for factor in (0.5, 1.0, 2.0)
            },
        },
        "exchange_rates": rates,
        "break_even_surfaces": surfaces,
        "mechanism": {**table, **mechanism},
    }
    save_json(report, log_dir / "analysis" / "exchange_rate.json")

    # --- 출력 ---------------------------------------------------------------
    banner("matched-budget reading (primary: no tau to choose)")
    print("  same bytes spent, difference in " + metric.upper()
          + "; * = mean exceeds twice its standard error")
    for row in report["matched_budget"]:
        if row["round"] not in (1, 5, 10, 15, 20, 30, 40, 50, 59):
            continue
        cells = []
        for depth, info in row["depths"].items():
            if info["mean"] is None:
                cells.append(f"d{depth}: n/a")
                continue
            se = info["stderr"]
            star = "*" if info["separated_from_zero"] else " "
            cells.append(f"d{depth}:{info['mean']:+5.2f}"
                         + (f"+-{se:.2f}{star}" if se is not None else f"{star}"))
        gb = (row["cumulative_bytes"] or 0) / 1e9
        print(f"  r{row['round']:>2}  {gb:6.4f} GB  base {row['baseline_mean']:6.2f}   "
              + "  ".join(cells))

    banner(f"tau range [{tau_low:.2f}, {tau_high:.2f}] from {len(taus)} attained values")
    print("  low  = max over seeds of the baseline at round 1 "
          "(below it, interpolation lands inside round 1, which cannot be run)")
    print("  high = min over seeds of the baseline peak "
          "(above it the weaker seeds drop out and only the strong ones remain)")
    print("  grid = the baseline curves' own evaluated values, so no spacing is declared")

    banner("baseline slope, used to convert a matched-budget gain into bytes")
    print("  reported as a range: the window changes the answer severalfold "
          "and can flip its sign")
    for window, slope in report["baseline_slope_by_window"].items():
        if slope is None:
            continue
        note = "  <- negative: conversion undefined here" if slope <= 0 else ""
        print(f"  {window:>10}  {slope:+8.4f} {metric}/round{note}")

    banner("what the fixed budget bought")
    for depth in sorted(curves_by_depth):
        stats = report["final_scores_at_fixed_budget"][str(depth)]
        if stats["mean"] is None:
            continue
        err = f" +/- {stats['stderr']:.2f}" if stats["stderr"] is not None else ""
        print(f"  d={depth:>2}  {metric.upper()} {stats['mean']:6.2f}{err}  (n={stats['n']})")

    banner(f"rounds-to-target at the representative tau = {midpoint['tau']:.2f}")
    for depth in sorted(curves_by_depth):
        info = midpoint["depths"][str(depth)]
        if info["rounds_mean"] is None:
            print(f"  d={depth:>2}  never reaches tau in {info['seeds_not_reaching_tau']} seeds "
                  "(reported as such, not extrapolated)")
            continue
        err = f" +/- {info['rounds_stderr']:.2f}" if info["rounds_stderr"] is not None else ""
        saved = "" if info["saved_mb"] is None else f"  saved {info['saved_mb']:8.1f} MB"
        print(f"  d={depth:>2}  rounds {info['rounds_mean']:6.2f}{err}{saved}")

    banner("exchange rate and break-even")
    for key, rate in rates.items():
        if rate["bytes_per_flop"] is None:
            print(f"  {key}: no saving to charge against")
            continue
        print(f"  {key}: {rate['bytes_per_flop']:.3e} bytes saved per extra FLOP")
        contour = surfaces[key]["break_even_contour"]
        for point in contour:
            bandwidth = point["break_even_bandwidth_mbps"]
            if bandwidth is None:
                continue
            print(f"      Q={point['query_volume']:>10.0f} queries -> pays for itself on links "
                  f"slower than {bandwidth:9.3f} Mbps")

    banner("mechanism check")
    for row in table["rows"]:
        print(f"  d={row['depth']:>2}  recall gain {_fmt(row['recall_gain'])}  "
              f"divergence reduction {_fmt(row['divergence_reduction'])}  "
              f"rounds saved {_fmt(row['rounds_saved'])}")
    print(f"\n  verdict: {mechanism['verdict']} - {mechanism['note']}")
    print(f"\n  wrote {log_dir / 'analysis' / 'exchange_rate.json'}")
    return 0


def _fmt(value) -> str:
    return "    n/a" if value is None else f"{value:7.4f}"


if __name__ == "__main__":
    raise SystemExit(main())

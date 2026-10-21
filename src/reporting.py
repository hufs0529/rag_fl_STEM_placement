"""분석 JSON -> 보고서 본문 (Week 6).

보고서의 숫자를 손으로 옮겨 적지 않는다. 손으로 옮기면 분석을 다시 돌린 뒤
보고서만 옛 숫자로 남는 일이 반드시 생기고, 6주 일정의 마지막 주에 그 불일치를
찾을 시간은 없다. 대신 results/logs/analysis/exchange_rate.json 에서 직접
문장과 표를 만든다.

Report prose and tables are generated from the analysis JSON rather than
transcribed, so the report cannot drift from the numbers it describes.
"""

from typing import Dict, List, Optional


def fmt(value: Optional[float], digits: int = 2, dash: str = "n/a") -> str:
    return dash if value is None else f"{value:.{digits}f}"


def fmt_pm(stats: Dict, digits: int = 2) -> str:
    """mean +/- stderr. 유의성 검정이 아니라 효과 크기와 구간을 보고한다."""
    if stats.get("mean") is None:
        return "n/a"
    if stats.get("stderr") is None:
        return f"{stats['mean']:.{digits}f} (n={stats.get('n', 1)})"
    return f"{stats['mean']:.{digits}f} ± {stats['stderr']:.{digits}f} (n={stats['n']})"


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


def fixed_budget_table(report: Dict) -> str:
    """고정 예산이 각 조건에서 무엇을 사주었는가."""
    metric = report["metric"].upper()
    rows = []
    for depth, stats in sorted(report["final_scores_at_fixed_budget"].items(), key=lambda kv: int(kv[0])):
        rows.append([
            f"d = {depth}" + _role(int(depth)),
            fmt_pm(stats),
            str(report["runs_per_depth"].get(depth, 0)),
        ])
    return markdown_table([f"Condition", f"Held-out {metric} at the fixed budget", "Runs"], rows)


def rounds_table(report: Dict) -> str:
    """대표 tau 에서의 rounds-to-target 과 절감 바이트."""
    entry = _representative(report)
    rows = []
    for depth, info in sorted(entry["depths"].items(), key=lambda kv: int(kv[0])):
        if info["rounds_mean"] is None:
            rows.append([f"d = {depth}" + _role(int(depth)),
                         f"never reaches τ ({info['seeds_not_reaching_tau']} seeds)", "—", "—"])
            continue
        rows.append([
            f"d = {depth}" + _role(int(depth)),
            fmt_pm({"mean": info["rounds_mean"], "stderr": info["rounds_stderr"],
                    "n": info["seeds_reaching_tau"]}),
            fmt(info["rounds_saved"]),
            fmt(info["saved_mb"], 1),
        ])
    return markdown_table(
        ["Condition", "Rounds to τ", "Rounds saved", "Communication saved (MB)"], rows
    )


def exchange_rate_table(report: Dict) -> str:
    rows = []
    for key, rate in sorted(report["exchange_rates"].items()):
        depth = key.split("_")[0].lstrip("d")
        assumption = "index amortised" if key.endswith("amortised") else "indexing charged"
        rows.append([
            f"d = {depth}", assumption,
            "n/a" if rate["bytes_per_flop"] is None else f"{rate['bytes_per_flop']:.3e}",
            f"{rate['total_flops']:.3e}",
        ])
    return markdown_table(
        ["Condition", "Assumption", "Bytes saved per extra FLOP", "Total FLOPs charged"], rows
    )


def break_even_table(report: Dict) -> str:
    rows = []
    for key, surface in sorted(report["break_even_surfaces"].items()):
        if not key.endswith("amortised"):
            continue
        for point in surface["break_even_contour"]:
            bandwidth = point["break_even_bandwidth_mbps"]
            rows.append([
                f"d = {surface['depth']}",
                f"{point['query_volume']:,.0f}",
                "n/a" if bandwidth is None else f"{bandwidth:.3f}",
            ])
    return markdown_table(
        ["Condition", "Query volume", "Break-even bandwidth (Mbps) — pays off below this"], rows
    )


def mechanism_table_markdown(report: Dict) -> str:
    rows = []
    for row in report["mechanism"]["rows"]:
        rows.append([
            f"d = {row['depth']}",
            fmt(row["gold_recall_at_3"], 4),
            fmt(row["recall_gain"], 4),
            fmt(row["divergence"], 4),
            fmt(row["divergence_reduction"], 4),
            fmt(row["rounds_saved"], 2),
        ])
    return markdown_table(
        ["Condition", "Treatment strength", "Gain vs baseline",
         "Divergence", "Reduction vs baseline", "Rounds saved"], rows
    )


def headline(report: Dict) -> str:
    """한 문단 요약 - 부호와 규모를 먼저 말한다."""
    entry = _representative(report)
    metric = report["metric"].upper()
    deepest = max(int(d) for d in entry["depths"])
    info = entry["depths"][str(deepest)]
    verdict = report["mechanism"]["verdict"]

    if info["rounds_saved"] is None:
        direction = (f"At the representative target of {entry['tau']:.2f} {metric}, the deepest "
                     f"condition (d = {deepest}) did not reach the target in every seed, so no "
                     "exchange rate is defined there.")
    elif info["rounds_saved"] > 0:
        direction = (f"At the representative target of {entry['tau']:.2f} {metric}, reranking at "
                     f"d = {deepest} reached the target {info['rounds_saved']:.2f} rounds earlier "
                     f"than dense retrieval alone, worth {info['saved_mb']:.1f} MB of communication "
                     f"across {report['num_clients']} clients.")
    else:
        direction = (f"At the representative target of {entry['tau']:.2f} {metric}, reranking at "
                     f"d = {deepest} was {abs(info['rounds_saved']):.2f} rounds SLOWER than dense "
                     "retrieval alone: the treatment cost communication rather than saving it.")

    mechanism = {
        "consistent": "Recall gains coincided with reduced client-update divergence, consistent "
                      "with the proposed path.",
        "other_mechanism": "Rounds were saved without a reduction in client-update divergence, "
                           "so the gain arrives by a different mechanism than the one proposed.",
        "no_effect": "No rounds-to-target saving was observed, so there is no mechanism to explain.",
        "within_noise": "The largest rounds-to-target saving is inside its own uncertainty, so "
                        "this measurement resolves no saving and there is no mechanism to "
                        "explain.",
    }[verdict]

    return direction + " " + mechanism


def _representative(report: Dict) -> Dict:
    target = report["representative_tau"]
    return min(report["tau_sweep"], key=lambda e: abs(e["tau"] - target))


def _role(depth: int) -> str:
    return {0: " (baseline)", 3: " (control)"}.get(depth, "")

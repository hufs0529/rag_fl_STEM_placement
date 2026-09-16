"""분석 스크립트 두 개의 종단 스모크 테스트.

가짜 run 로그를 만들어 analyze_exchange_rate.py -> plot_results.py 를 실제로
돌린다. 학습 없이 분석 경로만 검증할 수 있어야 한다는 것이 Week 5 설계의
전제이므로(분석은 results/logs 만 읽는다), 이 테스트가 그 전제를 확인한다.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPTHS = [0, 3, 10, 50]
SEEDS = [1, 2, 3, 4, 5]
PAYLOAD = 1_840_000
CLIENTS = 8
ROUNDS = 20
EVAL_ROUNDS = list(range(1, 11)) + [15, 20]


def _curve_value(depth, round_number, seed):
    """depth 가 클수록 같은 라운드에서 더 높은 F1 에 도달하는 합성 곡선."""
    ceiling = 30.0 + depth * 0.06
    rate = 0.18 + depth * 0.004
    return ceiling * (1 - 2.718281828 ** (-rate * round_number)) + 0.1 * seed


def _write_run(log_dir: Path, depth: int, seed: int):
    runs = log_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    name = f"d{depth}_s{seed}"
    summary = {
        "run": name, "depth": depth, "seed": seed, "partition_seed": 1001,
        "rounds": ROUNDS, "num_clients": CLIENTS, "local_steps_per_round": 16,
        "total_local_steps": 320, "eval_rounds": EVAL_ROUNDS, "early_stopping": False,
        "payload_bytes_per_client_per_direction": PAYLOAD,
        "total_bytes": ROUNDS * CLIENTS * PAYLOAD * 2,
        "trainable_parameters": 920_000, "completed": True,
        "final": {"f1": _curve_value(depth, ROUNDS, seed), "em": 10.0},
    }
    (runs / f"{name}_summary.json").write_text(json.dumps(summary))

    with (runs / f"{name}.jsonl").open("w") as fh:
        fh.write(json.dumps({"event": "meta", **summary}) + "\n")
        for rnd in range(1, ROUNDS + 1):
            record = {
                "event": "round", "round": rnd,
                "cumulative_bytes": rnd * CLIENTS * PAYLOAD * 2,
                "train_loss": 2.0 - 0.01 * rnd,
                "rank_change_rate": 0.0 if depth == 0 else 0.1 + 0.01 * depth,
                "promotion_rate": 0.0 if depth <= 3 else 0.02 * depth,
                # depth 가 클수록 drift 가 작다 - 기전 점검이 무언가를 찾을 수 있도록
                "drift_divergence": 0.09 - 0.0004 * depth,
                "drift_mean_pairwise_cosine": 0.3,
                "evaluated": rnd in EVAL_ROUNDS,
            }
            if record["evaluated"]:
                record["f1"] = _curve_value(depth, rnd, seed)
                record["em"] = record["f1"] * 0.6
                record["eval_gold_recall_at_3"] = 0.41
            fh.write(json.dumps(record) + "\n")


@pytest.fixture
def analysis_workspace(tmp_path):
    log_dir = tmp_path / "logs"
    (log_dir / "gates").mkdir(parents=True)
    (log_dir / "gates" / "calibrated_schedule.json").write_text(json.dumps(
        {"train": {"local_steps_per_round": 16, "total_local_steps": 320, "rounds": ROUNDS}}
    ))
    for depth in DEPTHS:
        for seed in SEEDS:
            _write_run(log_dir, depth, seed)
    return tmp_path, log_dir


def _run(script, tmp_path, log_dir, *extra):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script),
         "--set", f"paths.log_dir={log_dir}",
         "--set", f"paths.figure_dir={tmp_path / 'figures'}",
         *extra],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_analysis_runs_from_logs_alone_and_reports_a_saving(analysis_workspace):
    tmp_path, log_dir = analysis_workspace
    proc = _run("analyze_exchange_rate.py", tmp_path, log_dir)
    assert proc.returncode == 0, proc.stderr

    report = json.loads((log_dir / "analysis" / "exchange_rate.json").read_text())
    assert report["baseline_depth"] == 0
    assert report["runs_per_depth"] == {"0": 5, "3": 5, "10": 5, "50": 5}

    # tau 스윕이 baseline 도달 구간 위에 있고, 단일 tau 에 의존하지 않는다
    assert len(report["tau_sweep"]) > 1
    entry = report["tau_sweep"][len(report["tau_sweep"]) // 2]
    assert entry["depths"]["50"]["rounds_saved"] > 0
    assert entry["depths"]["50"]["saved_bytes"] > 0

    # 환율이 바이트/FLOP 로 보고된다
    rate = report["exchange_rates"]["d50_amortised"]
    assert rate["bytes_per_flop"] > 0
    # 2항 모델: 상각 가정에서는 일회성 항이 0
    assert rate["indexing_flops"] == 0.0
    assert report["exchange_rates"]["d50_with_indexing"]["indexing_flops"] > 0


def test_break_even_is_a_surface_over_bandwidth_and_query_volume(analysis_workspace):
    tmp_path, log_dir = analysis_workspace
    assert _run("analyze_exchange_rate.py", tmp_path, log_dir).returncode == 0
    report = json.loads((log_dir / "analysis" / "exchange_rate.json").read_text())

    surface = report["break_even_surfaces"]["d50_amortised"]
    grid = surface["net_benefit_seconds"]
    assert len(grid) == len(surface["query_volumes"])
    assert len(grid[0]) == len(surface["bandwidths_mbps"])

    contour = surface["break_even_contour"]
    bandwidths = [p["break_even_bandwidth_mbps"] for p in contour]
    # 쿼리가 많아질수록 손익분기 대역폭이 내려간다 (반복 비용이 쌓이므로)
    assert bandwidths == sorted(bandwidths, reverse=True)


def test_mechanism_check_reports_a_verdict(analysis_workspace):
    tmp_path, log_dir = analysis_workspace
    assert _run("analyze_exchange_rate.py", tmp_path, log_dir).returncode == 0
    report = json.loads((log_dir / "analysis" / "exchange_rate.json").read_text())
    assert report["mechanism"]["verdict"] in {"consistent", "other_mechanism", "no_effect"}
    # 합성 로그는 depth 가 클수록 drift 가 작도록 만들었다
    assert report["mechanism"]["verdict"] == "consistent"


def test_figures_are_produced_from_the_analysis_json(analysis_workspace):
    tmp_path, log_dir = analysis_workspace
    assert _run("analyze_exchange_rate.py", tmp_path, log_dir).returncode == 0
    proc = _run("plot_results.py", tmp_path, log_dir)
    assert proc.returncode == 0, proc.stderr
    figures = tmp_path / "figures"
    for name in ("learning_curves.png", "saving_vs_tau.png", "break_even.png", "mechanism.png"):
        assert (figures / name).exists() and (figures / name).stat().st_size > 0

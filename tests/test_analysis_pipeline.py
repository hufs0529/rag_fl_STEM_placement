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

# 행렬은 **출하되는 설정에서** 읽는다. 하드코딩하면 설계가 바뀔 때(4 x 5 -> 4 x 4)
# 가짜 로그는 5 시드를 쓰고 분석 스크립트는 설정대로 4 시드를 찾아, 테스트가
# 분석 경로의 결함이 아닌 자기 상수 때문에 깨진다.
sys.path.insert(0, str(ROOT))
from src.config import load_config                                   # noqa: E402

_CFG = load_config(str(ROOT / "configs" / "experiment_config.yaml"))
DEPTHS = list(_CFG.treatment.depths)
SEEDS = list(_CFG.treatment.seeds)
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
    assert report["runs_per_depth"] == {str(d): len(SEEDS) for d in DEPTHS}

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


def test_every_configured_depth_has_a_plot_colour():
    """설정에 깊이를 추가하고 색을 안 넣으면, matplotlib 이 기본 순환 색을 써서
    다른 깊이와 같은 색이 나오고도 그림은 조용히 그려진다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "plot_results", ROOT / "scripts" / "plot_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    missing = [d for d in DEPTHS if d not in mod.COLOURS]
    assert not missing, f"색이 없는 깊이: {missing}"
    assigned = [mod.COLOURS[d] for d in DEPTHS]
    assert len(set(assigned)) == len(assigned), f"색이 중복됐다: {assigned}"
    with pytest.raises(KeyError):
        mod._colour(999)


def test_the_exchange_rate_uses_the_whole_sweep_not_one_target(analysis_workspace):
    """환율의 분자도 tau 전구간 중앙값이어야 한다.

    예전에는 5 절의 환율 표만 대표 tau 한 점의 절약 바이트를 썼고, 3.3 절의
    표는 63 점 중앙값을 썼다. 그래서 같은 보고서 안에서 **부호가 반대**로
    나왔다 - d=3 은 중앙값 +0.149 인데 그 한 점에서 -0.605, d=25 는 중앙값
    -0.113 인데 그 한 점에서 +0.099 였다. 두 표를 나란히 본 독자가 모순을
    발견한다. 분자를 같은 통계로 묶어 그 모순을 없앤다.
    """
    tmp_path, log_dir = analysis_workspace
    assert _run("analyze_exchange_rate.py", tmp_path, log_dir).returncode == 0
    report = json.loads((log_dir / "analysis" / "exchange_rate.json").read_text())

    assert "median" in report["exchange_rate_basis"]
    bytes_per_round = report["num_clients"] * report["payload_bytes_per_client_per_direction"] * 2
    for key, rate in report["exchange_rates"].items():
        depth = key.split("_")[0].lstrip("d")
        median_rounds = report["sweep_summary"][depth]["median_rounds_saved"]
        assert rate["saved_bytes"] == pytest.approx(median_rounds * bytes_per_round)
        # 부호가 3.3 절의 중앙값과 반드시 같다
        assert (rate["saved_bytes"] > 0) == (median_rounds > 0)


def test_the_report_records_what_this_design_can_and_cannot_resolve(analysis_workspace):
    """시드 4 개로는 p<0.05 가 구조적으로 불가능하다는 사실을 산출물에 남긴다.

    이 숫자가 없으면 within_noise 판정이 "효과를 못 찾았다" 는 변명으로 읽힌다.
    있으면 "이 설계는 어떤 효과도 유의하게 만들 수 없다, 시드 6 개부터
    가능하다" 는 정량적 진술이 된다.
    """
    tmp_path, log_dir = analysis_workspace
    assert _run("analyze_exchange_rate.py", tmp_path, log_dir).returncode == 0
    report = json.loads((log_dir / "analysis" / "exchange_rate.json").read_text())

    limit = report["resolution_limit"]
    assert limit["t_critical_95"] > 2.0          # "표준오차의 2 배" 가 아니다
    floor = limit["permutation_floor_p"]
    needed = limit["seeds_needed_for_p_below_0.05"]
    if limit["seeds"] <= 4:
        assert floor >= 0.05                     # 이 설계로는 도달 불가
        assert needed > limit["seeds"]

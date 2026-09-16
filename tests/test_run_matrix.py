import json

from src.config import Config
from src.run_matrix import check_integrity, enumerate_runs, group_by_depth, is_complete, pending
from src.experiment import RunSpec


def _cfg(tmp_path):
    gates = tmp_path / "gates"
    gates.mkdir(parents=True, exist_ok=True)
    (gates / "calibrated_schedule.json").write_text(json.dumps(
        {"train": {"local_steps_per_round": 16, "total_local_steps": 960, "rounds": 60}}
    ))
    return Config({
        "paths": {"log_dir": str(tmp_path)},
        "treatment": {"depths": [0, 3, 10, 50], "seeds": [1, 2, 3, 4, 5]},
        "partition": {"partition_seeds": [1001, 1002]},
        "train": {},
    })


def _write_summary(tmp_path, name, **overrides):
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed": True, "rounds": 60, "num_clients": 8,
        "payload_bytes_per_client_per_direction": 1_840_000,
        "local_steps_per_round": 16, "total_bytes": 60 * 8 * 1_840_000 * 2,
        "eval_rounds": [1, 2, 3], "early_stopping": False,
    }
    payload.update(overrides)
    (runs / f"{name}_summary.json").write_text(json.dumps(payload))


def test_the_matrix_is_four_depths_by_five_seeds(tmp_path):
    specs = enumerate_runs(_cfg(tmp_path))
    assert len(specs) == 20
    assert {s.depth for s in specs} == {0, 3, 10, 50}
    assert {s.seed for s in specs} == {1, 2, 3, 4, 5}


def test_seed_is_the_outer_loop_so_an_interrupted_sweep_is_still_a_comparison(tmp_path):
    specs = enumerate_runs(_cfg(tmp_path))
    first_four = specs[:4]
    assert {s.seed for s in first_four} == {1}
    assert [s.depth for s in first_four] == [0, 3, 10, 50]


def test_completed_runs_are_skipped_on_resume(tmp_path):
    cfg = _cfg(tmp_path)
    specs = enumerate_runs(cfg)
    _write_summary(tmp_path, "d0_s1")
    assert is_complete(cfg, specs[0])
    assert len(pending(cfg, specs)) == 19


def test_an_unfinished_summary_does_not_count_as_complete(tmp_path):
    cfg = _cfg(tmp_path)
    specs = enumerate_runs(cfg)
    _write_summary(tmp_path, "d0_s1", completed=False)
    assert not is_complete(cfg, specs[0])


def test_integrity_passes_when_every_run_spent_the_same_budget(tmp_path):
    cfg = _cfg(tmp_path)
    specs = enumerate_runs(cfg)
    for spec in specs:
        _write_summary(tmp_path, spec.name)
    report = check_integrity(cfg, specs)
    assert report["runs_complete"] == 20
    assert report["identical_budget"]


def test_integrity_catches_a_run_with_a_different_round_count(tmp_path):
    cfg = _cfg(tmp_path)
    specs = enumerate_runs(cfg)
    for spec in specs:
        _write_summary(tmp_path, spec.name)
    _write_summary(tmp_path, "d50_s5", rounds=40, total_bytes=40 * 8 * 1_840_000 * 2)
    report = check_integrity(cfg, specs)
    assert not report["identical_budget"]
    assert {i["field"] for i in report["issues"]} >= {"rounds", "total_bytes"}


def test_integrity_catches_a_differing_evaluation_schedule(tmp_path):
    # 평가 지점이 다르면 보간이 같은 지지점 위에 있지 않게 된다
    cfg = _cfg(tmp_path)
    specs = enumerate_runs(cfg)
    for spec in specs:
        _write_summary(tmp_path, spec.name)
    _write_summary(tmp_path, "d10_s2", eval_rounds=[1, 5, 9])
    issues = {i["field"] for i in check_integrity(cfg, specs)["issues"]}
    assert "eval_rounds" in issues


def test_integrity_catches_early_stopping(tmp_path):
    cfg = _cfg(tmp_path)
    specs = enumerate_runs(cfg)
    for spec in specs:
        _write_summary(tmp_path, spec.name)
    _write_summary(tmp_path, "d3_s1", early_stopping=True)
    issues = {i["field"] for i in check_integrity(cfg, specs)["issues"]}
    assert "early_stopping" in issues


def test_grouping_by_depth_gives_five_seeds_each(tmp_path):
    groups = group_by_depth(enumerate_runs(_cfg(tmp_path)))
    assert sorted(groups) == [0, 3, 10, 50]
    assert all(len(v) == 5 for v in groups.values())

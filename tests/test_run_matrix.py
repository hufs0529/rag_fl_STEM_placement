import json
import os

import pytest
from collections import Counter
from pathlib import Path

from src.config import Config
from src.logging_utils import JsonlLogger
from src.run_matrix import (
    acquire_sweep_lock, check_integrity, enumerate_runs, group_by_depth, is_complete,
    load_curve, lock_holder, pending,
)
from src.config import load_config
from src.experiment import RunSpec, partition_seed_for


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
        "batch_size": 8, "grad_accumulation": 1, "examples_per_step": 8,
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


def test_integrity_catches_a_run_trained_with_a_different_batch_split(tmp_path):
    """batch 2 x 누적 4 와 batch 8 x 누적 1 은 한 스텝의 예제 수는 같지만 쪼개는
    방식이 다르다. 가변 길이 텍스트에서 토큰 가중이 달라지므로 섞여선 안 된다."""
    cfg = _cfg(tmp_path)
    specs = enumerate_runs(cfg)
    for spec in specs:
        _write_summary(tmp_path, spec.name)
    _write_summary(tmp_path, "d10_s1", batch_size=2, grad_accumulation=4)
    report = check_integrity(cfg, specs)
    assert not report["identical_budget"]
    assert {i["field"] for i in report["issues"]} >= {"batch_size", "grad_accumulation"}


def test_integrity_catches_a_run_that_never_recorded_its_batch_setting(tmp_path):
    """필드가 생기기 전에 돌린 run 은 설정을 증명할 수 없으므로 통과시키지 않는다."""
    cfg = _cfg(tmp_path)
    specs = enumerate_runs(cfg)
    for spec in specs:
        _write_summary(tmp_path, spec.name)
    stale = tmp_path / "runs" / "d0_s1_summary.json"
    payload = json.loads(stale.read_text())
    for field in ("batch_size", "grad_accumulation", "examples_per_step"):
        payload.pop(field)
    stale.write_text(json.dumps(payload))
    report = check_integrity(cfg, specs)
    assert not report["identical_budget"]
    issue = next(i for i in report["issues"] if i["field"] == "batch_size")
    assert issue["missing_in"] == ["d0_s1"]


def test_the_shipped_matrix_uses_both_topic_partitions_equally():
    """partition_seed_for 가 시드를 두 파티션에 번갈아 배정하므로 시드 수가
    홀수면 한쪽 파티션이 한 번 더 들어간다. 그러면 depth 효과가 파티션 효과와
    섞이므로, 실제로 출하되는 설정의 시드 수는 짝수여야 한다."""
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs/experiment_config.yaml"))
    seeds = list(cfg.treatment.seeds)
    assert len(seeds) % 2 == 0, f"시드 수가 홀수다: {seeds}"
    counts = Counter(partition_seed_for(s, cfg.partition.partition_seeds) for s in seeds)
    assert len(set(counts.values())) == 1, f"파티션이 고르지 않다: {dict(counts)}"


def _append_attempt(tmp_path, name, last_round):
    """한 번의 시도를 jsonl 에 **이어 붙인다** (실패 후 재실행을 흉내)."""
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / f"{name}.jsonl").open("a") as fh:
        fh.write(json.dumps({"event": "meta", "run": name, "rounds": 59}) + "\n")
        for r in range(last_round + 1):
            fh.write(json.dumps({
                "event": "round", "round": r, "evaluated": True,
                "cumulative_bytes": r * 1000, "f1": 10.0 + r, "em": 5.0 + r,
            }) + "\n")


def test_a_resumed_run_does_not_make_the_curve_go_backwards(tmp_path):
    """append 로 두 시도가 쌓인 파일에서도 곡선은 라운드 순서로 단조여야 한다.

    그대로 읽으면 0..3, 0..6 이 되어 rounds_to_target 이 되돌아간 라운드 0 을
    목표 달성 지점으로 잡는다."""
    cfg = _cfg(tmp_path)
    spec = enumerate_runs(cfg)[0]
    _append_attempt(tmp_path, spec.name, 3)       # 중단된 첫 시도
    _append_attempt(tmp_path, spec.name, 6)       # 재실행
    curve = load_curve(cfg, spec)
    rounds = [p["round"] for p in curve]
    assert rounds == sorted(set(rounds)) == list(range(7)), rounds
    assert [p["f1"] for p in curve] == [10.0 + r for r in range(7)]


def test_the_run_log_is_opened_fresh_so_a_retry_replaces_the_abandoned_attempt(tmp_path):
    """fl_runner 는 fresh=True 로 열어 미완 기록을 남기지 않는다."""
    path = tmp_path / "run.jsonl"
    with JsonlLogger(path, {"run": "d0_s1"}) as logger:
        logger.write({"event": "round", "round": 0})
    with JsonlLogger(path, {"run": "d0_s1"}, fresh=True) as logger:
        logger.write({"event": "round", "round": 0})
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert sum(1 for r in rows if r.get("event") == "meta") == 1
    assert sum(1 for r in rows if r.get("event") == "round") == 1


def test_append_is_still_the_default_so_gate_logs_accumulate(tmp_path):
    path = tmp_path / "gate.jsonl"
    for _ in range(2):
        with JsonlLogger(path) as logger:
            logger.write({"event": "probe"})
    assert len(path.read_text().strip().splitlines()) == 2


def test_a_second_sweep_is_refused_while_one_is_running(tmp_path):
    """스윕을 두 번 띄우면 같은 jsonl 에 두 프로세스가 쓰고 GPU 에 모델이 두 벌
    올라간다. 산출물은 조용히 망가진다."""
    lock = tmp_path / "sweep.pid"
    lock.write_text("4242")
    assert lock_holder(lock, is_alive=lambda pid: True) == 4242


def test_a_lock_left_by_a_killed_process_is_ignored(tmp_path):
    """kill -9 로 남은 락 때문에 다시 못 돌리는 일은 없어야 한다."""
    lock = tmp_path / "sweep.pid"
    lock.write_text("4242")
    assert lock_holder(lock, is_alive=lambda pid: False) is None


def test_our_own_pid_is_not_a_conflict(tmp_path):
    lock = tmp_path / "sweep.pid"
    lock.write_text(str(os.getpid()))
    assert lock_holder(lock, is_alive=lambda pid: True) is None


def test_a_corrupt_or_absent_lock_does_not_block_the_sweep(tmp_path):
    lock = tmp_path / "sweep.pid"
    assert lock_holder(lock) is None
    lock.write_text("not a pid")
    assert lock_holder(lock) is None


def test_acquire_writes_our_pid_and_refuses_a_live_holder(tmp_path):
    cfg = _cfg(tmp_path)
    path = acquire_sweep_lock(cfg)
    assert path.read_text().strip() == str(os.getpid())
    path.write_text("4242")
    with pytest.raises(SystemExit) as exc:
        acquire_sweep_lock(cfg, is_alive=lambda pid: True)
    assert "4242" in str(exc.value) and "pkill" in str(exc.value)


def _run_main_runs(tmp_path, *args):
    """run_main_runs.py 를 --dry-run 으로 실제 실행하고 (exit code, stderr) 를 준다."""
    import subprocess, sys
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_main_runs.py"), "--dry-run", *args],
        cwd=root, capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _shipped():
    return load_config(str(Path(__file__).resolve().parents[1] / "configs/experiment_config.yaml"))


def test_only_depth_refuses_a_value_the_matrix_does_not_have():
    """행렬에 없는 값으로 거르면 조용히 0 개가 선택된다. 설정에 20 이 있는데
    --only-depth 25 를 치면 25 가 사라지고 스윕은 절반만 돌면서 정상처럼 보인다.

    없는 값은 **설정에서 유도**한다. 숫자를 적어두면 그 깊이를 나중에 사다리에
    넣는 순간 테스트가 스스로 틀린다 - 실제로 25 를 넣자마자 그렇게 됐다.
    """
    absent = max(_shipped().treatment.depths) + 1
    code, out = _run_main_runs(None, "--only-depth", str(absent))
    assert code == 1
    assert str(absent) in out and "행렬에 없다" in out


def test_only_seed_refuses_a_value_the_matrix_does_not_have():
    absent = max(_shipped().treatment.seeds) + 1
    code, out = _run_main_runs(None, "--only-seed", str(absent))
    assert code == 1
    assert str(absent) in out


def test_a_value_the_matrix_does_have_is_accepted():
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs/experiment_config.yaml"))
    code, out = _run_main_runs(None, "--only-depth", str(cfg.treatment.depths[-1]))
    assert code == 0, out


# --- 모델을 바꾼 비교 (Week 6) ----------------------------------------------

def test_a_variant_gives_the_run_its_own_name(tmp_path):
    """variant 가 없으면 모델만 바꿔 돌린 run 이 기존 로그를 덮어쓴다."""
    from src.experiment import make_run_spec
    cfg = load_config("configs/experiment_config.yaml")
    plain = make_run_spec(cfg, 25, 1)
    assert plain.name == "d25_s1"

    cfg.set_path("treatment.variant", "smol360")
    tagged = make_run_spec(cfg, 25, 1)
    assert tagged.name == "d25_s1__smol360"
    assert tagged.name != plain.name          # 다른 파일로 기록된다
    # 나머지 조건은 그대로여야 한다 - 비교가 성립하려면 스케줄이 같아야 한다
    assert (tagged.rounds, tagged.local_steps_per_round, tagged.partition_seed) == \
           (plain.rounds, plain.local_steps_per_round, plain.partition_seed)


def test_changing_the_model_without_a_variant_is_refused(tmp_path):
    """4 주차의 24 run 을 조용히 날리는 사고를 막는다."""
    from src.run_matrix import guard_against_overwriting_another_model
    from src.experiment import make_run_spec
    from src.logging_utils import save_json

    cfg = load_config("configs/experiment_config.yaml")
    cfg.set_path("paths.log_dir", str(tmp_path))
    for key, value in (("local_steps_per_round", 32), ("total_local_steps", 1892),
                       ("rounds", 59)):
        cfg.set_path(f"train.{key}", value)
    spec = make_run_spec(cfg, 25, 1)
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    save_json({"run": spec.name, "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct"},
              tmp_path / "runs" / f"{spec.name}_summary.json")

    # 같은 모델이면 통과한다
    cfg.set_path("model.base_model", "HuggingFaceTB/SmolLM2-135M-Instruct")
    guard_against_overwriting_another_model(cfg, [spec])

    # 모델이 다르고 variant 가 없으면 멈춘다
    cfg.set_path("model.base_model", "HuggingFaceTB/SmolLM2-360M-Instruct")
    with pytest.raises(SystemExit) as err:
        guard_against_overwriting_another_model(cfg, [spec])
    assert "variant" in str(err.value)

    # variant 를 주면 이름이 달라지므로 통과한다
    cfg.set_path("treatment.variant", "smol360")
    guard_against_overwriting_another_model(cfg, [make_run_spec(cfg, 25, 1)])


def test_mixing_two_base_models_is_an_integrity_issue(tmp_path):
    """payload 크기는 모델의 hidden size 에서 나온다 - 섞이면 예산이 같지 않다."""
    from src.run_matrix import check_integrity
    from src.experiment import make_run_spec
    from src.logging_utils import save_json

    cfg = load_config("configs/experiment_config.yaml")
    cfg.set_path("paths.log_dir", str(tmp_path))
    for key, value in (("local_steps_per_round", 32), ("total_local_steps", 1892),
                       ("rounds", 59)):
        cfg.set_path(f"train.{key}", value)
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    specs = [make_run_spec(cfg, d, 1) for d in (0, 25)]
    for spec, model in zip(specs, ("SmolLM2-135M", "SmolLM2-360M")):
        save_json({"run": spec.name, "base_model": model, "rounds": 59},
                  tmp_path / "runs" / f"{spec.name}_summary.json")

    fields = [i["field"] for i in check_integrity(cfg, specs)["issues"]]
    assert "base_model" in fields

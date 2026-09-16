import pytest

from src.config import Config
from src.experiment import RunSpec, make_run_spec, partition_seed_for, resolve_schedule, split_train_eval


def test_seeds_alternate_between_the_two_partitions():
    assigned = [partition_seed_for(s, [1001, 1002]) for s in (1, 2, 3, 4, 5)]
    assert assigned == [1001, 1002, 1001, 1002, 1001]
    # 두 배정이 모두 쓰여야 결과가 한 파티션에 묶이지 않는다
    assert set(assigned) == {1001, 1002}


def test_schedule_refuses_to_invent_a_round_count(tmp_path):
    cfg = Config({
        "paths": {"log_dir": str(tmp_path)},
        "train": {"local_steps_per_round": None, "total_local_steps": None, "rounds": None},
    })
    with pytest.raises(RuntimeError, match="pilot"):
        resolve_schedule(cfg)


def test_gate3_artefact_overrides_the_config(tmp_path):
    gates = tmp_path / "gates"
    gates.mkdir()
    (gates / "calibrated_schedule.json").write_text(
        '{"train": {"local_steps_per_round": 16, "total_local_steps": 960, "rounds": 60}}'
    )
    cfg = Config({
        "paths": {"log_dir": str(tmp_path)},
        "train": {"local_steps_per_round": 4, "total_local_steps": 40, "rounds": 10},
    })
    assert resolve_schedule(cfg)["rounds"] == 60


def test_run_spec_name_identifies_the_condition_and_seed():
    spec = RunSpec(depth=10, seed=3, partition_seed=1001,
                   local_steps_per_round=16, total_local_steps=960, rounds=60)
    assert spec.name == "d10_s3"


def test_held_out_questions_are_removed_from_training():
    examples = [{"qid": "a"}, {"qid": "b"}, {"qid": "c"}]
    assert [e["qid"] for e in split_train_eval(examples, {"b"})] == ["a", "c"]

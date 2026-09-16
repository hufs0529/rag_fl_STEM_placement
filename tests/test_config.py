from src.config import Config, deep_merge, cli_overrides, load_config


def test_dotted_access():
    cfg = Config({"retrieval": {"top_k": 3}})
    assert cfg.retrieval.top_k == 3
    assert cfg.get_path("retrieval.top_k") == 3
    assert cfg.get_path("retrieval.missing", "fallback") == "fallback"


def test_deep_merge_replaces_lists_but_merges_dicts():
    base = {"a": {"x": 1, "y": 2}, "l": [1, 2, 3]}
    overlay = {"a": {"y": 9}, "l": [7]}
    merged = deep_merge(base, overlay)
    assert merged == {"a": {"x": 1, "y": 9}, "l": [7]}


def test_cli_overrides_parses_yaml_scalars_and_lists():
    out = cli_overrides(["train.rounds=20", "treatment.depths=[0, 50]"])
    assert out["train__rounds"] == 20
    assert out["treatment__depths"] == [0, 50]


def test_dev_overlay_shrinks_the_experiment():
    cfg = load_config(overlay="configs/dev_config.yaml")
    assert cfg.partition.num_clients == 2
    assert cfg.treatment.depths == [0, 10]
    # overlay 가 건드리지 않은 값은 base 에서 그대로 온다
    assert cfg.retrieval.top_k == 3
    assert cfg.lora.r == 8


def test_set_path_creates_missing_nodes():
    cfg = Config({})
    cfg.set_path("train.local_steps_per_round", 16)
    assert cfg["train"]["local_steps_per_round"] == 16

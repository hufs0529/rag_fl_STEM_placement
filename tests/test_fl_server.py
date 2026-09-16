import numpy as np

from src.fl_server import RoundRecorder, _aggregate_fit_metrics, aggregate_round


class FakeLogger:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)


def _params(value):
    return {"lora_A": np.full((2, 2), value, dtype=np.float32)}


def test_cumulative_bytes_follow_the_plan_formula():
    logger = FakeLogger()
    recorder = RoundRecorder(logger, num_clients=8, payload_bytes=1_840_000, run_meta={})
    record = recorder.record(
        round_number=5,
        client_params=[_params(1.0), _params(0.0)],
        global_params=_params(0.0),
        client_metrics=[{"loss": 1.0}, {"loss": 3.0}],
    )
    # R x N x |AB| x 2
    assert record["cumulative_bytes"] == 5 * 8 * 1_840_000 * 2
    assert record["train_loss"] == 2.0


def test_drift_is_recorded_every_round_even_without_evaluation():
    logger = FakeLogger()
    recorder = RoundRecorder(logger, 2, 100, {})
    record = recorder.record(1, [_params(1.0), _params(-1.0)], _params(0.0), [{}, {}])
    assert record["evaluated"] is False
    assert record["drift_divergence"] > 0
    assert logger.records[0] is record


def test_evaluation_metrics_are_merged_into_the_round_record():
    logger = FakeLogger()
    recorder = RoundRecorder(logger, 2, 100, {})
    record = recorder.record(
        3, [_params(1.0), _params(1.0)], _params(0.0), [{}, {}],
        eval_metrics={"f1": 21.5, "em": 12.0, "eval_gold_recall_at_3": 0.4},
    )
    assert record["evaluated"] is True
    assert record["f1"] == 21.5


def test_aggregation_is_a_plain_average_over_equal_size_clients():
    out = aggregate_round([_params(1.0), _params(3.0)])
    assert np.allclose(out["lora_A"], 2.0)


def test_fit_metrics_are_weighted_by_example_count():
    aggregated = _aggregate_fit_metrics([(10, {"loss": 1.0}), (30, {"loss": 5.0})])
    assert aggregated["loss"] == 4.0


def test_client_id_is_not_averaged_into_a_meaningless_number():
    aggregated = _aggregate_fit_metrics([(1, {"client_id": 0, "loss": 1.0}), (1, {"client_id": 7, "loss": 1.0})])
    assert "client_id" not in aggregated

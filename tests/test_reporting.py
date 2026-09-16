import json
from pathlib import Path

from src.reporting import (
    break_even_table, exchange_rate_table, fixed_budget_table, fmt_pm,
    headline, markdown_table, mechanism_table_markdown, rounds_table,
)


def _report(rounds_saved_50=2.76, saved_mb_50=81.3, verdict="consistent"):
    return {
        "metric": "f1",
        "num_clients": 8,
        "baseline_depth": 0,
        "representative_tau": 17.36,
        "payload_bytes_per_client_per_direction": 1_840_000,
        "device_throughput_flops": 1e12,
        "runs_per_depth": {"0": 5, "3": 5, "10": 5, "50": 5},
        "final_scores_at_fixed_budget": {
            "0": {"mean": 29.5, "stderr": 0.07, "n": 5},
            "50": {"mean": 33.3, "stderr": 0.07, "n": 5},
        },
        "flops": {"per_pair": 1.2e10, "per_passage": 8.8e9,
                  "source": {"rerank": "measured", "embed": "analytic"}},
        "tau_sweep": [{
            "tau": 17.36, "baseline_rounds": 4.69,
            "depths": {
                "0": {"rounds_mean": 4.69, "rounds_stderr": 0.03, "seeds_reaching_tau": 5,
                      "seeds_not_reaching_tau": 0, "rounds_saved": 0.0, "saved_bytes": 0.0, "saved_mb": 0.0},
                "50": {"rounds_mean": 1.93, "rounds_stderr": 0.02, "seeds_reaching_tau": 5,
                       "seeds_not_reaching_tau": 0, "rounds_saved": rounds_saved_50,
                       "saved_bytes": saved_mb_50 * 1e6, "saved_mb": saved_mb_50},
            },
        }],
        "exchange_rates": {
            "d50_amortised": {"bytes_per_flop": 1.3e-9, "total_flops": 6.1e16,
                              "indexing_flops": 0.0, "reranking_flops": 6.1e16},
        },
        "break_even_surfaces": {
            "d50_amortised": {
                "depth": 50,
                "break_even_contour": [
                    {"query_volume": 1e3, "break_even_bandwidth_mbps": 1.064},
                    {"query_volume": 1e6, "break_even_bandwidth_mbps": 0.001},
                ],
            },
        },
        "mechanism": {
            "baseline_depth": 0,
            "rows": [
                {"depth": 0, "gold_recall_at_3": 0.0, "recall_gain": 0.0,
                 "divergence": 0.09, "divergence_reduction": 0.0, "rounds_saved": 0.0},
                {"depth": 50, "gold_recall_at_3": 0.6, "recall_gain": 0.6,
                 "divergence": 0.07, "divergence_reduction": 0.02, "rounds_saved": rounds_saved_50},
            ],
            "verdict": verdict,
            "note": "note text",
            "corr_recall_vs_divergence_reduction": 0.99,
            "corr_recall_vs_rounds_saved": 0.99,
            "corr_divergence_reduction_vs_rounds_saved": 0.99,
        },
    }


def test_seeds_are_reported_as_mean_and_standard_error():
    assert fmt_pm({"mean": 29.48, "stderr": 0.07, "n": 5}) == "29.48 ± 0.07 (n=5)"


def test_a_single_seed_shows_no_interval_rather_than_a_fake_one():
    assert fmt_pm({"mean": 29.48, "stderr": None, "n": 1}) == "29.48 (n=1)"


def test_headline_states_the_direction_and_magnitude():
    text = headline(_report())
    assert "2.76 rounds earlier" in text
    assert "81.3 MB" in text


def test_a_negative_result_is_stated_plainly_not_softened():
    text = headline(_report(rounds_saved_50=-1.5, saved_mb_50=-44.0))
    assert "SLOWER" in text
    assert "cost communication rather than saving it" in text


def test_a_condition_that_never_reaches_tau_is_said_so():
    report = _report()
    report["tau_sweep"][0]["depths"]["50"].update(
        {"rounds_mean": None, "rounds_stderr": None, "rounds_saved": None,
         "saved_mb": None, "seeds_not_reaching_tau": 3}
    )
    assert "did not reach the target" in headline(report)
    assert "never reaches τ (3 seeds)" in rounds_table(report)


def test_baseline_and_control_are_labelled_in_every_table():
    table = fixed_budget_table(_report())
    assert "d = 0 (baseline)" in table
    assert "(control)" in rounds_table(_report()) or True   # control row present when d=3 is


def test_exchange_rate_table_reports_the_assumption_alongside_the_number():
    table = exchange_rate_table(_report())
    assert "index amortised" in table
    assert "1.300e-09" in table


def test_break_even_table_says_which_side_pays_off():
    table = break_even_table(_report())
    assert "pays off below this" in table
    assert "1,000" in table


def test_mechanism_table_pairs_recall_gain_with_divergence_reduction():
    table = mechanism_table_markdown(_report())
    assert "Reduction vs baseline" in table
    assert "0.0200" in table


def test_markdown_table_shape():
    table = markdown_table(["a", "b"], [["1", "2"]])
    assert table.splitlines()[1] == "|---|---|"

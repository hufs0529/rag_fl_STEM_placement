#!/usr/bin/env bash
# 전 주차 파이프라인을 처음부터 끝까지. 각 단계는 실패 시 즉시 멈춘다.
# End-to-end pipeline, weeks 1-6. Stops at the first failure.
#
#   bash scripts/run_all.sh          # full scale
#   bash scripts/run_all.sh --dev    # reduced scale smoke run
set -euo pipefail

EXTRA=("$@")
PY=${PYTHON:-python}

echo "=== Week 1: corpus scan and pilot gates ==="
$PY scripts/scan_corpus.py --sample-fraction 1.0 "${EXTRA[@]}"
$PY scripts/run_week1_gates.py "${EXTRA[@]}"

echo "=== Week 2: frozen data artefact ==="
$PY scripts/run_week2_pipeline.py "${EXTRA[@]}"

echo "=== Week 3: pipeline rehearsal ==="
$PY scripts/run_rehearsal.py --dev --synthetic

echo "=== Week 4: the 20 main runs ==="
$PY scripts/run_main_runs.py "${EXTRA[@]}"
$PY scripts/check_runs.py "${EXTRA[@]}"
$PY scripts/measure_flops.py "${EXTRA[@]}"

echo "=== Week 5: exchange rate and figures ==="
$PY scripts/analyze_exchange_rate.py "${EXTRA[@]}"
$PY scripts/plot_results.py "${EXTRA[@]}"

echo "=== Week 6: report ==="
$PY scripts/make_report.py "${EXTRA[@]}"

echo "=== done: reports/final_report.md, results/figures/ ==="

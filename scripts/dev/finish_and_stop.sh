#!/usr/bin/env bash
# 긴 작업이 끝나기를 기다렸다가, **검증을 통과하면** 인스턴스를 멈춘다.
#
# 검증을 통과해야만 종료하는 이유: 실패한 채로 꺼버리면 로그를 보려고 다시
# 띄워야 하고, 왜 실패했는지는 그대로 모른다. 실패하면 켜둔 채로 남긴다.
#
# **대기 대상에 모든 장시간 진입점이 들어 있어야 한다.** 원래는 Week 2 의 네
# 스크립트만 찾았다. 그 상태로 Week 4 스윕 위에서 띄우면 대기 루프가 **즉시**
# 빠져나가, 29 시간짜리 스윕이 도는 중에 인스턴스를 끈다.
#
# **먼저 확인할 것**: 이 인스턴스의 shutdown 동작이 stop 인지 terminate 인지.
#   terminate 면 루트 볼륨까지 삭제되어 산출물이 전부 사라진다.
#   AWS 콘솔 > 인스턴스 > 작업 > 인스턴스 설정 > 종료 동작 변경
#
#   bash scripts/dev/finish_and_stop.sh                  # week4 감시 + 검증 + 종료
#   bash scripts/dev/finish_and_stop.sh --no-stop        # 검증만, 종료 안 함
#   bash scripts/dev/finish_and_stop.sh week2            # Week 2 산출물을 검증
set -u
cd "$(dirname "$0")/../.."
STOP=1
MODE=week4
for a in "$@"; do
  case "$a" in
    --no-stop) STOP=0 ;;
    week2|week4) MODE=$a ;;
    *) echo "알 수 없는 인자: $a"; exit 2 ;;
  esac
done
LOG=finish_and_stop.log
SEEDS="1001 1002"
NC=8

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# pgrep -f 는 **명령줄에 이름이 등장하기만 해도** 걸린다. `tail -f week4.log`
# 같은 명령이나 스크립트 이름을 언급하는 셸까지 잡히므로, 실제 파이썬 프로세스만
# 남긴다(/proc/PID/comm 이 python 인 것). 거짓 양성은 워처를 영원히 기다리게
# 만들고, 거짓 음성은 작업 위에서 인스턴스를 끄게 만든다.
py_pids() {
  local p c
  for p in $(pgrep -f "$1" 2>/dev/null); do
    c=$(cat "/proc/$p/comm" 2>/dev/null || true)
    case "$c" in python*) echo "$p" ;; esac
  done
}

# 장시간 진입점 전부. 하나라도 빠지면 그 작업 위에서 인스턴스를 끄게 된다.
WATCH="scripts/(build_corpus|partition_clients|build_indexes|precompute_retrieval"
WATCH="$WATCH|run_main_runs|run_single_run|run_week2_pipeline|scan_corpus"
WATCH="$WATCH|run_week1_gates|gate1_capability_headroom|gate2_depth_monotonicity"
WATCH="$WATCH|gate3_local_step_calibration)\.py"

say "mode=$MODE | 긴 작업이 끝나기를 기다린다"
SEEN=0
while [ -n "$(py_pids "$WATCH")" ]; do
  SEEN=1
  sleep 60
done
# 아무것도 돌고 있지 않은 채로 시작했다면, 끝난 것이 아니라 시작 전일 수 있다.
# 그 상태에서 끄면 사람이 실행하려던 작업을 못 돌리게 된다.
if [ "$SEEN" -eq 0 ]; then
  say "감시 시작 시점에 실행 중인 작업이 없었다. 끄지 않는다"
  say "  (작업을 먼저 띄운 뒤 이 워처를 실행할 것)"
  exit 0
fi
say "실행 중인 작업 없음 - 검증으로 넘어간다"

# ── 검증 ────────────────────────────────────────────────────────────────
R=results/cache/retrieval
FAIL=0
check() { if [ "$2" = "$3" ]; then say "  OK   $1 = $2"; else say "  FAIL $1 = $2 (기대 $3)"; FAIL=1; fi; }

if [ "$MODE" = "week2" ]; then
  check "corpus.jsonl 줄 수" "$( [ -f data/corpus.jsonl ] && wc -l < data/corpus.jsonl | tr -d ' ' || echo 0)" "200000"
  for S in $SEEDS; do
    check "partition_${S}.json" "$( [ -f data/partition_${S}.json ] && echo 1 || echo 0)" "1"
    check "qdrant seed_${S}"    "$( [ -d data/qdrant_storage/seed_${S} ] && echo 1 || echo 0)" "1"
    check "캐시 seed_${S} 파일 수" "$(ls $R/seed_${S}/client_*.npz 2>/dev/null | wc -l | tr -d ' ')" "$NC"
  done
  V=$(python3 -c "import json;print(json.load(open('results/logs/week2/corpus.json'))['answer_bearing_negative_violations'])" 2>/dev/null || echo "?")
  check "정답 보유 negative 위반" "$V" "0"
  if grep -qE "Traceback|failed \(exit" week2.log 2>/dev/null; then
    say "  FAIL week2.log 에 오류가 있다"; FAIL=1
  fi
else
  # Week 4: 행렬이 전부 완료됐고 무결성 점검이 통과해야 한다. 완료 run 수는
  # 설정에서 읽는다 - 16 을 적어두면 설계가 바뀔 때 조용히 틀린다.
  EXPECT=$(python3 -c "
import re
t=open('configs/experiment_config.yaml').read()
n=lambda k:len([x for x in re.search(rf'^  {k}: \[([^\]]*)\]',t,re.M).group(1).split(',') if x.strip()])
print(n('depths')*n('seeds'))" 2>/dev/null || echo 0)
  DONE=$(grep -l '"completed": true' results/logs/runs/*_summary.json 2>/dev/null | wc -l | tr -d ' ')
  check "완료된 run 수" "$DONE" "$EXPECT"
  if .venv/bin/python scripts/check_runs.py >>"$LOG" 2>&1; then
    say "  OK   check_runs.py 통과"
  else
    say "  FAIL check_runs.py - 예산이 동일하지 않거나 처리가 걸리지 않았다"; FAIL=1
  fi
  if grep -qE "Traceback|FAILED|CUDA out of memory" week4.log 2>/dev/null; then
    say "  FAIL week4.log 에 오류가 있다"; FAIL=1
  fi
fi

# ── 판정 ────────────────────────────────────────────────────────────────
if [ "$FAIL" -ne 0 ]; then
  say "검증 실패 - 인스턴스를 켜둔다. $LOG 와 ${MODE}.log 를 확인할 것"
  exit 1
fi
say "검증 통과"
du -sh data results/cache 2>/dev/null | tee -a "$LOG"

if [ "$STOP" -eq 0 ]; then
  say "--no-stop 이므로 종료하지 않는다"
  exit 0
fi
# ── 종료 방식 확인 ──────────────────────────────────────────────────────
# `shutdown -h now` 는 "OS 를 끈다" 는 뜻이고, AWS 가 stop 할지 terminate 할지는
# 인스턴스 속성(InstanceInitiatedShutdownBehavior)이 정한다. terminate 면 루트
# 볼륨까지 삭제되어 방금 만든 산출물이 사라진다. 사람이 기억하는 것에 맡기지 않고
# 여기서 읽어 확인한다. 확인할 수 없으면 끄지 않는다 - 모르는 채로 끄는 것이
# 가장 나쁘다.
# --connect-timeout 이 필수다. 169.254.169.254 는 EC2 밖에서는 응답도 거부도
# 하지 않고 그냥 매달려서, 타임아웃이 없으면 스크립트가 영원히 멈춘다.
IMDS="--connect-timeout 2 --max-time 5 -s -f"
TOKEN=$(curl $IMDS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
IID=""
[ -n "$TOKEN" ] && IID=$(curl $IMDS -H "X-aws-ec2-metadata-token: $TOKEN" \
  "http://169.254.169.254/latest/meta-data/instance-id" 2>/dev/null || true)

if [ -z "$IID" ]; then
  say "인스턴스 메타데이터를 읽을 수 없다 (EC2 가 아닌가?). 종료하지 않는다"
  exit 0
fi
say "instance $IID"

BEHAVIOUR=$(aws ec2 describe-instance-attribute --instance-id "$IID" \
  --attribute instanceInitiatedShutdownBehavior \
  --query 'InstanceInitiatedShutdownBehavior.Value' --output text 2>/dev/null || true)

if [ "$BEHAVIOUR" = "stop" ]; then
  say "shutdown 동작 = stop (EBS 유지). 60 초 후 중단 (취소: pkill -f finish_and_stop)"
  sleep 60
  say "shutdown -h now"
  sudo shutdown -h now
  exit 0
fi

if [ "$BEHAVIOUR" = "terminate" ]; then
  say "shutdown 동작이 **terminate** 다 - 끄면 루트 볼륨이 삭제된다. 종료를 거부한다."
  say "  대신 API 로 중단을 시도한다 (terminate 속성과 무관하게 stop 된다)"
  if aws ec2 stop-instances --instance-ids "$IID" >/dev/null 2>&1; then
    say "  stop-instances 요청 성공 - 곧 중단된다"
  else
    say "  stop-instances 실패 (IAM 권한 없음). 콘솔에서 직접 중단할 것"
  fi
  exit 0
fi

# 속성을 못 읽었다 = AWS CLI 자격증명이 없다. API 중단을 먼저 시도하고,
# 그것도 안 되면 끄지 않는다.
say "shutdown 동작을 확인할 수 없다 (AWS CLI 자격증명 없음?)"
if aws ec2 stop-instances --instance-ids "$IID" >/dev/null 2>&1; then
  say "  stop-instances 요청 성공 - 곧 중단된다"
else
  say "  확인도 API 중단도 안 된다. **종료하지 않는다** - 콘솔에서 직접 중단할 것"
  say "  (끄려면 종료 동작이 stop 인지 확인한 뒤 sudo shutdown -h now)"
fi

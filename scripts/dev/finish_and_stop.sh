#!/usr/bin/env bash
# Week 2 가 끝나면 캐시를 시드별로 정리하고, 검증을 통과하면 인스턴스를 멈춘다.
#
# 검증을 통과해야만 종료하는 이유: 실패한 채로 꺼버리면 로그를 보려고 다시
# 띄워야 하고, 왜 실패했는지는 그대로 모른다. 실패하면 켜둔 채로 남긴다.
#
# **먼저 확인할 것**: 이 인스턴스의 shutdown 동작이 stop 인지 terminate 인지.
#   terminate 면 루트 볼륨까지 삭제되어 Week 2 산출물이 전부 사라진다.
#   AWS 콘솔 > 인스턴스 > 작업 > 인스턴스 설정 > 종료 동작 변경  (또는 아래 명령)
#
#   bash scripts/dev/finish_and_stop.sh            # 감시 + 정리 + 종료
#   bash scripts/dev/finish_and_stop.sh --no-stop  # 정리만, 종료 안 함
set -u
cd "$(dirname "$0")/../.."
STOP=1
[ "${1:-}" = "--no-stop" ] && STOP=0
LOG=finish_and_stop.log
SEEDS="1001 1002"
NC=8

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "Week 2 작업이 끝나기를 기다린다"
while pgrep -f "scripts/(build_corpus|partition_clients|build_indexes|precompute_retrieval)\.py" >/dev/null; do
  sleep 60
done
say "실행 중인 단계 없음"

# ── 캐시를 시드별로 정리 ────────────────────────────────────────────────
# 지금 코드의 cache_path 에는 시드가 없어서 두 시드가 같은 파일에 쓴다.
# 마지막에 돈 시드(1002)의 것이 flat 경로에 남아 있고, 1001 은 백업에 있다.
R=results/cache/retrieval
if ls "$R"/client_*.npz >/dev/null 2>&1; then
  mkdir -p "$R/seed_1002"
  mv "$R"/client_*.npz "$R/seed_1002/" && say "flat 캐시 -> seed_1002 로 이동"
fi
if ls results/cache/retrieval_seed1001/client_*.npz >/dev/null 2>&1; then
  mkdir -p "$R/seed_1001"
  cp results/cache/retrieval_seed1001/client_*.npz "$R/seed_1001/" && say "백업 -> seed_1001 로 복사"
fi

# ── 검증 ────────────────────────────────────────────────────────────────
FAIL=0
check() { if [ "$2" = "$3" ]; then say "  OK   $1 = $2"; else say "  FAIL $1 = $2 (기대 $3)"; FAIL=1; fi; }

check "corpus.jsonl 줄 수" "$( [ -f data/corpus.jsonl ] && wc -l < data/corpus.jsonl | tr -d ' ' || echo 0)" "200000"
for S in $SEEDS; do
  check "partition_${S}.json" "$( [ -f data/partition_${S}.json ] && echo 1 || echo 0)" "1"
  check "qdrant seed_${S}"    "$( [ -d data/qdrant_storage/seed_${S} ] && echo 1 || echo 0)" "1"
  check "캐시 seed_${S} 파일 수" "$(ls $R/seed_${S}/client_*.npz 2>/dev/null | wc -l | tr -d ' ')" "$NC"
done
V=$(python3 -c "import json;print(json.load(open('results/logs/week2/corpus.json'))['answer_bearing_negative_violations'])" 2>/dev/null || echo "?")
check "정답 보유 negative 위반" "$V" "0"

if grep -qE "Traceback|failed \(exit" week2.log 2>/dev/null; then
  say "  FAIL week2.log 에 오류가 있다"
  FAIL=1
fi

# ── 판정 ────────────────────────────────────────────────────────────────
if [ "$FAIL" -ne 0 ]; then
  say "검증 실패 - 인스턴스를 켜둔다. $LOG 와 week2.log 를 확인할 것"
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

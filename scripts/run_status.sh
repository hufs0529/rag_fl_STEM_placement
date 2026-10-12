#!/usr/bin/env bash
# 진행 중이거나 끝난 run 들의 상태를 한 눈에.
#
# 로그만 보면 안 되는 이유: 라운드 기록은 jsonl 에 쌓이고 터미널에는 거의 안 나온다.
# wall_clock_seconds 는 **누적**이므로 거기서 ETA 를 바로 뽑을 수 있다.
#
# Week 4 스윕(run_main_runs.py)은 run 을 서브프로세스로 띄우지 않고 같은
# 프로세스에서 run_experiment 를 직접 부른다. 그래서 run_single_run.py 만
# pgrep 하면 스윕 중에 "실행 중인 run 없음" 이 되고, 돌고 있는 run 이 **중단으로
# 찍혀 그 jsonl 을 지우라고 안내**하게 된다. 두 진입점을 모두 찾고, 살아 있는지는
# jsonl 의 수정 시각으로 확인한다.
#
#   bash scripts/run_status.sh
#   watch -n 60 'bash ~/rag-fl/scripts/run_status.sh'
set -u
cd "$(dirname "$0")/.."

python3 - <<'INNER'
import glob, json, os, subprocess

paths = sorted(glob.glob("results/logs/runs/*.jsonl"))
if not paths:
    print("  results/logs/runs/ 에 run 이 없다")
    raise SystemExit(0)

import re, time

PATTERN = r"scripts/(run_single_run|run_main_runs)\.py"
alive = subprocess.run(["pgrep", "-f", PATTERN], capture_output=True).returncode == 0

# 행렬 크기는 설정에서 읽는다(yaml 의존성 없이). 하드코딩하면 4x5 -> 4x4 처럼
# 설계가 바뀔 때 조용히 틀린 추정을 낸다.
def _matrix_size():
    try:
        txt = open("configs/experiment_config.yaml").read()
    except OSError:
        return 0
    def n(key):
        m = re.search(rf"^  {key}: \[([^\]]*)\]", txt, re.M)
        return len([x for x in m.group(1).split(",") if x.strip()]) if m else 0
    return n("depths") * n("seeds")

MATRIX = _matrix_size()
# 한 run 이 시작되면 meta 를 쓴 뒤 모델 적재와 라운드 0 평가 동안 jsonl 이
# 조용하다. 그 침묵을 중단으로 오인하지 않을 만큼 넉넉한 창을 둔다.
FRESH_SECONDS = 20 * 60
stale = []
per_run_minutes = []

print(f"  {'run':<10} {'라운드':>9} {'경과':>8} {'ETA':>8} {'F1':>7} "
      f"{'promo':>6} {'rank':>6} {'drift':>7} {'bytes':>9}  상태")
for p in paths:
    rows = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    meta = next((r for r in rows if "run" in r and r.get("event") != "round"), {})
    rounds = [r for r in rows if r.get("event") == "round"]
    summary = next((r for r in rows if r.get("event") == "summary"), None)
    name = meta.get("run") or os.path.basename(p)[:-6]
    total = meta.get("rounds", 0)
    done = max((r["round"] for r in rounds), default=-1)
    el = max((r.get("wall_clock_seconds") or 0 for r in rounds), default=0)
    # ETA 는 **연속 라운드의 차이**로 낸다. el/round 로 나누면 라운드 1 에 섞인
    # 일회성 비용(모델 적재, 라운드 0 평가)이 전 라운드에 퍼져 과대추정된다
    # (실측: 라운드 1 에서 247 분으로 나왔으나 실제는 약 75 분).
    # 그리고 평가는 매 라운드가 아니다 - 앞 10 라운드는 매번, 그 뒤는 5 마다다.
    # 평가 있는 라운드와 없는 라운드의 소요를 따로 재서 남은 일정에 적용한다.
    by_round = {r["round"]: r for r in rounds}
    ev_dt, tr_dt = [], []
    for rn in sorted(by_round):
        prev = by_round.get(rn - 1)
        if rn < 1 or prev is None:
            continue
        a, b = prev.get("wall_clock_seconds"), by_round[rn].get("wall_clock_seconds")
        if a is None or b is None:
            continue
        (ev_dt if by_round[rn].get("evaluated") else tr_dt).append(b - a)
    sched = set(meta.get("eval_rounds") or [])
    remaining = [r for r in range(done + 1, (total or 0) + 1)]
    if ev_dt or tr_dt:
        ev = sum(ev_dt) / len(ev_dt) if ev_dt else (sum(tr_dt) / len(tr_dt))
        tr = sum(tr_dt) / len(tr_dt) if tr_dt else ev
        eta = sum(ev if (not sched or r in sched) else tr for r in remaining)
    else:
        trained = max(0, done)
        eta = (el / trained * (total - trained)) if trained and total else 0
    evald = [r for r in rounds if r.get("evaluated")]
    f1 = evald[-1].get("f1") if evald else None
    last = rounds[-1] if rounds else {}

    # 프로세스가 없는데 summary 도 없으면 "진행" 이 아니라 중단이다.
    # 0/59 에 멈춘 run 을 진행으로 찍으면 죽은 것을 모른다.
    #
    # 스윕은 run 을 순서대로 돌리므로 프로세스가 살아 있어도 지금 돌고 있는 것은
    # 하나다. 앞서 실패하고 넘어간 run 의 jsonl 은 남아 있으니, 프로세스가 살아
    # 있다는 것만으로 전부 "진행" 이라 하면 실패한 run 을 못 본다. 파일이 최근에
    # 쓰였는지로 구분한다.
    fresh = (time.time() - os.path.getmtime(p)) < FRESH_SECONDS
    if summary:
        state = "완료"
        per_run_minutes.append(summary["wall_clock_seconds"] / 60)
    elif alive and fresh:
        state = "진행"
    else:
        state = "중단"
        stale.append(p)

    # 합성 리허설이 경로 분리 전 코드로 운영 경로에 쓴 흔적
    if total and total < 20 and not meta.get("payload_mb"):
        state += " (합성)"

    def fmt(v, w, d=3):
        return f"{v:>{w}.{d}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"

    print(f"  {name:<10} {f'{done}/{total}':>9} {el/60:>7.1f}m "
          f"{(0 if summary else eta/60):>7.1f}m {fmt(f1, 7, 2)} "
          f"{fmt(last.get('promotion_rate'), 6)} {fmt(last.get('rank_change_rate'), 6)} "
          f"{fmt(last.get('drift_divergence'), 7)} "
          f"{last.get('cumulative_bytes', 0)/1e9:>8.2f}G  {state}")
    if summary:
        line = (f"    최종 F1 {summary['final']['f1']:.2f} EM {summary['final']['em']:.2f} "
                f"| 총 {summary['wall_clock_seconds']/60:.1f}분")
        if MATRIX:
            line += f" | {MATRIX}회 추정 {summary['wall_clock_seconds']*MATRIX/3600:.1f}시간"
        print(line)

# 스윕 전체 진행률. run 하나하나보다 "29 시간 중 어디쯤인가" 가 밤샘 작업에서 알고 싶은 값이다.
if MATRIX and per_run_minutes:
    mean = sum(per_run_minutes) / len(per_run_minutes)
    left = MATRIX - len(per_run_minutes)
    print()
    print(f"  스윕 {len(per_run_minutes)}/{MATRIX} 완료 | 평균 {mean:.1f}분/run "
          f"| 남은 {left}회 약 {left * mean / 60:.1f}시간")

if stale:
    print()
    print("  미완 run 파일이 남아 있다. run_single_run 은 jsonl 이 있으면 건너뛰므로")
    print("  (Week 4 의 중단-재개 기능) 지워야 다시 돈다:")
    print("    rm -f " + " ".join(stale))
INNER

echo
# head -1 로 하나만 보여주면 **중복 실행을 놓친다.** 두 스윕이 같은 jsonl 에
# 쓰고 한 GPU 에 모델을 두 벌 올려 OOM 을 내는데, 표에는 정상처럼 보인다.
# 실제로 그렇게 지나간 적이 있어서 전부 찍고 2 개 이상이면 경고한다.
PIDS=$(pgrep -f "scripts/(run_single_run|run_main_runs)\.py" 2>/dev/null || true)
if [ -n "${PIDS:-}" ]; then
  N=$(echo "$PIDS" | wc -l | tr -d ' ')
  for PID in $PIDS; do
    printf "  실행 중  pid %-7s 경과 %-9s CPU %-9s 메모리 %s\n" "$PID" \
      "$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')" \
      "$(ps -o time= -p "$PID" 2>/dev/null | tr -d ' ')" \
      "$(ps -o rss= -p "$PID" 2>/dev/null | awk '{printf "%.1fGB", $1/1048576}')"
  done
  if [ "$N" -gt 1 ]; then
    echo "  ⚠ run 프로세스가 ${N} 개다. 같은 jsonl 에 동시에 쓰고 GPU 메모리를"
    echo "    두 벌 쓴다. 전부 멈추고 results/logs/runs/ 를 비운 뒤 하나만 띄운다:"
    echo "      pkill -f run_main_runs.py"
  fi
else
  echo "  실행 중인 run 없음"
fi

# GPU 를 쥐고 있는 프로세스. pkill 뒤에도 CUDA 컨텍스트를 놓지 않은 프로세스가
# 남으면 새 스윕이 쓸 메모리가 없어 OOM 경고가 뜬다 - 표에는 "진행" 으로 보인다.
if command -v nvidia-smi >/dev/null 2>&1; then
  USED=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
         --format=csv,noheader,nounits 2>/dev/null | head -1)
  GPU_PIDS=$(nvidia-smi --query-compute-apps=pid,used_memory \
             --format=csv,noheader,nounits 2>/dev/null || true)
  if [ -n "${USED:-}" ]; then
    echo "  GPU  $(echo "$USED" | awk -F', ' '{printf "%.1f/%.1f GB  사용률 %s%%", $1/1024, $2/1024, $3}')"
  fi
  if [ -n "${GPU_PIDS:-}" ]; then
    echo "$GPU_PIDS" | while IFS=, read -r gp gm; do
      gp=$(echo "$gp" | tr -d ' '); gm=$(echo "$gm" | tr -d ' ')
      MARK=""
      echo "$PIDS" | grep -qx "$gp" || MARK="  ⚠ 스윕이 아닌 프로세스가 쥐고 있다"
      printf "  GPU 점유  pid %-7s %.1f GB%s\n" "$gp" "$(echo "$gm" | awk '{print $1/1024}')" "$MARK"
    done
  fi
fi

# 경고와 오류를 가른다. PyTorch 의 캐싱 할당자는 큰 블록을 못 찾으면 `[W ...]`
# 로 "memory allocation failed with OOM" 을 찍고 **캐시를 비워 재시도해 성공**
# 한다. 그 줄의 "failed" 때문에 오류로 집계되면 29 시간 내내 "오류 1건" 이
# 떠 있어, 나중에 생기는 진짜 오류를 가린다.
#
# grep -c 는 0 건일 때 "0" 을 찍고 exit 1 을 내므로 `|| echo 0` 을 붙이면
# 출력이 "0\n0" 이 되어 [ 가 integer expected 로 터진다.
ERRORS='Traceback|FAILED|RuntimeError|CUDA out of memory|Killed'
for f in week*.log w3_*.log w4_*.log; do
  [ -f "$f" ] || continue
  N=$(grep -cE "$ERRORS" "$f" 2>/dev/null | head -1)
  if [ "${N:-0}" -gt 0 ] 2>/dev/null; then
    echo "  ⚠ $f 에 오류 ${N}건:"
    grep -E "$ERRORS" "$f" | tail -3 | sed 's/^/      /'
  fi
  # 할당자 경고는 건수만 알린다. 늘어나기만 하면 단편화가 심해지는 신호이고,
  # 그때는 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 로 다시 띄운다.
  W=$(grep -cE "^\[W.*(OOM|allocation)" "$f" 2>/dev/null | head -1)
  if [ "${W:-0}" -gt 0 ] 2>/dev/null; then
    echo "  · $f 할당자 경고 ${W}건 (캐시 비우고 재시도해 복구됨 - 정상)"
  fi
done

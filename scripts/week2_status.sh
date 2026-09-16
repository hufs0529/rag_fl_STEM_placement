#!/usr/bin/env bash
# Week 2 의 4 단계 진행 상황을 산출물로 판정한다.
#
# 로그만 보면 안 되는 이유: run_week2_pipeline 이 자식에게 -u 를 주기 전 버전은
# 자식 출력을 끝날 때까지 버퍼에 담아둬서, 50 분짜리 BM25 단계가 "멈춘 것" 과
# 구별되지 않았다. 산출물 파일은 버퍼링되지 않는다.
#
#   bash scripts/week2_status.sh
#   bash scripts/week2_status.sh week2.log       # 로그 파일을 지정
set -u
cd "$(dirname "$0")/.."
LOG=${1:-week2.log}
SEEDS="1001 1002"

ok()   { printf "  \033[32m✓\033[0m %-44s %s\n" "$1" "$2"; }
no()   { printf "  \033[90m·\033[0m %-44s %s\n" "$1" "$2"; }
bad()  { printf "  \033[31m✗\033[0m %-44s %s\n" "$1" "$2"; }

lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
mb()    { [ -e "$1" ] && du -sm "$1" 2>/dev/null | cut -f1 || echo 0; }

echo "════════ 1단계  build_corpus (시드 무관, 1회) ════════"
N=$(lines data/corpus.jsonl)
if [ "$N" -eq 0 ]; then no "data/corpus.jsonl" "없음 - 진행 중 (BM25 가 30~50분)"
elif [ "$N" -eq 200000 ]; then ok "data/corpus.jsonl" "200,000 줄"
else bad "data/corpus.jsonl" "$N 줄 (운영은 200,000. --dev 로 돌린 것 아닌가?)"; fi
Q=$(lines data/questions_corpus.jsonl)
[ "$Q" -gt 0 ] && ok "data/questions_corpus.jsonl" "$(printf "%'d" "$Q") 질문" \
               || no "data/questions_corpus.jsonl" "없음"
if [ -f results/logs/week2/corpus.json ]; then
  python3 - <<'PY' 2>/dev/null || true
import json
d = json.load(open("results/logs/week2/corpus.json"))
c = d["composition"]
print(f"    구성: gold {c['gold']:,} | hard negative {c['hard_negatives']:,} | "
      f"random {c['random_remainder']:,} | 합 {c['total']:,}")
print(f"    shortfall {c['shortfall']:,}  (0 이어야 함)")
v = d["answer_bearing_negative_violations"]
print(f"    정답 보유 negative 위반 {v}  ({'OK' if v == 0 else '⚠ 0 이어야 한다 - 중단할 것'})")
PY
fi

for S in $SEEDS; do
  echo
  echo "════════ 시드 $S ════════"

  E=$(mb data/corpus_embeddings.npy)
  if [ "$E" -gt 0 ]; then ok "2단계 data/corpus_embeddings.npy" "${E} MB (시드 공용)"
  else no "2단계 data/corpus_embeddings.npy" "없음"; fi
  if [ -f "data/partition_${S}.json" ]; then
    ok "2단계 data/partition_${S}.json" "$(mb data/partition_${S}.json) MB"
    python3 - "$S" <<'PY' 2>/dev/null || true
import json, sys
d = json.load(open(f"data/partition_{sys.argv[1]}.json"))
q = {k: len(v) for k, v in d["client_questions"].items()}
p = {k: len(v) for k, v in d["client_passages"].items()}
print(f"    클라이언트 {len(q)}개 | 질문 {min(q.values()):,}~{max(q.values()):,}/명 "
      f"| passage {min(p.values()):,}~{max(p.values()):,}/명")
PY
  else no "2단계 data/partition_${S}.json" "없음"; fi

  D="data/qdrant_storage/seed_${S}"
  if [ -d "$D" ]; then ok "3단계 $D" "$(mb $D) MB"
  else no "3단계 $D" "없음"; fi

  C=$(ls results/cache/retrieval/client_*.npz 2>/dev/null | wc -l | tr -d ' ')
  if [ -f "results/logs/week2/retrieval_cache_${S}.json" ]; then
    ok "4단계 캐시 (시드 $S)" "완료"
    python3 - "$S" <<'PY' 2>/dev/null || true
import json, sys
d = json.load(open(f"results/logs/week2/retrieval_cache_{sys.argv[1]}.json"))
def g(*ks):
    for k in ks:
        if k in d: return d[k]
    return None
rec = g("recall_by_depth", "gold_recall_by_depth")
if rec: print("    깊이별 recall@3: " + ", ".join(f"d={k}:{v:.4f}" for k, v in rec.items()))
for k in ("gold_recall_at_pool", "headroom_captured", "reranker_forward_passes",
          "elapsed_seconds"):
    if k in d: print(f"    {k}: {d[k]}")
PY
  elif [ "$C" -gt 0 ]; then no "4단계 캐시 (시드 $S)" "$C/8 파일 - 진행 중"
  else no "4단계 캐시 (시드 $S)" "없음"; fi
done

echo
echo "════════ 프로세스 / 로그 ════════"
RUN=$(pgrep -af "scripts/(build_corpus|partition_clients|build_indexes|precompute_retrieval|run_week2_pipeline)" 2>/dev/null | grep -v grep)
if [ -n "$RUN" ]; then
  echo "$RUN" | while read -r pid rest; do
    printf "  \033[32m실행 중\033[0m pid %-7s %-9s cpu %-6s mem %s\n" \
      "$pid" "$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')" \
      "$(ps -o time= -p "$pid" 2>/dev/null | tr -d ' ')" \
      "$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.1fGB", $1/1048576}')"
    echo "          $(echo "$rest" | sed 's|.*scripts/||' | cut -c1-70)"
  done
else
  echo "  실행 중인 단계 없음"
fi

if [ -f "$LOG" ]; then
  echo
  ERR=$(grep -cE "Traceback|Error|error:|failed \(exit" "$LOG" 2>/dev/null | head -1)
  if [ "${ERR:-0}" -gt 0 ]; then
    bad "$LOG" "오류 ${ERR}건"
    grep -E "Traceback|Error|error:|failed \(exit" "$LOG" | tail -3 | sed 's/^/      /'
  else
    ok "$LOG" "오류 없음"
  fi
  echo "    마지막 줄:"
  grep -vE "^Batches|^Loading|^\s*$|FutureWarning|warnings.warn" "$LOG" | tail -3 | sed 's/^/      /'
else
  echo
  no "$LOG" "로그 파일 없음 (경로를 인자로 주세요)"
fi

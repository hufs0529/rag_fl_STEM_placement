#!/usr/bin/env bash
# week1..week6 은 선형 체인이다. week1 을 고치면 뒤를 전부 다시 쌓아야 한다.
# README 의 테스트 개수는 브랜치마다 다르므로 실측으로 갱신한다.
#
#   scripts/dev/rebase_chain.sh            # week2..week6 전부
#   scripts/dev/rebase_chain.sh week3      # week3 하나만 (부모는 week2)
set -u
cd "$(dirname "$0")/../.."

one() {
  local W=$1 PARENT=$2 OLD
  OLD=$(git rev-parse "$W^")
  git rebase --onto "$PARENT" "$OLD" "$W" >/dev/null 2>&1 || true
  while [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; do
    local C
    C=$(git status --porcelain | grep -E '^(UU|DU|UD|AA)' | awk '{print $2}')
    for f in $C; do
      case "$f" in
        README.md) git checkout --theirs "$f" >/dev/null 2>&1 || true; git add "$f";;
        .gitignore|configs/*|docs/week1.md|src/*|tests/*|requirements.txt)
          git checkout --ours "$f" >/dev/null 2>&1 || true; git add "$f";;
        results/*|*.log|*.pid|*_done.txt|*_watch.sh)
          git rm --cached -q --force "$f" >/dev/null 2>&1 || git rm -q --force "$f" >/dev/null 2>&1;;
        *) echo "  !! 예상 밖 충돌: $f  (손으로 해결 후 git rebase --continue)"; return 1;;
      esac
    done
    GIT_EDITOR=true git rebase --continue >/dev/null 2>&1 || break
  done
  local N
  N=$(.venv/bin/python -m pytest tests -q 2>&1 | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+')
  sed -i -E "s/^pytest -q( +)# [0-9]+ tests on this branch/pytest -q\1# ${N} tests on this branch/" README.md
  git add README.md && git commit -q --amend --no-edit
  echo "$W -> $(git rev-parse --short HEAD) | ${N} passed"
}

if [ $# -eq 1 ]; then
  n=${1#week}; one "week$n" "week$((n-1))"
else
  for n in 2 3 4 5 6; do one "week$n" "week$((n-1))" || exit 1; done
fi

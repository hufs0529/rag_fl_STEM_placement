#!/usr/bin/env python3
"""아카이브 무결성 매니페스트 생성 (MANIFEST.sha256).

Zenodo 배포본이 전송 중에 또는 나중에 손상되지 않았음을 독자가 **스스로**
확인할 수 있어야 한다. `sha256sum -c MANIFEST.sha256` 한 줄로 끝난다.

재생성되는 것(.venv, __pycache__, data/)과 매니페스트 자신은 제외한다 -
자기 자신을 담으면 체크섬이 성립하지 않는다.

Usage:
  python scripts/make_manifest.py
"""

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"

# 아카이브에 들어가는 것은 **git 이 추적하는 파일**이다. 디렉터리 이름으로
# 거르면 규칙이 두 군데(여기와 .gitignore)로 갈라져 서로 어긋난다 - 실제로
# results/cache 의 38 MB 임베딩 캐시가 그렇게 들어왔다. 추적 목록을 쓰면
# "Zenodo 배포본은 main 브랜치 스냅샷" 이라는 설명과 정의상 일치한다.
# 매니페스트 자신은 제외한다 - 자기 체크섬을 자기가 담을 수 없다.


def tracked_files() -> list:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, check=True)
    names = [n for n in out.stdout.decode().split("\0") if n]
    return [ROOT / n for n in names
            if (ROOT / n).is_file() and (ROOT / n) != MANIFEST]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    files = sorted(tracked_files())
    lines = [f"{digest(p)}  {p.relative_to(ROOT).as_posix()}" for p in files]
    MANIFEST.write_text("\n".join(lines) + "\n")
    total = sum(p.stat().st_size for p in files)
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(files)} files, "
          f"{total / 1e6:.1f} MB")
    print("verify with:  sha256sum -c MANIFEST.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())

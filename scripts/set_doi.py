#!/usr/bin/env python3
"""Zenodo 가 발급한 DOI 를 모든 산출물에 한 번에 박고 PDF 를 다시 만든다.

**왜 스크립트인가.** DOI 는 배포가 끝나야 나오므로, 그 전에 쓰인 문서에는
자리표시자가 들어간다. 자리표시자가 남은 PDF 를 제출하는 것이 이 과정에서
가장 흔한 사고다 - 네 파일을 손으로 고치고 PDF 두 개를 다시 뽑는 동안 한 곳을
빠뜨리기 쉽다. 그래서 한 명령으로 묶고, **끝나고 자리표시자가 남아 있으면
실패**하게 한다.

Zenodo 는 식별자를 두 개 발급한다:
  concept DOI  - 모든 버전을 가리킨다. 보고서와 CITATION.cff 가 인용하는 값.
  version DOI  - 그 릴리스 하나를 가리킨다. 어떤 수치가 어느 스냅샷에서
                 나왔는지 되짚을 때 필요하므로 연구기록에 함께 적는다.

Usage:
  python scripts/set_doi.py --concept 10.5281/zenodo.12345678
  python scripts/set_doi.py --concept 10.5281/zenodo.12345678 \
                            --version 10.5281/zenodo.12345679
  python scripts/set_doi.py --check        # 자리표시자가 남았는지만 확인
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONCEPT_PLACEHOLDER = "10.5281/zenodo.XXXXXXXX"
VERSION_PLACEHOLDER = "10.5281/zenodo.YYYYYYYY"

# 자리표시자를 품을 수 있는 파일 전부. 하나라도 빠지면 그 파일만 옛 값으로
# 남는데, 그게 하필 인용되는 쪽이면 잘못된 DOI 가 퍼진다.
TARGETS = [
    "docs/final_report.md",
    "docs/research_records.md",
    "CITATION.cff",
    "README.md",
]

DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$")


def normalise(value: str) -> str:
    """`https://doi.org/10.5281/...` 로 붙여넣어도 받아들인다."""
    value = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.lower().startswith(prefix):
            value = value[len(prefix):]
    if not DOI_RE.match(value):
        raise SystemExit(f"DOI 형태가 아니다: {value!r}\n"
                         f"예: 10.5281/zenodo.12345678")
    return value


def remaining_placeholders() -> dict:
    found = {}
    for name in TARGETS:
        path = ROOT / name
        if not path.exists():
            continue
        text = path.read_text()
        hits = [p for p in (CONCEPT_PLACEHOLDER, VERSION_PLACEHOLDER) if p in text]
        if hits:
            found[name] = hits
    return found


def rebuild_pdfs() -> None:
    """두 PDF 를 다시 만든다. 부제는 md2pdf 를 처음 부를 때와 같아야 한다."""
    common = ("Jeonghun Kim · Macquarie University\n"
              "STEM research placement · 16 September – 28 October 2026\n")
    for src, out, tail in (
        ("docs/final_report.md", "docs/final_report.pdf",
         "Placement deliverable: report"),
        ("docs/research_records.md", "docs/research_records.pdf",
         "Placement deliverable: data / lab book / experimental records"),
    ):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "md2pdf.py"),
             str(ROOT / src), str(ROOT / out), "--subtitle", common + tail],
            check=True, cwd=ROOT,
        )


def verify_pdfs(concept: str, version: str = None) -> list:
    """**제출할 PDF 안에** DOI 가 실제로 들어갔는지 본다.

    마크다운만 확인하면 부족하다 - 심사자에게 가는 것은 PDF 이고, 재생성이
    실패했거나 옛 PDF 가 남아 있으면 마크다운은 맞는데 PDF 는 틀린 상태가
    된다. 그 조합이 가장 알아채기 어렵다.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  (pypdf 가 없어 PDF 내용 검증을 건너뛴다: pip install pypdf)")
        return []
    problems = []
    for name in ("docs/final_report.pdf", "docs/research_records.pdf"):
        path = ROOT / name
        if not path.exists():
            problems.append(f"{name}: 파일이 없다")
            continue
        text = "".join(page.extract_text() or "" for page in PdfReader(path).pages)
        if concept not in text:
            problems.append(f"{name}: concept DOI 가 보이지 않는다")
        for placeholder in (CONCEPT_PLACEHOLDER, VERSION_PLACEHOLDER):
            if placeholder in text:
                problems.append(f"{name}: 자리표시자 {placeholder} 가 남아 있다")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concept", help="concept DOI (모든 버전을 가리킨다)")
    ap.add_argument("--version", help="version DOI (이 릴리스 하나)")
    ap.add_argument("--check", action="store_true",
                    help="채우지 않고, 자리표시자가 남았는지만 본다")
    args = ap.parse_args()

    if args.check:
        left = remaining_placeholders()
        if not left:
            print("자리표시자 없음 - 제출 가능")
            return 0
        print("아직 자리표시자가 남아 있다:")
        for name, hits in left.items():
            print(f"  {name}: {', '.join(hits)}")
        return 1

    if not args.concept:
        raise SystemExit("--concept 가 필요하다 (또는 --check)")

    concept = normalise(args.concept)
    version = normalise(args.version) if args.version else None

    changed = []
    for name in TARGETS:
        path = ROOT / name
        if not path.exists():
            continue
        text = original = path.read_text()
        text = text.replace(CONCEPT_PLACEHOLDER, concept)
        if version:
            text = text.replace(VERSION_PLACEHOLDER, version)
        if text != original:
            path.write_text(text)
            changed.append(name)

    # CITATION.cff 의 doi 는 주석 처리되어 있다. 주석을 풀어 준다 - 풀지 않으면
    # GitHub 의 "Cite this repository" 가 DOI 없이 인용문을 만든다.
    cff = ROOT / "CITATION.cff"
    if cff.exists():
        text = cff.read_text()
        commented = f"# doi: {concept}"
        if commented in text:
            cff.write_text(text.replace(commented, f"doi: {concept}"))
            if "CITATION.cff" not in changed:
                changed.append("CITATION.cff")

    for name in changed:
        print(f"  updated {name}")
    if not changed:
        print("  바뀐 파일 없음 - 이미 채워져 있거나 자리표시자가 없다")

    print()
    rebuild_pdfs()

    left = remaining_placeholders()
    if left:
        print("\n자리표시자가 남았다 - 제출 전에 반드시 해결할 것:")
        for name, hits in left.items():
            print(f"  {name}: {', '.join(hits)}")
        if not version:
            print("  (version DOI 를 주지 않았다면 --version 으로 함께 넘길 것)")
        return 1

    problems = verify_pdfs(concept, version)
    if problems:
        print("\nPDF 검증 실패 - 제출하면 안 된다:")
        for line in problems:
            print(f"  {line}")
        return 1

    print("\n마크다운과 PDF 양쪽 모두 DOI 를 담고 있다. 다음으로:")
    print("  python scripts/make_manifest.py")
    print("  git add -A && git commit -m 'Record the Zenodo DOI'")
    return 0


if __name__ == "__main__":
    sys.exit(main())

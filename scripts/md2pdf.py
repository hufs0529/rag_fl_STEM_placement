#!/usr/bin/env python3
"""보고서 마크다운 -> PDF (MRes thesis 서식 강제).

범용 변환기가 아니라 이 보고서가 쓰는 문법만 다룬다 - md2docx.py 와 같은 부분
집합이다:
  # ## ### #### 제목 / 본문 / **굵게** / *기울임* / `코드`
  파이프 표 / ``` 코드블록 / > 인용 / - * 목록 / 1. 번호목록 / --- 구분선
  ![범례](경로)  그림 + 범례

**서식을 인자로 받지 않는 이유.** FOSE7901/7902 는 서식을 규정한다 - 12 pt
Times, 1.5 행간, 2 cm 여백, 쪽번호. 그 값을 호출자가 바꿀 수 있게 두면 어느
PDF 가 규정을 지켰는지 알 수 없다. 그래서 상수로 박아 두고, 바꿔야 하면 이
파일을 고치게 한다.

Usage:
  python scripts/md2pdf.py docs/final_report.md docs/final_report.pdf
  python scripts/md2pdf.py in.md out.pdf --title "..." --subtitle "..."
"""

import argparse
import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, Image, KeepTogether, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

# --- 규정 서식 (FOSE7901/7902) ----------------------------------------------
BODY_FONT = "Times-Roman"
BODY_SIZE = 12
LEADING = BODY_SIZE * 1.5          # 1.5 행간
MARGIN = 2 * cm
PAGE = A4
FRAME_W = PAGE[0] - 2 * MARGIN

INK = colors.HexColor("#14181E")
INK2 = colors.HexColor("#4A5568")
RULE = colors.HexColor("#C9CDD4")
SHADE = colors.HexColor("#F2F3F5")

# 표와 코드는 본문보다 작게 둔다. 본문이 12 pt 라는 규정을 지키면서도 17 cm
# 폭에 열이 들어가야 하기 때문이다 - 학위논문에서도 통상 그렇게 한다.
TABLE_SIZE = 10
CODE_SIZE = 9.5


def styles():
    base = dict(fontName=BODY_FONT, fontSize=BODY_SIZE, leading=LEADING, textColor=INK)
    return {
        "body": ParagraphStyle("body", alignment=TA_JUSTIFY, spaceAfter=6, **base),
        "h1": ParagraphStyle("h1", fontName="Times-Bold", fontSize=16, leading=21,
                             spaceBefore=18, spaceAfter=8, textColor=INK),
        "h2": ParagraphStyle("h2", fontName="Times-Bold", fontSize=13.5, leading=19,
                             spaceBefore=14, spaceAfter=6, textColor=INK),
        "h3": ParagraphStyle("h3", fontName="Times-Bold", fontSize=12, leading=18,
                             spaceBefore=11, spaceAfter=4, textColor=INK),
        "h4": ParagraphStyle("h4", fontName="Times-Italic", fontSize=12, leading=18,
                             spaceBefore=9, spaceAfter=3, textColor=INK),
        "quote": ParagraphStyle("quote", alignment=TA_LEFT, leftIndent=16,
                                rightIndent=8, spaceBefore=4, spaceAfter=8,
                                borderPadding=0, textColor=INK2,
                                fontName="Times-Italic", fontSize=BODY_SIZE,
                                leading=LEADING),
        "li": ParagraphStyle("li", alignment=TA_LEFT, leftIndent=18, bulletIndent=6,
                             spaceAfter=3, **base),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=CODE_SIZE,
                               leading=CODE_SIZE * 1.35, textColor=INK,
                               leftIndent=8, spaceBefore=4, spaceAfter=8),
        "cell": ParagraphStyle("cell", fontName=BODY_FONT, fontSize=TABLE_SIZE,
                               leading=TABLE_SIZE * 1.25, textColor=INK),
        "cellb": ParagraphStyle("cellb", fontName="Times-Bold", fontSize=TABLE_SIZE,
                                leading=TABLE_SIZE * 1.25, textColor=INK),
        "legend": ParagraphStyle("legend", fontName=BODY_FONT, fontSize=10.5,
                                 leading=15, textColor=INK2, alignment=TA_LEFT,
                                 spaceBefore=4, spaceAfter=12),
        "title": ParagraphStyle("title", fontName="Times-Bold", fontSize=18,
                                leading=24, alignment=TA_CENTER, spaceAfter=10,
                                textColor=INK),
        "subtitle": ParagraphStyle("subtitle", fontName="Times-Roman", fontSize=12.5,
                                   leading=18, alignment=TA_CENTER, spaceAfter=4,
                                   textColor=INK2),
    }


# --- 인라인 마크업 -----------------------------------------------------------
# reportlab 은 단락 안에서 <b> <i> <font> 를 받는다. 그러므로 먼저 &, <, > 를
# 이스케이프한 **뒤에** 태그를 넣어야 한다. 순서를 뒤집으면 본문의 부등호가
# 태그로 해석되어 조용히 사라진다.
_INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*\n]+?\*)")


def inline(text: str) -> str:
    out = []
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append(f"<b>{html.escape(part[2:-2])}</b>")
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            out.append(f'<font face="Courier" size="{BODY_SIZE - 1.5}">'
                       f"{html.escape(part[1:-1])}</font>")
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append(f"<i>{html.escape(part[1:-1])}</i>")
        else:
            out.append(html.escape(part))
    return "".join(out)


def cell_text(text: str, S):
    """표 칸. <br> 은 줄바꿈으로, 굵게는 전용 스타일로."""
    text = text.replace("<br>", "\n").strip()
    style = S["cellb"] if text.startswith("**") and text.endswith("**") else S["cell"]
    return Paragraph(inline(text).replace("\n", "<br/>"), style)


def make_table(rows, S):
    header, body = rows[0], rows[1:]
    data = [[cell_text(c, S) for c in header]] + \
           [[cell_text(c, S) for c in r] for r in body]
    ncols = max(len(r) for r in data)
    data = [r + [Paragraph("", S["cell"])] * (ncols - len(r)) for r in data]
    # 첫 열은 보통 라벨이라 좁게, 나머지는 균등하게 준다. 균등 분배만 쓰면
    # "d" 같은 한 글자 열이 본문 열과 같은 폭을 먹어 표가 넘친다.
    first = min(FRAME_W * 0.26, max(FRAME_W / ncols, 2.2 * cm))
    rest = (FRAME_W - first) / (ncols - 1) if ncols > 1 else FRAME_W
    widths = [first] + [rest] * (ncols - 1)
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SHADE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("LINEABOVE", (0, 0), (-1, 0), 0.75, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.75, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def figure(path: Path, legend: str, S):
    """그림 + 범례. 규정이 "범례를 그림 아래 또는 옆에" 요구한다.

    KeepTogether 로 묶어 그림과 범례가 쪽을 넘어 갈라지지 않게 한다.
    """
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    w = min(FRAME_W, iw * 0.75)
    img = Image(str(path), width=w, height=ih * (w / iw))
    img.hAlign = "CENTER"
    return KeepTogether([Spacer(1, 6), img,
                         Paragraph(inline(legend), S["legend"])])


# --- 파서 -------------------------------------------------------------------
IMG = re.compile(r"^!\[(?P<legend>.*)\]\((?P<path>[^)]+)\)\s*$")


def parse(md: str, S, root: Path):
    flow, i = [], 0
    lines = md.split("\n")
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):                      # 코드블록
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            text = "<br/>".join(html.escape(b).replace(" ", "&nbsp;") for b in buf)
            flow.append(Table([[Paragraph(text, S["code"])]], colWidths=[FRAME_W],
                              style=TableStyle([
                                  ("BACKGROUND", (0, 0), (-1, -1), SHADE),
                                  ("BOX", (0, 0), (-1, -1), 0.25, RULE),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                  ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                                  ("TOPPADDING", (0, 0), (-1, -1), 6),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                              ]), hAlign="LEFT"))
            flow.append(Spacer(1, 4))
            continue

        m = IMG.match(stripped)                             # 그림
        if m:
            path = root / m.group("path")
            if path.exists():
                flow.append(figure(path, m.group("legend"), S))
            else:
                print(f"  warning: figure not found, skipped: {path}", file=sys.stderr)
            i += 1
            continue

        if stripped.startswith("|"):                        # 표
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):   # 구분행은 버린다
                    rows.append(cells)
                i += 1
            if rows:
                flow.append(make_table(rows, S))
                flow.append(Spacer(1, 8))
            continue

        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):    # 구분선
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", thickness=0.5, color=RULE))
            flow.append(Spacer(1, 6))
            i += 1
            continue

        if stripped.startswith("#"):                        # 제목
            level = len(stripped) - len(stripped.lstrip("#"))
            flow.append(Paragraph(inline(stripped[level:].strip()),
                                  S[f"h{min(level, 4)}"]))
            i += 1
            continue

        if stripped.startswith(">"):                        # 인용 (여러 줄)
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            flow.append(Paragraph(inline(" ".join(buf)), S["quote"]))
            continue

        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)   # 목록
        if m:
            while i < len(lines):
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if not mm:
                    break
                marker = "•" if mm.group(2) in "-*" else mm.group(2)
                flow.append(Paragraph(inline(mm.group(3)), S["li"],
                                      bulletText=marker))
                i += 1
            flow.append(Spacer(1, 4))
            continue

        buf = []                                            # 본문 단락
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^\s*(#|\||>|```|!\[|[-*]\s|\d+\.\s|-{3,}$)", lines[i]):
            buf.append(lines[i].strip())
            i += 1
        flow.append(Paragraph(inline(" ".join(buf)), S["body"]))
    return flow


def build(src: Path, out: Path, title=None, subtitle=None):
    S = styles()
    md = src.read_text()

    # 첫 h1 을 표제로 올린다. 본문에 다시 찍으면 표제가 두 번 나온다.
    lines = md.split("\n")
    if title is None:
        for n, line in enumerate(lines):
            if line.startswith("# "):
                title = line[2:].strip()
                lines = lines[:n] + lines[n + 1:]
                break
        md = "\n".join(lines)

    doc = BaseDocTemplate(
        str(out), pagesize=PAGE,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
        title=title or src.stem, author="Jeonghun Kim",
    )

    def footer(canvas, document):
        """쪽번호 - 규정이 "sequentially numbered pages" 를 요구한다."""
        canvas.saveState()
        canvas.setFont(BODY_FONT, 10)
        canvas.setFillColor(INK2)
        canvas.drawCentredString(PAGE[0] / 2, MARGIN * 0.45, str(document.page))
        canvas.restoreState()

    frame = Frame(MARGIN, MARGIN, FRAME_W, PAGE[1] - 2 * MARGIN, id="body",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])

    flow = []
    if title:
        flow.append(Paragraph(inline(title), S["title"]))
    for part in (subtitle or "").split("\n"):
        if part.strip():
            flow.append(Paragraph(inline(part.strip()), S["subtitle"]))
    if title or subtitle:
        flow.append(Spacer(1, 10))
        flow.append(HRFlowable(width="100%", thickness=0.5, color=RULE))
        flow.append(Spacer(1, 8))
    # 그림 경로는 **저장소 루트** 기준으로 푼다. 원본 파일 위치 기준으로 풀면
    # docs/ 안의 문서와 다른 데 둔 문서가 서로 다르게 해석된다.
    flow += parse(md, S, Path(__file__).resolve().parents[1])

    doc.build(flow)
    words = len(re.findall(r"\b[\w'-]+\b", re.sub(r"[|`#>*]", " ", md)))
    print(f"saved {out}  ({doc.page} pages, ~{words:,} words of source)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default=None)
    args = ap.parse_args()
    build(Path(args.source), Path(args.output), args.title, args.subtitle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

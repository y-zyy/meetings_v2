"""
회의록 생성기
사용법: python generate.py input.json [output.docx]

필요 패키지: pip install python-docx

JSON 키:
  회의명, 일시 및 장소, 참석자,
  회의 안건, 주요 회의 내용, Action Item
"""

import sys
import json
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


# ── 색상 ──────────────────────────────────────────────────────────────
BLUE       = "1F497D"   # 라벨·섹션 헤더 배경
WHITE      = "FFFFFF"   # 값 셀 배경 / 흰색 텍스트
DARK       = "1A1A1A"   # 본문 텍스트
DARK_BDR   = "333333"   # 값 셀 테두리
WHITE_BDR  = "FFFFFF"   # 라벨 셀 테두리 (배경과 동일 → 안 보임)

FONT_NAME  = "맑은 고딕"


# ── XML 헬퍼 ──────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color: str):
    """셀 배경색 설정."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    for old in tcPr.findall(qn("w:shd")):
        tcPr.remove(old)
    tcPr.append(shd)


def set_cell_border(cell, color: str, size: int = 12):
    """셀 4면 테두리 설정 (size: 1/8 pt 단위)."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:tcBorders")):
        tcPr.remove(old)
    tcBorders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        tcBorders.append(el)
    tcPr.append(tcBorders)


def set_cell_margin(cell, top=80, bottom=80, left=160, right=160):
    """셀 내부 여백 설정 (단위: DXA = 1/1440 inch)."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:tcMar")):
        tcPr.remove(old)
    mar = OxmlElement("w:tcMar")
    for side, val in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tcPr.append(mar)


def set_cell_valign(cell, align: str = "center"):
    """셀 수직 정렬 (top / center / bottom)."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:vAlign")):
        tcPr.remove(old)
    va = OxmlElement("w:vAlign")
    va.set(qn("w:val"), align)
    tcPr.append(va)


def set_row_height(row, height_cm: float):
    """행 높이 고정 (cm)."""
    tr = row._tr
    trPr = tr.get_or_add_trPr()
    for old in trPr.findall(qn("w:trHeight")):
        trPr.remove(old)
    trH = OxmlElement("w:trHeight")
    trH.set(qn("w:val"), str(int(height_cm * 567)))  # 1cm ≈ 567 DXA
    trH.set(qn("w:hRule"), "atLeast")
    trPr.append(trH)


def remove_table_border(table):
    """테이블 전체 바깥 테두리 제거."""
    tbl = table._tbl
    tblPr = tbl.find(qn("w:tblPr"))
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr")
        tbl.insert(0, tblPr)
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    tblBorders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "none")
        tblBorders.append(el)
    tblPr.append(tblBorders)


def set_paragraph_spacing(para, before=0, after=0, line=None):
    pPr = para._p.get_or_add_pPr()
    for old in pPr.findall(qn("w:spacing")):
        pPr.remove(old)
    sp = OxmlElement("w:spacing")
    sp.set(qn("w:before"), str(before))
    sp.set(qn("w:after"), str(after))
    if line:
        sp.set(qn("w:line"), str(line))
        sp.set(qn("w:lineRule"), "auto")
    pPr.append(sp)


# ── 텍스트 헬퍼 ───────────────────────────────────────────────────────

def add_run(para, text: str, size_pt: float = 10, bold: bool = False,
            color: str = DARK, font: str = FONT_NAME):
    run = para.add_run(text)
    run.font.name = font
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:eastAsia"), font)
    return run


# ── 값 정규화 ─────────────────────────────────────────────────────────

def to_lines(val) -> list[str]:
    """문자열 또는 배열 → 줄 목록."""
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        return [l.rstrip() for l in val.splitlines()]
    return [str(val)]


# ── 문서 빌더 ─────────────────────────────────────────────────────────

def build_document(data: dict) -> Document:
    doc = Document()

    sec = doc.sections[0]
    sec.page_width  = Cm(21)
    sec.page_height = Cm(29.7)
    for attr in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, attr, Cm(1.9))

    style = doc.styles["Normal"]
    style.paragraph_format.space_before = Pt(0)
    style.paragraph_format.space_after  = Pt(0)

    _add_title(doc)
    _add_spacer(doc, 4)
    _add_info_table(doc, data)
    _add_spacer(doc, 10)

    for key in ["회의 안건", "주요 회의 내용", "Action Item"]:
        _add_section(doc, key, to_lines(data.get(key, [])))
        _add_spacer(doc, 10)

    return doc


def _add_spacer(doc: Document, pt: float = 6):
    p = doc.add_paragraph()
    set_paragraph_spacing(p, before=0, after=0)
    p.paragraph_format.space_after = Pt(pt)


def _add_title(doc: Document):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_spacing(p, before=120, after=60)
    add_run(p, "회  의  록", size_pt=22, bold=True, color=DARK)


def _add_info_table(doc: Document, data: dict):
    raw = str(data.get("일시 및 장소", ""))
    if "/" in raw:
        idx = raw.index("/")
        time_val  = raw[:idx].strip()
        place_val = raw[idx+1:].strip()
    else:
        time_val, place_val = raw, ""

    fields = [
        ("회의명",  str(data.get("회의명",  ""))),
        ("일시",    time_val),
        ("장소",    place_val),
        ("참석자",  str(data.get("참석자",  ""))),
    ]

    LABEL_W = Cm(3.8)
    VALUE_W = Cm(21 - 1.9*2 - 3.8)

    table = doc.add_table(rows=len(fields), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    remove_table_border(table)

    for i, (label, value) in enumerate(fields):
        row = table.rows[i]
        set_row_height(row, 0.85)

        lc = row.cells[0]
        lc.width = LABEL_W
        set_cell_bg(lc, BLUE)
        set_cell_border(lc, WHITE_BDR, size=8)
        set_cell_margin(lc, top=60, bottom=60, left=120, right=120)
        set_cell_valign(lc, "center")
        lc.paragraphs[0].clear()
        lp = lc.paragraphs[0]
        lp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_paragraph_spacing(lp)
        add_run(lp, label, size_pt=10, bold=True, color=WHITE)

        vc = row.cells[1]
        vc.width = VALUE_W
        set_cell_bg(vc, WHITE)
        set_cell_border(vc, DARK_BDR, size=12)
        set_cell_margin(vc, top=60, bottom=60, left=160, right=120)
        set_cell_valign(vc, "center")
        vc.paragraphs[0].clear()
        vp = vc.paragraphs[0]
        set_paragraph_spacing(vp)
        add_run(vp, value, size_pt=10, color=DARK)


def _add_section(doc: Document, title: str, lines: list[str]):
    h_table = doc.add_table(rows=1, cols=1)
    h_table.alignment = WD_TABLE_ALIGNMENT.LEFT
    remove_table_border(h_table)

    hc = h_table.rows[0].cells[0]
    set_row_height(h_table.rows[0], 0.85)
    set_cell_bg(hc, BLUE)
    set_cell_border(hc, BLUE, size=8)
    set_cell_margin(hc, top=80, bottom=80, left=200, right=200)
    set_cell_valign(hc, "center")
    hc.paragraphs[0].clear()
    hp = hc.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_spacing(hp)
    add_run(hp, title, size_pt=12, bold=True, color=WHITE)

    if not lines:
        lines = [""]

    b_table = doc.add_table(rows=1, cols=1)
    b_table.alignment = WD_TABLE_ALIGNMENT.LEFT
    remove_table_border(b_table)

    bc = b_table.rows[0].cells[0]
    set_cell_bg(bc, WHITE)
    set_cell_border(bc, DARK_BDR, size=12)
    set_cell_margin(bc, top=80, bottom=80, left=200, right=200)

    for i, line in enumerate(lines):
        if i == 0:
            p = bc.paragraphs[0]
            p.clear()
        else:
            p = bc.add_paragraph()
        set_paragraph_spacing(p, before=0, after=40)
        add_run(p, line, size_pt=10, color=DARK)


# ── 음성인식 원문 DOCX ────────────────────────────────────────────────

def _split_transcript(transcript: str) -> list[str]:
    """전사 텍스트를 화면에 표시하기 좋은 줄 단위로 분할합니다."""
    import re
    sentences = re.split(r"(?<=[.!?。])\s+", transcript.strip())
    lines: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > 120:
            if current:
                lines.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent
    if current:
        lines.append(current.strip())
    return lines if lines else [transcript]


def build_transcript_document(transcript: str, metadata: dict) -> Document:
    """ASR 전사 결과를 회의록과 동일한 포맷의 DOCX로 생성합니다."""
    date_time = metadata.get("일시", "")
    location = metadata.get("장소", "")
    date_place = " / ".join(part for part in [date_time, location] if part)

    agenda_raw = metadata.get("안건", [])
    agenda_lines = (
        [f"{i+1}. {a}" for i, a in enumerate(agenda_raw)]
        if agenda_raw else []
    )

    data = {
        "회의명": "음성 인식 결과",
        "일시 및 장소": date_place,
        "참석자": metadata.get("참석자", ""),
        "회의 안건": agenda_lines,
        "주요 회의 내용": _split_transcript(transcript),
        "Action Item": [],
    }
    return build_document(data)


# ── CLI 진입점 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python generate.py input.json [output.docx]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix(".docx")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    doc = build_document(data)
    doc.save(output_path)
    print(f"저장 완료: {output_path}")

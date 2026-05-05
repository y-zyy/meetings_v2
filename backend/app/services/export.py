"""Generate DOCX, PDF, and TXT exports from a meeting record."""

import io
from datetime import date

from docx import Document
from docx.shared import Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_date(d: date | None) -> str:
    return d.strftime("%Y-%m-%d") if d else "미지정"


def _fmt_duration(secs: int | None) -> str:
    if not secs:
        return "-"
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}"


# ── DOCX ─────────────────────────────────────────────────────────────────────

def build_docx(meeting) -> bytes:
    doc = Document()

    # Title
    title_p = doc.add_heading(meeting.title, level=1)
    title_p.runs[0].font.color.rgb = RGBColor(0x1B, 0x14, 0x64)

    # Meta table
    meta = [
        ["일시", _fmt_date(meeting.meeting_date)],
        ["장소", meeting.location or "-"],
        ["참석자", meeting.attendees or "-"],
        ["소요 시간", _fmt_duration(meeting.duration_seconds)],
    ]
    tbl = doc.add_table(rows=len(meta), cols=2)
    tbl.style = "Table Grid"
    for i, (k, v) in enumerate(meta):
        tbl.rows[i].cells[0].text = k
        tbl.rows[i].cells[1].text = v
        tbl.rows[i].cells[0].paragraphs[0].runs[0].bold = True
    doc.add_paragraph()

    # Summary
    doc.add_heading("주요 내용 요약", level=2)
    doc.add_paragraph(meeting.summary or "")

    # Decisions
    doc.add_heading("결정 사항", level=2)
    for d in meeting.decisions:
        doc.add_paragraph(d.content, style="List Bullet")

    # Action items
    doc.add_heading("액션 아이템", level=2)
    if meeting.action_items:
        at = doc.add_table(rows=1, cols=4)
        at.style = "Table Grid"
        hdr = at.rows[0].cells
        for i, h in enumerate(["내용", "담당자", "기한", "상태"]):
            hdr[i].text = h
            hdr[i].paragraphs[0].runs[0].bold = True
        for item in meeting.action_items:
            row = at.add_row().cells
            row[0].text = item.content
            row[1].text = item.assignee or "-"
            row[2].text = _fmt_date(item.due_date)
            row[3].text = "완료" if item.status == "done" else "진행 중"

    # Transcript
    if meeting.transcript:
        doc.add_heading("발화 기록 (ASR)", level=2)
        doc.add_paragraph(meeting.transcript)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ── PDF ──────────────────────────────────────────────────────────────────────

_ACCENT = colors.HexColor("#1B1464")


def build_pdf(meeting) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    h1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=_ACCENT, fontSize=16, spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=_ACCENT, fontSize=12, spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=16)

    story = []
    story.append(Paragraph(meeting.title, h1))
    story.append(Spacer(1, 0.3*cm))

    meta_data = [
        ["일시", _fmt_date(meeting.meeting_date)],
        ["장소", meeting.location or "-"],
        ["참석자", meeting.attendees or "-"],
        ["소요 시간", _fmt_duration(meeting.duration_seconds)],
    ]
    meta_tbl = Table(meta_data, colWidths=[3*cm, 14*cm])
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ECEEF9")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d2d5da")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 0.5*cm))

    if meeting.summary:
        story.append(Paragraph("주요 내용 요약", h2))
        story.append(Paragraph(meeting.summary, body))

    if meeting.decisions:
        story.append(Paragraph("결정 사항", h2))
        for d in meeting.decisions:
            story.append(Paragraph(f"• {d.content}", body))

    if meeting.action_items:
        story.append(Paragraph("액션 아이템", h2))
        ai_data = [["내용", "담당자", "기한", "상태"]]
        for item in meeting.action_items:
            ai_data.append([
                item.content,
                item.assignee or "-",
                _fmt_date(item.due_date),
                "완료" if item.status == "done" else "진행 중",
            ])
        ai_tbl = Table(ai_data, colWidths=[9*cm, 3*cm, 2.5*cm, 2.5*cm])
        ai_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d2d5da")),
            ("ROWBACKGROUNDS", (1, 0), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(ai_tbl)

    if meeting.transcript:
        story.append(Paragraph("발화 기록 (ASR)", h2))
        story.append(Paragraph(meeting.transcript.replace("\n", "<br/>"), body))

    doc.build(story)
    return buf.getvalue()


# ── TXT ──────────────────────────────────────────────────────────────────────

def build_txt(meeting) -> bytes:
    lines = [
        f"{'='*60}",
        f"회의록: {meeting.title}",
        f"{'='*60}",
        f"일시    : {_fmt_date(meeting.meeting_date)}",
        f"장소    : {meeting.location or '-'}",
        f"참석자  : {meeting.attendees or '-'}",
        f"소요시간: {_fmt_duration(meeting.duration_seconds)}",
        "",
        "[주요 내용 요약]",
        meeting.summary or "",
        "",
        "[결정 사항]",
    ]
    for d in meeting.decisions:
        lines.append(f"  • {d.content}")
    lines += ["", "[액션 아이템]"]
    for item in meeting.action_items:
        status_label = "완료" if item.status == "done" else "진행 중"
        lines.append(f"  • {item.content} / 담당: {item.assignee or '-'} / 기한: {_fmt_date(item.due_date)} / {status_label}")
    if meeting.transcript:
        lines += ["", "[발화 기록 (ASR)]", meeting.transcript]
    return "\n".join(lines).encode("utf-8")

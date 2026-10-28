"""Render report.md to report.pdf.

Owned by: Somyak Mohanta (241CS257)

    python report/build_pdf.py

A small Markdown subset renderer built on ReportLab -- enough for this report (headings,
paragraphs, pipe tables, figures, code blocks, quotes, lists) and nothing more. It exists so
the PDF is a build artefact rather than something someone exported by hand once and then
forgot to update when the numbers changed.

Fonts come from the DejaVu family bundled with matplotlib, which is already a dependency.
ReportLab's built-in fonts are Latin-1 only and would render the report's ×, ≈, σ and — as
black boxes.
"""

from __future__ import annotations

import html
import os
import re
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5a5a5a")
ACCENT = colors.HexColor("#0072B2")
RULE = colors.HexColor("#d0d0d0")
CODE_BG = colors.HexColor("#f4f4f6")


def register_fonts() -> None:
    faces = {
        "DejaVu": "DejaVuSans.ttf",
        "DejaVu-Bold": "DejaVuSans-Bold.ttf",
        "DejaVu-Italic": "DejaVuSans-Oblique.ttf",
        "DejaVu-BoldItalic": "DejaVuSans-BoldOblique.ttf",
        "DejaVuMono": "DejaVuSansMono.ttf",
        "DejaVuMono-Bold": "DejaVuSansMono-Bold.ttf",
    }
    for name, filename in faces.items():
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                                  italic="DejaVu-Italic", boldItalic="DejaVu-BoldItalic")


def build_styles():
    base = getSampleStyleSheet()
    s = {}
    s["body"] = ParagraphStyle("body", parent=base["Normal"], fontName="DejaVu",
                               fontSize=9.3, leading=14, spaceAfter=7, textColor=INK,
                               alignment=TA_JUSTIFY)
    s["h1"] = ParagraphStyle("h1", parent=s["body"], fontName="DejaVu-Bold", fontSize=19,
                             leading=24, spaceBefore=4, spaceAfter=10, alignment=0)
    s["h2"] = ParagraphStyle("h2", parent=s["body"], fontName="DejaVu-Bold", fontSize=13.5,
                             leading=18, spaceBefore=16, spaceAfter=7, alignment=0,
                             textColor=ACCENT)
    s["h3"] = ParagraphStyle("h3", parent=s["body"], fontName="DejaVu-Bold", fontSize=11,
                             leading=15, spaceBefore=11, spaceAfter=5, alignment=0)
    s["h4"] = ParagraphStyle("h4", parent=s["body"], fontName="DejaVu-Bold", fontSize=9.8,
                             leading=13, spaceBefore=8, spaceAfter=4, alignment=0)
    s["bullet"] = ParagraphStyle("bullet", parent=s["body"], leftIndent=12,
                                 bulletIndent=3, spaceAfter=3)
    s["quote"] = ParagraphStyle("quote", parent=s["body"], leftIndent=10, rightIndent=6,
                                borderPadding=(5, 5, 5, 8), backColor=colors.HexColor("#fdf6e3"),
                                borderColor=colors.HexColor("#e0c96a"), borderWidth=0,
                                spaceBefore=6, spaceAfter=8, textColor=colors.HexColor("#4a3f10"))
    s["code"] = ParagraphStyle("code", parent=s["body"], fontName="DejaVuMono", fontSize=7.9,
                               leading=11, backColor=CODE_BG, borderPadding=(5, 5, 5, 6),
                               spaceBefore=4, spaceAfter=8, alignment=0, textColor=INK)
    s["cell"] = ParagraphStyle("cell", parent=s["body"], fontSize=7.9, leading=10.5,
                               spaceAfter=0, alignment=0)
    s["cellhead"] = ParagraphStyle("cellhead", parent=s["cell"], fontName="DejaVu-Bold",
                                   textColor=colors.white)
    s["caption"] = ParagraphStyle("caption", parent=s["body"], fontSize=8, leading=11,
                                  textColor=MUTED, alignment=1, spaceBefore=3, spaceAfter=10)
    return s


INLINE_CODE = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*([^*]+)\*\*")
ITALIC = re.compile(r"(?<![\*\w])\*([^*\n]+)\*(?!\*)")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def inline(text: str) -> str:
    """Markdown inline formatting -> ReportLab's mini-HTML."""
    text = html.escape(text, quote=False)
    # Code first: its contents must not be re-parsed for bold/italic.
    holds: list[str] = []

    def stash(m):
        holds.append(f'<font face="DejaVuMono" size="8.2" backColor="#f0f0f3">'
                     f'{m.group(1)}</font>')
        return f"\x00{len(holds)-1}\x00"

    text = INLINE_CODE.sub(stash, text)
    text = LINK.sub(r'<font color="#0072B2">\1</font>', text)
    text = BOLD.sub(r"<b>\1</b>", text)
    text = ITALIC.sub(r"<i>\1</i>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: holds[int(m.group(1))], text)


def make_table(rows: list[list[str]], s) -> Table:
    header, body = rows[0], rows[1:]
    ncols = len(header)
    data = [[Paragraph(inline(c), s["cellhead"]) for c in header]]
    data += [[Paragraph(inline(c), s["cell"]) for c in r] for r in body]

    # First column carries labels (policy names); give it more room.
    if ncols > 1:
        first = min(0.34, max(0.18, 1.6 / ncols))
        widths = [CONTENT_W * first] + [CONTENT_W * (1 - first) / (ncols - 1)] * (ncols - 1)
    else:
        widths = [CONTENT_W]

    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
    ]))
    return t


def add_image(story, path: Path, alt: str, s) -> None:
    if not path.exists():
        story.append(Paragraph(f"[missing figure: {path.name}]", s["caption"]))
        return
    from reportlab.lib.utils import ImageReader

    iw, ih = ImageReader(str(path)).getSize()
    w = min(CONTENT_W, 165 * mm)
    h = w * ih / iw
    max_h = 105 * mm
    if h > max_h:
        h, w = max_h, max_h * iw / ih
    block = [Image(str(path), width=w, height=h)]
    if alt:
        block.append(Paragraph(inline(alt), s["caption"]))
    story.append(KeepTogether(block))


def render(md: str, s) -> list:
    story: list = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):                       # fenced code
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            body = html.escape("\n".join(buf)).replace(" ", "&nbsp;").replace("\n", "<br/>")
            story.append(Paragraph(body, s["code"]))
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and \
                set(lines[i + 1].strip()) <= set("|-: "):    # pipe table
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw = lines[i].strip().strip("|")
                if set(raw) <= set("-: |"):
                    i += 1
                    continue
                rows.append([c.strip() for c in raw.split("|")])
                i += 1
            if rows:
                story.append(Spacer(1, 3))
                story.append(make_table(rows, s))
                story.append(Spacer(1, 9))
            continue

        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", stripped)  # image
        if m:
            add_image(story, (HERE / m.group(2)).resolve(), m.group(1), s)
            i += 1
            continue

        if re.match(r"^(-{3,}|\*{3,})$", stripped):           # rule
            story.append(Spacer(1, 5))
            story.append(HRFlowable(width="100%", color=RULE, thickness=0.6))
            story.append(Spacer(1, 7))
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)", stripped)           # heading
        if m:
            level = len(m.group(1))
            story.append(Paragraph(inline(m.group(2)), s[f"h{level}"]))
            if level == 1:
                story.append(HRFlowable(width="100%", color=ACCENT, thickness=1.2,
                                        spaceAfter=8))
            i += 1
            continue

        if stripped.startswith(">"):                           # blockquote
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            story.append(Paragraph(inline(" ".join(buf)), s["quote"]))
            continue

        m = re.match(r"^(\d+)\.\s+(.*)", stripped)             # ordered list
        if m:
            buf = [m.group(2)]
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() \
                    and not re.match(r"^\d+\.\s", lines[i].strip()):
                buf.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline(" ".join(buf)), s["bullet"],
                                   bulletText=f"{m.group(1)}."))
            continue

        if stripped.startswith(("- ", "* ")):                  # bullet list
            buf = [stripped[2:]]
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() \
                    and not lines[i].strip().startswith(("- ", "* ")):
                buf.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline(" ".join(buf)), s["bullet"], bulletText="•"))
            continue

        buf = [stripped]                                       # paragraph
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#{1,4}\s|\||!\[|>|-\s|\*\s|\d+\.\s|```|-{3,}$)", lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline(" ".join(buf)), s["body"]))

    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 12 * mm,
                      "Benchmarking LLM Inference Scheduling Policies — Group 18, NITK Surathkal")
    canvas.drawRightString(PAGE_W - MARGIN, 12 * mm, str(doc.page))
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, 15 * mm, PAGE_W - MARGIN, 15 * mm)
    canvas.restoreState()


def main() -> None:
    register_fonts()
    s = build_styles()
    md = (HERE / "report.md").read_text(encoding="utf-8")
    out = HERE / "report.pdf"

    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title="Benchmarking Scheduling Policies for LLM Inference Serving",
        author="Group 18, NITK Surathkal",
        subject="Cloud Computing Research Assignment",
    )
    doc.build(render(md, s), onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()

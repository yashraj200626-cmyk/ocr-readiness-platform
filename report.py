"""
PDF report generator for OCR Readiness Platform.
Uses reportlab to produce a clean single-page report.
"""

import io
from datetime import datetime
from typing import Dict, Any, Optional, List

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.enums import (
    TA_CENTER
)

from factors import DISPLAY_NAMES, WEIGHTS


# Brand colours
NAVY = colors.HexColor("#1A2B4A")
BLUE = colors.HexColor("#2F80ED")
TEAL = colors.HexColor("#00C4B4")
GREEN = colors.HexColor("#10B981")
ORANGE = colors.HexColor("#F59E0B")
RED = colors.HexColor("#EF4444")
LIGHT = colors.HexColor("#F8FAFC")
LIGHTBLUE = colors.HexColor("#EEF6FF")
GREY = colors.HexColor("#6B7280")


def status_color(score):

    if score >= 81:
        return GREEN

    elif score >= 61:
        return BLUE

    elif score >= 41:
        return ORANGE

    return RED

def page_number(canvas, doc):

    canvas.saveState()

    canvas.setFont("Helvetica",9)

    canvas.setFillColor(colors.grey)

    canvas.drawRightString(
        19.5*cm,
        1*cm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


def generate_pdf_report(
        image_name,
        factor_results,
        ocr_readiness,
        ocr_confidence,
        recommendations,
        image_path=None,
        ocr_text=""
):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.6*cm,
        rightMargin=1.6*cm,
        topMargin=1.8*cm,
        bottomMargin=1.8*cm
    )

    styles = getSampleStyleSheet()

    story=[]

    title_style = ParagraphStyle(

        "Title",

        parent=styles["Heading1"],

        alignment=TA_CENTER,

        textColor=NAVY,

        fontSize=24,

        leading=30,

        spaceAfter=18,

        fontName="Helvetica-Bold"

    )

    heading = ParagraphStyle(

        "Heading",

        parent=styles["Heading2"],

        textColor=NAVY,

        fontSize=16,

        spaceBefore=16,

        spaceAfter=10,

        fontName="Helvetica-Bold"

    )

    normal = ParagraphStyle(

        "Normal",

        parent=styles["BodyText"],

        fontSize=10,

        leading=17

    )

    small = ParagraphStyle(

        "Small",

        parent=styles["BodyText"],

        fontSize=8,

        textColor=GREY

    )


    # --------------------------------------------------
    # HEADER
    # --------------------------------------------------

    story.append(Paragraph(
        "OCR Readiness Evaluation Report",
        title_style
        ))

    story.append(Paragraph(
    "AI Based OCR Image Quality Assessment Platform",
    small
))

    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%d %B %Y • %I:%M %p')}",
        small
    ))

    story.append(Spacer(1, 8))

    story.append(
        HRFlowable(
            width="100%",
            thickness=2,
            color=TEAL
        )
    )

    story.append(Spacer(1, 15))

    # --------------------------------------------------
    # OCR READINESS SCORE
    # --------------------------------------------------

    status = factor_results.get("ocr_readiness_status", "Unknown")

    score_table = Table(
        [[
            Paragraph(
                f"<font size=30><b>{ocr_readiness:.1f}/100</b></font>",
                normal
            ),
            (
            Paragraph(
                f"<font color='{status_color(ocr_readiness).hexval()}'><b>{status}</b></font>",
                normal
            )
            )
        ]],
        colWidths=[9*cm,6*cm]
    )

    score_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),LIGHT),
        ("BOX",(0,0),(-1,-1),1,TEAL),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),10),
        ("LEFTPADDING",(0,0),(-1,-1),12)
    ]))

    story.append(Paragraph("OCR Readiness Score", heading))
    story.append(score_table)

    story.append(Spacer(1,10))

    if ocr_confidence is not None:
        story.append(
            Paragraph(
                f"<b>Tesseract OCR Confidence:</b> {ocr_confidence:.2f}%",
                normal
            )
        )

    story.append(Spacer(1,10))

    # --------------------------------------------------
    # FACTOR SCORES
    # --------------------------------------------------

    story.append(Paragraph("Factor Scores", heading))

    table_data = [["Factor", "Score", "Status", "Weight"]]

    for key, display in DISPLAY_NAMES.items():

        result = factor_results.get(key, {})

        score = result.get("score", "-")
        status = result.get("status", "-")
        weight = f"{WEIGHTS[key]*100:.0f}%"

        table_data.append([
            display,
            str(score),
            status,
            weight
        ])

    factor_table = Table(
        table_data,
        colWidths=[8*cm,2.2*cm,3.5*cm,2*cm]
    )

    style = TableStyle([
        ("BACKGROUND",(0,0),(-1,0),NAVY),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("GRID",(0,0),(-1,-1),0.3,colors.grey),
        ("BACKGROUND",(0,1),(-1,-1),LIGHT),
        ("ALIGN",(1,1),(-1,-1),"CENTER"),
        ("BOTTOMPADDING",(0,0),(-1,0),8)
    ])

    for row, key in enumerate(DISPLAY_NAMES.keys(), start=1):

        clr = status_color(
            factor_results.get(key, {}).get("score", 50)
        )

        style.add("TEXTCOLOR",(1,row),(1,row),clr)
        style.add("TEXTCOLOR",(2,row),(2,row),clr)
        style.add("FONTNAME",(1,row),(2,row),"Helvetica-Bold")

    factor_table.setStyle(style)

    story.append(factor_table)
    story.append(Spacer(1,12))

    # ── Recommendations ──────────────────────────
    story.append(Paragraph("Recommendations", heading))

    if recommendations:

        for rec in recommendations:

            clean = rec.replace("**","").replace("🔧","•").replace("✅","✓")

            story.append(
                Paragraph(f"• {clean}", normal)
            )

    else:

        story.append(
            Paragraph(
                "No improvements required. The uploaded image already satisfies the recommended OCR quality standards.",
                normal
            )
        )

    story.append(Spacer(1,12))

    # ── Footer ───────────────────────────────────
    story.append(Spacer(1,20))
    story.append(Paragraph(
        "OCR Readiness Evaluation Platform · SNLP Department · "
        "Team: Yash (Lead), Vivek, Mansi, Krish",
        ParagraphStyle(
        "Footer",
        parent=normal,
        alignment=TA_CENTER,
        fontSize=8,
        textColor=GREY
        )
    ))

    doc.build(
        story,
        onFirstPage=page_number,
        onLaterPages=page_number
    )
    return buffer.getvalue()

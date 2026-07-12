"""
Professional PDF Report Generator
OCR Readiness Evaluation Platform
"""

import io
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Image,
    PageBreak,
)

from factors import DISPLAY_NAMES, WEIGHTS


# ----------------------------------------------------
# Brand Colours
# ----------------------------------------------------

NAVY = colors.HexColor("#1A2B4A")
TEAL = colors.HexColor("#00C4B4")
BLUE = colors.HexColor("#2F80ED")
GREEN = colors.HexColor("#10B981")
ORANGE = colors.HexColor("#F59E0B")
RED = colors.HexColor("#EF4444")
LIGHT = colors.HexColor("#F8FAFC")
GREY = colors.HexColor("#6B7280")


# ----------------------------------------------------
# Score Colour
# ----------------------------------------------------

def status_color(score):

    if score >= 81:
        return GREEN

    elif score >= 61:
        return BLUE

    elif score >= 41:
        return ORANGE

    return RED


# ----------------------------------------------------
# Footer
# ----------------------------------------------------

def page_number(canvas, doc):

    canvas.saveState()

    canvas.setFont("Helvetica", 9)

    canvas.setFillColor(GREY)

    canvas.drawString(
        2 * cm,
        1 * cm,
        "OCR Readiness Evaluation Platform"
    )

    canvas.drawRightString(
        19 * cm,
        1 * cm,
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
        leftMargin=1.5*cm,
        rightMargin=1.5*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm
    )

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=24,
        alignment=TA_CENTER,
        textColor=NAVY,
        spaceAfter=10,
        fontName="Helvetica-Bold"
    )

    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=NAVY,
        spaceAfter=8,
        spaceBefore=12,
        fontName="Helvetica-Bold"
    )

    normal = ParagraphStyle(
        "Normal",
        parent=styles["BodyText"],
        fontSize=10,
        leading=16
    )

    small = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=8,
        textColor=GREY,
        alignment=TA_CENTER
    )

    story = []

    # =====================================================
    # TITLE
    # =====================================================

    story.append(
        Paragraph(
            "OCR Readiness Evaluation Report",
            title
        )
    )

    story.append(
        Paragraph(
            "AI Based OCR Image Quality Assessment Platform",
            small
        )
    )

    story.append(
        Paragraph(
            f"Generated : {datetime.now().strftime('%d %B %Y  %I:%M %p')}",
            small
        )
    )

    story.append(Spacer(1,10))

    story.append(
        HRFlowable(
            width="100%",
            color=TEAL,
            thickness=2
        )
    )

    story.append(Spacer(1,15))

    # =====================================================
    # IMAGE INFORMATION
    # =====================================================

    info_table = Table(
        [
            ["Image Name", image_name],
            ["OCR Readiness", f"{ocr_readiness:.1f}/100"],
            ["OCR Confidence",
             f"{ocr_confidence:.1f}%" if ocr_confidence is not None else "Not Available"],
            ["Status",
             factor_results.get("ocr_readiness_status","-")]
        ],
        colWidths=[5*cm,11*cm]
    )

    info_table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(0,-1),NAVY),

        ("TEXTCOLOR",(0,0),(0,-1),colors.white),

        ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),

        ("BACKGROUND",(1,0),(1,-1),LIGHT),

        ("GRID",(0,0),(-1,-1),0.5,colors.grey),

        ("BOTTOMPADDING",(0,0),(-1,-1),8),

        ("TOPPADDING",(0,0),(-1,-1),8)

    ]))

    story.append(info_table)

    story.append(Spacer(1,18))

    # =====================================================
    # UPLOADED IMAGE
    # =====================================================

    if image_path is not None:

        if os.path.exists(image_path):

            story.append(
                Paragraph(
                    "Uploaded Image",
                    heading
                )
            )

            img = Image(image_path)

            max_width = 15*cm
            max_height = 10*cm

            ratio = min(
                max_width/img.drawWidth,
                max_height/img.drawHeight
            )

            img.drawWidth *= ratio
            img.drawHeight *= ratio

            story.append(img)

            story.append(Spacer(1,15))

    # =====================================================
    # SCORE CARD
    # =====================================================

    story.append(
        Paragraph(
            "OCR Readiness Summary",
            heading
        )
    )

    score_colour = status_color(ocr_readiness)

    summary = Table(
        [[

            Paragraph(
                f"""
                <para align=center>

                <font size=32 color="{score_colour.hexval()}">

                <b>{ocr_readiness:.1f}</b>

                </font>

                <br/>

                <font size=13>

                {factor_results.get("ocr_readiness_status","")}

                </font>

                </para>
                """,
                normal
            ),

            Paragraph(
                f"""
                <b>Tesseract Confidence</b>

                <br/><br/>

                <font size=20>

                {ocr_confidence if ocr_confidence is not None else '--'}

                </font>
                """,
                normal
            )

        ]],
        colWidths=[8*cm,8*cm]
    )

    summary.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,-1),LIGHT),

        ("BOX",(0,0),(-1,-1),1,TEAL),

        ("GRID",(0,0),(-1,-1),0.5,colors.grey),

        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),

        ("BOTTOMPADDING",(0,0),(-1,-1),15),

        ("TOPPADDING",(0,0),(-1,-1),15)

    ]))

    story.append(summary)

    story.append(PageBreak())

    # =====================================================
    # FACTOR SCORES
    # =====================================================

    story.append(
        Paragraph(
            "OCR Quality Factor Scores",
            heading
        )
    )

    table_data = [
        ["Factor", "Score", "Status", "Weight"]
    ]

    for key, display in DISPLAY_NAMES.items():

        result = factor_results.get(key, {})

        score = float(result.get("score", 0))
        status = result.get("status", "-")
        weight = f"{WEIGHTS[key]*100:.0f}%"

        table_data.append([
            display,
            f"{score:.1f}",
            status,
            weight
        ])

    factor_table = Table(
        table_data,
        colWidths=[8*cm, 2.5*cm, 3.5*cm, 2*cm]
    )

    factor_style = TableStyle([

        ("BACKGROUND",(0,0),(-1,0),NAVY),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),

        ("BACKGROUND",(0,1),(-1,-1),LIGHT),

        ("GRID",(0,0),(-1,-1),0.4,colors.grey),

        ("ALIGN",(1,1),(-1,-1),"CENTER"),

        ("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),8),

    ])

    for row, key in enumerate(DISPLAY_NAMES.keys(), start=1):

        clr = status_color(
            factor_results.get(key, {}).get("score", 0)
        )

        factor_style.add(
            "TEXTCOLOR",
            (1,row),
            (2,row),
            clr
        )

        factor_style.add(
            "FONTNAME",
            (1,row),
            (2,row),
            "Helvetica-Bold"
        )

    factor_table.setStyle(factor_style)

    story.append(factor_table)

    story.append(Spacer(1,18))


    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    story.append(
        Paragraph(
            "Recommendations",
            heading
        )
    )

    if recommendations:

        for rec in recommendations:

            clean = (
                rec.replace("**","")
                   .replace("🔧","")
                   .replace("✅","")
            )

            story.append(
                Paragraph(
                    f"• {clean}",
                    normal
                )
            )

            story.append(Spacer(1,4))

    else:

        story.append(
            Paragraph(
                "No recommendations. The uploaded image already meets the recommended OCR quality standards.",
                normal
            )
        )

    story.append(PageBreak())

    # =====================================================
    # OCR EXTRACTED TEXT
    # =====================================================

    story.append(
        Paragraph(
            "OCR Extracted Text",
            heading
        )
    )

    if ocr_text.strip():

        # Prevent ReportLab HTML parsing issues
        safe_text = (
            ocr_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

        story.append(
            Paragraph(
                safe_text,
                normal
            )
        )

    else:

        story.append(
            Paragraph(
                "No OCR text was extracted.",
                normal
            )
        )

    story.append(Spacer(1,20))


    # =====================================================
    # FOOTER
    # =====================================================

    story.append(
        HRFlowable(
            width="100%",
            color=TEAL,
            thickness=1
        )
    )

    story.append(Spacer(1,8))

    footer = ParagraphStyle(
        "Footer",
        parent=normal,
        alignment=TA_CENTER,
        fontSize=8,
        textColor=GREY
    )

    story.append(
        Paragraph(
            "OCR Readiness Evaluation Platform",
            footer
        )
    )

    story.append(
        Paragraph(
            "SNLP Department",
            footer
        )
    )

    story.append(
        Paragraph(
            "Team: Yash (Lead), Vivek, Mansi, Krish, Tanusha",
            footer
        )
    )


    # =====================================================
    # BUILD PDF
    # =====================================================

    doc.build(
        story,
        onFirstPage=page_number,
        onLaterPages=page_number
    )

    pdf = buffer.getvalue()

    buffer.close()

    return pdf
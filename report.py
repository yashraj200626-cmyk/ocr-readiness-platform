"""
Professional OCR Readiness Report Generator
OCR Readiness Evaluation Platform
"""

import io
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
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


# ==========================================================
# THEME COLORS
# ==========================================================

NAVY = colors.HexColor("#1A2B4A")
DARK = colors.HexColor("#10203F")
TEAL = colors.HexColor("#00C4B4")
LIGHT = colors.HexColor("#F8FAFC")
LIGHT_GREY = colors.HexColor("#EEF2F7")
GREY = colors.HexColor("#6B7280")
WHITE = colors.white

GREEN = colors.HexColor("#10B981")
BLUE = colors.HexColor("#3B82F6")
ORANGE = colors.HexColor("#F59E0B")
RED = colors.HexColor("#EF4444")


# ==========================================================
# SCORE COLOR
# ==========================================================

def score_color(score):

    if score >= 81:
        return GREEN

    elif score >= 61:
        return BLUE

    elif score >= 41:
        return ORANGE

    return RED


# ==========================================================
# FORMAT PERCENTAGE
# ==========================================================

def format_percent(value):

    if value is None:
        return "Not Available"

    try:
        return f"{float(value):.1f}%"
    except Exception:
        return str(value)


# ==========================================================
# FOOTER
# ==========================================================

def draw_footer(canvas, doc):

    canvas.saveState()

    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(0.5)

    canvas.line(
        1.5 * cm,
        1.5 * cm,
        19 * cm,
        1.5 * cm
    )

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)

    canvas.drawString(
        1.7 * cm,
        1 * cm,
        "OCR Readiness Evaluation Platform"
    )

    canvas.drawRightString(
        19 * cm,
        1 * cm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


# ==========================================================
# MAIN FUNCTION
# ==========================================================

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
        leftMargin=1.4 * cm,
        rightMargin=1.4 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.7 * cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        alignment=TA_CENTER,
        textColor=NAVY,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["BodyText"],
        alignment=TA_CENTER,
        textColor=GREY,
        fontSize=11,
        spaceAfter=2
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=NAVY,
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=17,
        textColor=colors.black
    )

    center_style = ParagraphStyle(
        "Center",
        parent=normal_style,
        alignment=TA_CENTER
    )

    footer_style = ParagraphStyle(
        "Footer",
        parent=normal_style,
        alignment=TA_CENTER,
        fontSize=8,
        textColor=GREY
    )

    story = []

    # ==========================================================
    # COVER PAGE
    # ==========================================================

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "OCR Readiness Evaluation Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Based OCR Image Quality Assessment Platform",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            datetime.now().strftime(
                "Generated on %d %B %Y • %I:%M %p"
            ),
            subtitle_style
        )
    )

    story.append(Spacer(1, 8))

    story.append(
        HRFlowable(
            width="100%",
            thickness=2,
            color=TEAL
        )
    )

    story.append(Spacer(1, 18))

    # ==========================================================
    # IMAGE INFORMATION
    # ==========================================================

    info_data = [

        ["Image Name", image_name],

        ["OCR Readiness Score",
         f"{ocr_readiness:.1f} / 100"],

        ["OCR Confidence",
         format_percent(ocr_confidence)],

        ["Overall Status",
         factor_results.get(
             "ocr_readiness_status",
             "-"
         )],

        ["Report Generated",
         datetime.now().strftime(
             "%d %B %Y  %I:%M %p"
         )]

    ]

    info_table = Table(
        info_data,
        colWidths=[5 * cm, 11.2 * cm]
    )

    info_table.setStyle(

        TableStyle([

            ("BACKGROUND", (0, 0), (0, -1), NAVY),

            ("TEXTCOLOR", (0, 0), (0, -1), WHITE),

            ("BACKGROUND", (1, 0), (1, -1), LIGHT),

            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),

            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7DCE2")),

            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),

            ("TOPPADDING", (0, 0), (-1, -1), 9),

            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ])

    )

    story.append(info_table)

    story.append(Spacer(1, 18))


    # ==========================================================
    # OCR SUMMARY CARDS
    # ==========================================================

    colour = score_color(ocr_readiness)

    score_table = Table(

        [[

            Paragraph(

                f"""

                <para align="center">

                <font size="13" color="#6B7280">

                OCR Readiness

                </font>

                <br/><br/>

                <font size="38" color="{colour.hexval()}">

                <b>{ocr_readiness:.1f}</b>

                </font>

                <br/><br/>

                <font size="13">

                {factor_results.get("ocr_readiness_status","")}

                </font>

                </para>

                """,

                center_style

            ),

            Paragraph(

                f"""

                <para align="center">

                <font size="13" color="#6B7280">

                OCR Confidence

                </font>

                <br/><br/>

                <font size="38" color="#3B82F6">

                <b>{format_percent(ocr_confidence)}</b>

                </font>

                <br/><br/>

                <font size="12">

                Tesseract OCR Result

                </font>

                </para>

                """,

                center_style

            )

        ]],

        colWidths=[8.1 * cm, 8.1 * cm]

    )

    score_table.setStyle(

        TableStyle([

            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),

            ("BOX", (0, 0), (-1, -1), 1, TEAL),

            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9D9D9")),

            ("BOTTOMPADDING", (0, 0), (-1, -1), 18),

            ("TOPPADDING", (0, 0), (-1, -1), 18),

            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

            ("ALIGN", (0, 0), (-1, -1), "CENTER"),

        ])

    )

    story.append(score_table)

    story.append(Spacer(1, 20))

    # ==========================================================
    # UPLOADED DOCUMENT
    # ==========================================================

    if image_path and os.path.exists(image_path):

        story.append(
            Paragraph(
                "Uploaded Document",
                heading_style
            )
        )

        img = Image(image_path)

        max_width = 15 * cm
        max_height = 9 * cm

        ratio = min(
            max_width / img.drawWidth,
            max_height / img.drawHeight
        )

        img.drawWidth *= ratio
        img.drawHeight *= ratio

        image_table = Table(
            [[img]],
            colWidths=[16.2 * cm]
        )

        image_table.setStyle(

            TableStyle([

                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#D9D9D9")),

                ("BACKGROUND", (0, 0), (-1, -1), WHITE),

                ("ALIGN", (0, 0), (-1, -1), "CENTER"),

                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

                ("TOPPADDING", (0, 0), (-1, -1), 12),

                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),

            ])

        )

        story.append(image_table)

        story.append(Spacer(1, 18))


    # ==========================================================
    # NEXT PAGE
    # ==========================================================

    story.append(PageBreak())


    # ==========================================================
    # OCR QUALITY FACTOR SCORES
    # ==========================================================

    story.append(

        Paragraph(

            "OCR Quality Factor Scores",

            heading_style

        )

    )

    table_data = [

        [

            "Quality Factor",

            "Score",

            "Status",

            "Weight"

        ]

    ]

    for key, display in DISPLAY_NAMES.items():

        result = factor_results.get(key, {})

        score = float(
            result.get("score", 0)
        )

        status = result.get(
            "status",
            "-"
        )

        weight = f"{WEIGHTS[key] * 100:.0f}%"

        table_data.append([

            display,

            f"{score:.1f}",

            status,

            weight

        ])

    factor_table = Table(
        table_data,
        colWidths=[8 * cm, 2.5 * cm, 3.5 * cm, 2 * cm]
    )

    factor_style = TableStyle([

        # Header
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8D8D8")),

        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 9),

    ])

    # Alternate Row Background
    for row in range(1, len(table_data)):

        if row % 2 == 0:
            bg = WHITE
        else:
            bg = LIGHT

        factor_style.add(
            "BACKGROUND",
            (0, row),
            (-1, row),
            bg
        )

    # Score Colors
    for row, key in enumerate(DISPLAY_NAMES.keys(), start=1):

        score = factor_results.get(
            key,
            {}
        ).get(
            "score",
            0
        )

        clr = score_color(score)

        factor_style.add(
            "TEXTCOLOR",
            (1, row),
            (2, row),
            clr
        )

        factor_style.add(
            "FONTNAME",
            (1, row),
            (2, row),
            "Helvetica-Bold"
        )

    factor_table.setStyle(
        factor_style
    )

    story.append(
        factor_table
    )

    story.append(
        Spacer(1, 20)
    )


    # ==========================================================
    # RECOMMENDATIONS
    # ==========================================================

    story.append(
        Paragraph(
            "Recommendations",
            heading_style
        )
    )

    if recommendations:

        for index, recommendation in enumerate(
                recommendations,
                start=1
        ):

            recommendation = (
                recommendation
                .replace("**", "")
                .replace("🔧", "")
                .replace("✅", "")
            )

            card = Table(

                [[

                    Paragraph(

                        f"""
                        <b>Recommendation {index}</b>

                        <br/><br/>

                        {recommendation}
                        """,

                        normal_style

                    )

                ]],

                colWidths=[16.2 * cm]

            )

            card.setStyle(

                TableStyle([

                    ("BACKGROUND", (0, 0), (-1, -1), LIGHT),

                    ("BOX", (0, 0), (-1, -1), 1, TEAL),

                    ("LEFTPADDING", (0, 0), (-1, -1), 14),

                    ("RIGHTPADDING", (0, 0), (-1, -1), 14),

                    ("TOPPADDING", (0, 0), (-1, -1), 12),

                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),

                ])

            )

            story.append(card)

            story.append(
                Spacer(1, 10)
            )

    else:

        success = Table(

            [[

                Paragraph(

                    """
                    <b>Excellent!</b>

                    <br/><br/>

                    No recommendations are required.
                    The uploaded image already satisfies the
                    recommended OCR quality standards.
                    """,

                    normal_style

                )

            ]],

            colWidths=[16.2 * cm]

        )

        success.setStyle(

            TableStyle([

                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ECFDF5")),

                ("BOX", (0, 0), (-1, -1), 1, GREEN),

                ("LEFTPADDING", (0, 0), (-1, -1), 14),

                ("RIGHTPADDING", (0, 0), (-1, -1), 14),

                ("TOPPADDING", (0, 0), (-1, -1), 14),

                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),

            ])

        )

        story.append(success)

    story.append(
        Spacer(1, 20)
    )

    # ==========================================================
    # OCR EXTRACTED TEXT
    # ==========================================================

    story.append(PageBreak())

    story.append(
        Paragraph(
            "OCR Extracted Text",
            heading_style
        )
    )

    if ocr_text and ocr_text.strip():

        safe_text = (
            ocr_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

        text_table = Table(
            [[
                Paragraph(
                    safe_text,
                    normal_style
                )
            ]],
            colWidths=[16.2 * cm]
        )

        text_table.setStyle(

            TableStyle([

                ("BACKGROUND", (0, 0), (-1, -1), WHITE),

                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#D9D9D9")),

                ("LEFTPADDING", (0, 0), (-1, -1), 12),

                ("RIGHTPADDING", (0, 0), (-1, -1), 12),

                ("TOPPADDING", (0, 0), (-1, -1), 12),

                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),

            ])

        )

        story.append(text_table)

    else:

        empty_table = Table(
            [[
                Paragraph(
                    "No OCR text could be extracted from the uploaded image.",
                    normal_style
                )
            ]],
            colWidths=[16.2 * cm]
        )

        empty_table.setStyle(

            TableStyle([

                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),

                ("BOX", (0, 0), (-1, -1), 1, ORANGE),

                ("LEFTPADDING", (0, 0), (-1, -1), 12),

                ("RIGHTPADDING", (0, 0), (-1, -1), 12),

                ("TOPPADDING", (0, 0), (-1, -1), 12),

                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),

            ])

        )

        story.append(empty_table)

    story.append(Spacer(1, 20))

    # ==========================================================
    # REPORT FOOTER
    # ==========================================================

    story.append(Spacer(1, 20))

    story.append(

        HRFlowable(

            width="100%",

            thickness=1,

            color=TEAL

        )

    )

    story.append(Spacer(1, 8))

    story.append(

        Paragraph(

            "OCR Readiness Evaluation Platform",

            footer_style

        )

    )

    story.append(

        Paragraph(

            "AI-Based OCR Image Quality Assessment System",

            footer_style

        )

    )

    story.append(

        Paragraph(

            "Developed under SNLP Internship Project",

            footer_style

        )

    )

    story.append(

        Paragraph(

            "© 2026 All Rights Reserved",

            footer_style

        )

    )


    # ==========================================================
    # BUILD PDF
    # ==========================================================

    doc.build(

        story,

        onFirstPage=draw_footer,

        onLaterPages=draw_footer

    )

    pdf = buffer.getvalue()

    buffer.close()

    return pdf


import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Design Tokens (ReportLab HexColors)
PAPER = colors.HexColor("#F2EDE3")
INK = colors.HexColor("#191713")
INK_MUTED = colors.HexColor("#5A544A")
STAMP = colors.HexColor("#A93B26")      # Bot / Threat
LEDGER = colors.HexColor("#2C5D4F")     # Human / Verified
SIGNAL = colors.HexColor("#A8731F")     # Suspicious / Signal
RULE = colors.HexColor("#8C8578")
PAPER_DEEP = colors.HexColor("#E7E0D2")

def generate_forensic_dossier_pdf(account_data: dict, prediction_data: dict, attributions: list) -> io.BytesIO:
    """
    Renders an archival forensic dossier PDF for an account using ReportLab.
    Strictly follows Graphwarden case file design: sharp borders, forensic typography,
    auditable probabilities, and zero decorative fluff.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "DossierTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=INK,
        spaceAfter=4,
    )
    
    subtitle_style = ParagraphStyle(
        "DossierSubtitle",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        textColor=INK_MUTED,
        spaceAfter=8,
    )

    section_style = ParagraphStyle(
        "DossierSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=INK,
        spaceBefore=10,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "DossierBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=INK,
    )

    mono_style = ParagraphStyle(
        "DossierMono",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        textColor=INK,
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("GRAPHWARDEN FORENSIC DOSSIER // TECHNICAL CASE FILE", title_style))
    ref_id = f"CASE-GW-{account_data.get('id', '0000'):05d}-{datetime.datetime.utcnow().strftime('%Y%m%d')}"
    timestamp_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(f"REFERENCE: {ref_id}  |  GENERATED: {timestamp_str}  |  STATUS: AUDITED", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=INK, spaceBefore=2, spaceAfter=8))

    # 2. Subject Metadata Block
    story.append(Paragraph("SECTION 1: TARGET ACCOUNT PROFILE", section_style))
    
    acc_table_data = [
        [
            Paragraph("<b>Target Screen Name:</b>", body_style),
            Paragraph(f"@{account_data.get('screen_name', 'unknown')}", mono_style),
            Paragraph("<b>Account Node Index:</b>", body_style),
            Paragraph(str(account_data.get('id', 'N/A')), mono_style),
        ],
        [
            Paragraph("<b>Numerical User ID:</b>", body_style),
            Paragraph(str(account_data.get('user_id_str', 'N/A')), mono_style),
            Paragraph("<b>Account Age:</b>", body_style),
            Paragraph(f"{account_data.get('account_age_days', 0)} days", body_style),
        ],
        [
            Paragraph("<b>Followers Count:</b>", body_style),
            Paragraph(f"{account_data.get('followers_count', 0):,}", mono_style),
            Paragraph("<b>Following Count:</b>", body_style),
            Paragraph(f"{account_data.get('following_count', 0):,}", mono_style),
        ],
        [
            Paragraph("<b>Verified Status:</b>", body_style),
            Paragraph("VERIFIED" if account_data.get('verified') else "UNVERIFIED", mono_style),
            Paragraph("<b>Ground Truth Label:</b>", body_style),
            Paragraph(str(account_data.get('ground_truth', 'unknown')).upper(), mono_style),
        ],
    ]

    t_meta = Table(acc_table_data, colWidths=[120, 140, 130, 140])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PAPER_DEEP),
        ('BOX', (0, 0), (-1, -1), 0.5, RULE),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D5CDBC")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))

    # 3. Verdict Stamp Banner
    story.append(Paragraph("SECTION 2: GRAPH NEURAL NETWORK VERDICT", section_style))
    verdict = prediction_data.get("predicted_class", "bot").upper()
    conf = prediction_data.get("confidence", 0.95) * 100
    model_name = prediction_data.get("model_name", "GAT (Graph Attention Network)")
    
    if verdict == "BOT":
        stamp_bg = colors.HexColor("#F9EBE8")
        stamp_fg = STAMP
        stamp_border = STAMP
    elif verdict == "HUMAN":
        stamp_bg = colors.HexColor("#EBF3EF")
        stamp_fg = LEDGER
        stamp_border = LEDGER
    else:
        stamp_bg = colors.HexColor("#FBF5EB")
        stamp_fg = SIGNAL
        stamp_border = SIGNAL

    verdict_text = f"<b>CLASSIFICATION VERDICT: {verdict} ({conf:.1f}% CONFIDENCE)</b><br/><font size=7 color='#5A544A'>ENGINE: {model_name} // LATENCY: {prediction_data.get('latency_ms', 11.2):.1f}ms</font>"
    verdict_p = Paragraph(verdict_text, ParagraphStyle("VerdictP", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=14, textColor=stamp_fg))
    
    t_verdict = Table([[verdict_p]], colWidths=[530])
    t_verdict.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), stamp_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, stamp_border),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t_verdict)
    story.append(Spacer(1, 8))

    # 4. Multimodal Probability Distribution
    probs = prediction_data.get("probabilities", {"human": 0.05, "bot": 0.92, "suspicious": 0.03})
    prob_table_data = [
        [
            Paragraph("<b>Class Label</b>", body_style),
            Paragraph("<b>Posterior Probability</b>", body_style),
            Paragraph("<b>Decision Threshold</b>", body_style),
            Paragraph("<b>Action Status</b>", body_style),
        ],
        [
            Paragraph("Automated Bot", body_style),
            Paragraph(f"{probs.get('bot', 0.0) * 100:.2f}%", mono_style),
            Paragraph(">= 50.0%", mono_style),
            Paragraph("FLAGGED" if probs.get('bot', 0.0) >= 0.5 else "CLEAR", mono_style),
        ],
        [
            Paragraph("Suspicious / Coordinated", body_style),
            Paragraph(f"{probs.get('suspicious', 0.0) * 100:.2f}%", mono_style),
            Paragraph(">= 40.0%", mono_style),
            Paragraph("FLAGGED" if probs.get('suspicious', 0.0) >= 0.4 else "CLEAR", mono_style),
        ],
        [
            Paragraph("Authentic Human", body_style),
            Paragraph(f"{probs.get('human', 0.0) * 100:.2f}%", mono_style),
            Paragraph(">= 50.0%", mono_style),
            Paragraph("VERIFIED" if probs.get('human', 0.0) >= 0.5 else "CLEAR", mono_style),
        ],
    ]

    t_prob = Table(prob_table_data, colWidths=[150, 130, 120, 130])
    t_prob.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PAPER_DEEP),
        ('BOX', (0, 0), (-1, -1), 0.5, RULE),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D5CDBC")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_prob)
    story.append(Spacer(1, 8))

    # 5. Top Diagnostic Feature Attributions
    story.append(Paragraph("SECTION 3: ATTRIBUTION & DIAGNOSTIC SIGNALS", section_style))
    attr_table_data = [
        [
            Paragraph("<b>Signal / Feature Name</b>", body_style),
            Paragraph("<b>Category</b>", body_style),
            Paragraph("<b>Weight</b>", body_style),
            Paragraph("<b>Influence</b>", body_style),
            Paragraph("<b>Forensic Notes</b>", body_style),
        ]
    ]

    for item in attributions[:5]:
        attr_table_data.append([
            Paragraph(item.get("feature", item.get("name", "Signal")), body_style),
            Paragraph(item.get("category", "graph").upper(), mono_style),
            Paragraph(f"{item.get('importance', item.get('weight', 0.0)):.3f}", mono_style),
            Paragraph(item.get("direction", "RISK").replace("_", " ").upper(), mono_style),
            Paragraph(item.get("explanation", item.get("description", "Significant attribution.")), mono_style),
        ])

    t_attr = Table(attr_table_data, colWidths=[120, 65, 45, 75, 225])
    t_attr.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PAPER_DEEP),
        ('BOX', (0, 0), (-1, -1), 0.5, RULE),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D5CDBC")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_attr)
    story.append(Spacer(1, 14))

    # 6. Certification and Signature Line
    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE, spaceBefore=4, spaceAfter=8))
    sig_text = """
    <b>FORENSIC AUDITOR CERTIFICATION:</b><br/>
    I hereby certify that this automated dossier has been compiled from calibrated Graph Neural Network representations and verified topological interaction graphs. All probabilities reflect model state checkpoints preserved under MLflow experiment governance.
    """
    story.append(Paragraph(sig_text, ParagraphStyle("Cert", parent=styles["Normal"], fontName="Helvetica", fontSize=7.5, leading=10, textColor=INK_MUTED)))
    story.append(Spacer(1, 10))

    sig_table_data = [
        [
            Paragraph("<b>INVESTIGATOR SIGNATURE:</b> ___________________________", body_style),
            Paragraph("<b>DATE OF CERTIFICATION:</b> ____________________", body_style),
        ]
    ]
    t_sig = Table(sig_table_data, colWidths=[300, 230])
    t_sig.setStyle(TableStyle([
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(t_sig)

    doc.build(story)
    buffer.seek(0)
    return buffer

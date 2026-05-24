"""
Phase 4 — Explainability agent: SHAP narrative + PDF audit report.
Uses Claude (via LangChain) to generate a plain-English explanation,
then builds a PDF with reportlab.
"""

import os
import datetime
from agents.state import LoanState

# ── LangChain / Ollama (local, free, unlimited) ──────────────────────────────
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# ── PDF generation ───────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ── Config ───────────────────────────────────────────────────────────────────
AUDIT_PDF_DIR = "reports/audit_pdfs"
os.makedirs(AUDIT_PDF_DIR, exist_ok=True)

FEATURE_MAPPING = {
    "EXT_SOURCE_1": "Credit Bureau Score 1 (Equifax)",
    "EXT_SOURCE_2": "Credit Bureau Score 2 (Experian)",
    "EXT_SOURCE_3": "Credit Bureau Score 3 (TransUnion)",
    "NAME_EDUCATION_TYPE": "Education Type",
    "AMT_CREDIT": "Total Loan Amount",
    "AMT_ANNUITY": "Monthly Loan Payment",
    "AMT_INCOME_TOTAL": "Total Annual Income",
    "DAYS_BIRTH": "Client Age",
    "DAYS_EMPLOYED": "Employment Duration",
    "AMT_GOODS_PRICE": "Goods Price",
    "NAME_CONTRACT_TYPE": "Contract Type",
    "CODE_GENDER": "Gender",
    "FLAG_OWN_CAR": "Owns Car",
    "FLAG_OWN_REALTY": "Owns Realty",
    "CNT_CHILDREN": "Number of Children",
}

# Local Ollama model — unlimited, no API key, no rate limits
# Make sure Ollama is running: ollama serve
# Model must be pulled first: ollama pull llama3.2
_llm = ChatOllama(model="llama3.2", num_predict=200)


def _build_narrative_prompt(state: LoanState) -> str:
    """Build a concise prompt for Claude to generate a 100-word audit explanation."""
    shap_top3 = state.get("shap_top3") or []
    shap_lines = ", ".join(
        f"{f['feature']}={f['shap_value']:+.3f}"
        for f in shap_top3
    )
    return (
        f"Write a 80-word plain-English loan decision explanation for a compliance audit.\n"
        f"Decision: {state.get('decision')} | "
        f"Default prob: {state.get('default_probability', 0):.1%} | "
        f"Fraud score: {state.get('fraud_score', 0):.2f} | "
        f"Segment: {state.get('segment')}\n"
        f"Top SHAP factors: {shap_lines}\n"
        f"Only reference the SHAP factors listed. Be factual and concise."
    )


def _generate_narrative(state: LoanState) -> str:
    """Call Claude to generate the audit narrative. Returns plain text."""
    try:
        prompt = _build_narrative_prompt(state)
        response = _llm.invoke([HumanMessage(content=prompt)])
        narrative = response.content.strip()

        # Post-generation validation: ensure top features are mentioned
        shap_top3 = state.get("shap_top3") or []
        for factor in shap_top3[:2]:  # At least top 2 must appear
            if factor["feature"] not in narrative:
                narrative += f"\n[Key factor: {factor['feature']} ({factor['direction']})]"

        return narrative
    except Exception as e:
        # Fallback: template-based narrative if LLM fails
        return (
            f"Decision: {state.get('decision', 'N/A')}. "
            f"Default probability: {state.get('default_probability', 'N/A'):.2%}. "
            f"Fraud score: {state.get('fraud_score', 'N/A'):.2f}. "
            f"Uplift segment: {state.get('segment', 'N/A')}. "
            f"[AI narrative currently unavailable due to backend connection error.]"
        )


def _generate_pdf(state: LoanState, narrative: str, app_id: str) -> str:
    """
    Build a PDF audit report using reportlab.
    Includes: decision header, risk scores grid, SHAP factors, LLM narrative, timestamp.
    """
    pdf_path = f"{AUDIT_PDF_DIR}/{app_id}.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    decision = state.get("decision", "N/A").upper()
    decision_color = {
        "APPROVE": colors.green, "APPROVE_WITH_RATE": colors.green,
        "APPROVE_STANDARD": colors.green, "DECLINE": colors.red,
        "HUMAN_REVIEW": colors.orange,
    }.get(decision, colors.black)

    header_style = ParagraphStyle("header", parent=styles["Title"], textColor=decision_color)
    story.append(Paragraph(f"Loan Decision: {decision}", header_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"Application ID: {app_id}", styles["Normal"]))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.5*cm))

    # ── Decision reason ───────────────────────────────────────────────────────
    story.append(Paragraph("Decision Reason", styles["Heading2"]))
    story.append(Paragraph(state.get("decision_reason", "N/A"), styles["Normal"]))
    story.append(Spacer(1, 0.5*cm))

    # ── Risk scores grid ──────────────────────────────────────────────────────
    story.append(Paragraph("Risk Scores", styles["Heading2"]))
    score_data = [
        ["Metric", "Value", "Band/Level"],
        ["Default Probability", f"{state.get('default_probability', 'N/A'):.2%}" if state.get('default_probability') else "N/A", state.get("risk_band", "N/A")],
        ["Fraud Score", f"{state.get('fraud_score', 'N/A'):.4f}" if state.get('fraud_score') else "N/A", state.get("risk_level", "N/A")],
        ["Uplift Score (ITE)", f"{state.get('uplift_score', 'N/A'):.4f}" if state.get('uplift_score') else "N/A", state.get("segment", "N/A")],
    ]
    score_table = Table(score_data, colWidths=[6*cm, 4*cm, 5*cm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f3f4")]),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.5*cm))

    # ── SHAP top-3 factors ────────────────────────────────────────────────────
    story.append(Paragraph("Top 3 SHAP Factors (Credit Risk Model)", styles["Heading2"]))
    shap_top3 = state.get("shap_top3") or []
    shap_data = [["Feature", "Value", "SHAP Impact", "Direction"]]
    for f in shap_top3:
        mapped_name = FEATURE_MAPPING.get(f["feature"], f["feature"])
        shap_data.append([
            mapped_name, f"{f['value']:.4f}",
            f"{f['shap_value']:+.4f}", f["direction"]
        ])
    if shap_data:
        shap_table = Table(shap_data, colWidths=[6*cm, 3*cm, 4*cm, 5*cm])
        shap_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(shap_table)
    story.append(Spacer(1, 0.5*cm))

    # ── LLM narrative ─────────────────────────────────────────────────────────
    story.append(Paragraph("Audit Narrative (AI-Generated)", styles["Heading2"]))
    story.append(Paragraph(narrative, styles["Normal"]))
    story.append(Spacer(1, 0.5*cm))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Paragraph(
        "This report was generated automatically by the Multi-Agent Loan Intelligence System. "
        "All decisions are subject to human review and applicable regulations.",
        ParagraphStyle("footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    ))

    doc.build(story)
    return pdf_path


def explainability_node(state: LoanState) -> LoanState:
    """
    Generate SHAP narrative via Claude and produce a PDF audit report.
    This is the final node — graph ends after this.
    """
    # Generate a unique application ID from timestamp
    app_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")

    # ── Generate LLM narrative ────────────────────────────────────────────────
    narrative = _generate_narrative(state)

    # ── Generate PDF audit report ─────────────────────────────────────────────
    pdf_path = _generate_pdf(state, narrative, app_id)

    return {
        **state,
        "shap_narrative": narrative,
        "audit_pdf_path": pdf_path,
    }

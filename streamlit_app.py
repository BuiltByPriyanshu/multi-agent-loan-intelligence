"""
Multi-Agent Loan Evaluation System — Advanced Streamlit UI
Indian Banking Context | BML Munjal University
Powered by LangGraph · LightGBM · T-Learner · SHAP · Gemini AI
"""

import httpx
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

import os

os.makedirs(".streamlit", exist_ok=True)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    new_theme = "light" if st.session_state.theme == "dark" else "dark"
    st.session_state.theme = new_theme
    with open(".streamlit/config.toml", "w") as f:
        f.write(f'[theme]\nbase="{new_theme}"\n')

st.set_page_config(
    page_title="Multi-Agent Loan Evaluation System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
dark_css = """
  /* ── Dark navy background ── */
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e8eaf0;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%);
    border-right: 1px solid #1e3a5f;
  }
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: #c9a84c;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin-top: 18px;
  }
  .stNumberInput input, .stSelectbox select, .stSlider {
    background-color: #0f2035 !important;
    color: #e8eaf0 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
  }
  label[data-testid="stWidgetLabel"] p {
    color: #a8b8cc !important;
    font-size: 0.82rem !important;
  }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #c9a84c 0%, #e8c96a 100%) !important;
    color: #0a0e1a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(201,168,76,0.3) !important;
    transition: all 0.3s ease !important;
  }
  .stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(201,168,76,0.5) !important;
  }
  .agent-card {
    background: linear-gradient(135deg, #0f2035 0%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 24px;
    margin: 8px 0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .agent-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(201,168,76,0.15);
    border-color: #c9a84c;
  }
  .agent-card h4 {
    color: #c9a84c;
    margin: 0 0 8px 0;
    font-size: 1.05rem;
    font-weight: 600;
  }
  .agent-card p {
    color: #8a9bb0;
    font-size: 0.88rem;
    margin: 0;
    line-height: 1.5;
  }
  .agent-card .agent-icon {
    font-size: 2rem;
    margin-bottom: 12px;
    display: block;
  }
  .decision-sanctioned {
    background: linear-gradient(135deg, #0d3320 0%, #0a2a1a 100%);
    border: 2px solid #2ecc71;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
  }
  .decision-special-rate {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 100%);
    border: 2px solid #3498db;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
  }
  .decision-committee {
    background: linear-gradient(135deg, #2d2000 0%, #1a1400 100%);
    border: 2px solid #f39c12;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
  }
  .decision-rejected {
    background: linear-gradient(135deg, #2d0a0a 0%, #1a0606 100%);
    border: 2px solid #e74c3c;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
  }
  .decision-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin: 0;
  }
  .decision-subtitle {
    font-size: 0.9rem;
    opacity: 0.75;
    margin-top: 6px;
  }
  .reason-box {
    background: #0f2035;
    border-left: 4px solid #c9a84c;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 16px 0;
    color: #c8d8e8;
    font-size: 0.92rem;
    line-height: 1.6;
  }
  .metric-card {
    background: linear-gradient(135deg, #0f2035 0%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
  }
  .metric-card .metric-label {
    color: #8a9bb0;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }
  .metric-card .metric-value {
    color: #c9a84c;
    font-size: 1.6rem;
    font-weight: 700;
  }
  .metric-card .metric-sub {
    color: #5a7a9a;
    font-size: 0.8rem;
    margin-top: 4px;
  }
  .shap-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }
  .shap-table th {
    background: #0f2035;
    color: #c9a84c;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #1e3a5f;
  }
  .shap-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #1a2f45;
    color: #c8d8e8;
  }
  .shap-table tr:hover td {
    background: #0f2035;
  }
  .shap-positive { color: #e74c3c; font-weight: 600; }
  .shap-negative { color: #2ecc71; font-weight: 600; }
  .narrative-card {
    background: linear-gradient(135deg, #0f2035 0%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    color: #c8d8e8;
    font-size: 0.9rem;
    line-height: 1.7;
  }
  .narrative-card h4 {
    color: #c9a84c;
    margin: 0 0 12px 0;
    font-size: 1rem;
    font-weight: 600;
  }
  .pipeline-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 20px 0;
    flex-wrap: wrap;
  }
  .pipeline-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: #0f2035;
    border: 1px solid #2ecc71;
    border-radius: 10px;
    padding: 12px 16px;
    min-width: 110px;
    text-align: center;
  }
  .pipeline-step .step-icon { font-size: 1.4rem; }
  .pipeline-step .step-name {
    color: #c8d8e8;
    font-size: 0.72rem;
    margin-top: 6px;
    font-weight: 500;
  }
  .pipeline-step .step-check {
    color: #2ecc71;
    font-size: 0.85rem;
    margin-top: 4px;
  }
  .pipeline-arrow {
    color: #c9a84c;
    font-size: 1.4rem;
    padding: 0 6px;
  }
  .section-header {
    color: #c9a84c;
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
  }
  .status-badge-online {
    display: inline-block;
    background: #0d3320;
    border: 1px solid #2ecc71;
    color: #2ecc71;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
  }
  .status-badge-offline {
    display: inline-block;
    background: #2d0a0a;
    border: 1px solid #e74c3c;
    color: #e74c3c;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
  }
  .metrics-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }
  .metrics-table th {
    background: #0f2035;
    color: #c9a84c;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #1e3a5f;
  }
  .metrics-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #1a2f45;
    color: #c8d8e8;
  }
  .metrics-table tr:nth-child(even) td { background: #0a1628; }
  .footer {
    text-align: center;
    color: #3a5a7a;
    font-size: 0.78rem;
    padding: 24px 0 12px 0;
    border-top: 1px solid #1e3a5f;
    margin-top: 40px;
  }
  .stDownloadButton > button {
    background: linear-gradient(135deg, #0f2035 0%, #0d1b2a 100%) !important;
    color: #c9a84c !important;
    border: 1px solid #c9a84c !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
  }
  .stDownloadButton > button:hover {
    background: linear-gradient(135deg, #c9a84c 0%, #e8c96a 100%) !important;
    color: #0a0e1a !important;
  }
  .streamlit-expanderHeader {
    background: #0f2035 !important;
    color: #c9a84c !important;
    border-radius: 8px !important;
  }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }
"""

light_css = """
  /* ── Light background ── */
  .stApp {
    background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 50%, #f0f4f8 100%);
    color: #1e293b;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e2e8f0 0%, #cbd5e1 100%);
    border-right: 1px solid #94a3b8;
  }
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: #b45309;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    border-bottom: 1px solid #94a3b8;
    padding-bottom: 6px;
    margin-top: 18px;
  }
  .stNumberInput input, .stSelectbox select, .stSlider {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #94a3b8 !important;
    border-radius: 6px !important;
  }
  label[data-testid="stWidgetLabel"] p {
    color: #475569 !important;
    font-size: 0.82rem !important;
  }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(245,158,11,0.3) !important;
    transition: all 0.3s ease !important;
  }
  .stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(245,158,11,0.5) !important;
  }
  .agent-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #94a3b8;
    border-radius: 12px;
    padding: 24px;
    margin: 8px 0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .agent-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(245,158,11,0.15);
    border-color: #f59e0b;
  }
  .agent-card h4 {
    color: #b45309;
    margin: 0 0 8px 0;
    font-size: 1.05rem;
    font-weight: 600;
  }
  .agent-card p {
    color: #475569;
    font-size: 0.88rem;
    margin: 0;
    line-height: 1.5;
  }
  .agent-card .agent-icon {
    font-size: 2rem;
    margin-bottom: 12px;
    display: block;
  }
  .decision-sanctioned {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border: 2px solid #22c55e;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
  }
  .decision-special-rate {
    background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
    border: 2px solid #0ea5e9;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
  }
  .decision-committee {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 2px solid #f59e0b;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
  }
  .decision-rejected {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
  }
  .decision-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin: 0;
    color: #1e293b;
  }
  .decision-subtitle {
    font-size: 0.9rem;
    opacity: 0.75;
    margin-top: 6px;
    color: #1e293b;
  }
  .reason-box {
    background: #ffffff;
    border-left: 4px solid #b45309;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 16px 0;
    color: #1e293b;
    font-size: 0.92rem;
    line-height: 1.6;
  }
  .metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #94a3b8;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
  }
  .metric-card .metric-label {
    color: #475569;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }
  .metric-card .metric-value {
    color: #b45309;
    font-size: 1.6rem;
    font-weight: 700;
  }
  .metric-card .metric-sub {
    color: #64748b;
    font-size: 0.8rem;
    margin-top: 4px;
  }
  .shap-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }
  .shap-table th {
    background: #e2e8f0;
    color: #b45309;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #94a3b8;
  }
  .shap-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #cbd5e1;
    color: #1e293b;
  }
  .shap-table tr:hover td {
    background: #f1f5f9;
  }
  .shap-positive { color: #ef4444; font-weight: 600; }
  .shap-negative { color: #22c55e; font-weight: 600; }
  .narrative-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #94a3b8;
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    color: #1e293b;
    font-size: 0.9rem;
    line-height: 1.7;
  }
  .narrative-card h4 {
    color: #b45309;
    margin: 0 0 12px 0;
    font-size: 1rem;
    font-weight: 600;
  }
  .pipeline-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 20px 0;
    flex-wrap: wrap;
  }
  .pipeline-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: #ffffff;
    border: 1px solid #22c55e;
    border-radius: 10px;
    padding: 12px 16px;
    min-width: 110px;
    text-align: center;
  }
  .pipeline-step .step-icon { font-size: 1.4rem; }
  .pipeline-step .step-name {
    color: #1e293b;
    font-size: 0.72rem;
    margin-top: 6px;
    font-weight: 500;
  }
  .pipeline-step .step-check {
    color: #22c55e;
    font-size: 0.85rem;
    margin-top: 4px;
  }
  .pipeline-arrow {
    color: #b45309;
    font-size: 1.4rem;
    padding: 0 6px;
  }
  .section-header {
    color: #b45309;
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    border-bottom: 1px solid #94a3b8;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
  }
  .status-badge-online {
    display: inline-block;
    background: #dcfce7;
    border: 1px solid #22c55e;
    color: #15803d;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
  }
  .status-badge-offline {
    display: inline-block;
    background: #fee2e2;
    border: 1px solid #ef4444;
    color: #b91c1c;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
  }
  .metrics-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }
  .metrics-table th {
    background: #e2e8f0;
    color: #b45309;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #94a3b8;
  }
  .metrics-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #cbd5e1;
    color: #1e293b;
  }
  .metrics-table tr:nth-child(even) td { background: #f8fafc; }
  .footer {
    text-align: center;
    color: #64748b;
    font-size: 0.78rem;
    padding: 24px 0 12px 0;
    border-top: 1px solid #94a3b8;
    margin-top: 40px;
  }
  .stDownloadButton > button {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
    color: #b45309 !important;
    border: 1px solid #b45309 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
  }
  .stDownloadButton > button:hover {
    background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%) !important;
    color: #ffffff !important;
  }
  .streamlit-expanderHeader {
    background: #ffffff !important;
    color: #b45309 !important;
    border-radius: 8px !important;
  }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }
"""

theme_css = light_css if st.session_state.theme == "light" else dark_css

st.markdown(f"""
<style>
  /* ── Base & fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
  }}

  {theme_css}
</style>
""", unsafe_allow_html=True)

# ── Constants & Mappings ──────────────────────────────────────────────────────
DECISION_MAP = {
    "approve_with_rate":  ("SANCTIONED WITH SPECIAL RATE", "decision-special-rate", "#3498db", "🔵"),
    "approve_standard":   ("SANCTIONED",                   "decision-sanctioned",   "#2ecc71", "🟢"),
    "approve":            ("SANCTIONED",                   "decision-sanctioned",   "#2ecc71", "🟢"),
    "decline":            ("REJECTED",                     "decision-rejected",     "#e74c3c", "🔴"),
    "human_review":       ("REFERRED TO CREDIT COMMITTEE", "decision-committee",    "#f39c12", "🟡"),
    "APPROVE_WITH_RATE":  ("SANCTIONED WITH SPECIAL RATE", "decision-special-rate", "#3498db", "🔵"),
    "APPROVE_STANDARD":   ("SANCTIONED",                   "decision-sanctioned",   "#2ecc71", "🟢"),
    "APPROVE":            ("SANCTIONED",                   "decision-sanctioned",   "#2ecc71", "🟢"),
    "DECLINE":            ("REJECTED",                     "decision-rejected",     "#e74c3c", "🔴"),
    "HUMAN_REVIEW":       ("REFERRED TO CREDIT COMMITTEE", "decision-committee",    "#f39c12", "🟡"),
}

SEGMENT_MAP = {
    "Persuadable":    "Rate Sensitive",
    "Sure Thing":     "Low Risk Borrower",
    "Lost Cause":     "High Risk Borrower",
    "Do Not Disturb": "Leave Unchanged",
    "persuadable":    "Rate Sensitive",
    "sure_thing":     "Low Risk Borrower",
    "lost_cause":     "High Risk Borrower",
    "do_not_disturb": "Leave Unchanged",
}

FEATURE_MAP = {
    "EXT_SOURCE_3":        "CIBIL TransUnion Score",
    "EXT_SOURCE_1":        "Equifax Credit Score",
    "EXT_SOURCE_2":        "Experian Credit Score",
    "NAME_EDUCATION_TYPE": "Education Qualification",
    "DAYS_BIRTH":          "Applicant Age",
    "AMT_CREDIT":          "Sanctioned Amount",
    "AMT_INCOME_TOTAL":    "Annual Income",
    "DAYS_EMPLOYED":       "Employment Tenure",
    "fico_range_low":      "FICO Score",
    "dti":                 "Debt-to-Income Ratio",
    "annual_inc":          "Annual Income",
    "loan_amnt":           "Loan Amount",
    "int_rate":            "Interest Rate",
    "emp_length_num":      "Employment Length",
    "AMT_ANNUITY":         "Monthly EMI",
}

# ── Helper Functions ──────────────────────────────────────────────────────────
def check_api_health():
    """Returns (is_online, models_loaded) tuple."""
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=3.0)
        data = resp.json()
        return True, data.get("models_loaded", False)
    except Exception:
        return False, False


def map_feature_name(raw: str) -> str:
    return FEATURE_MAP.get(raw, raw.replace("_", " ").title())


def map_segment(raw: str) -> str:
    return SEGMENT_MAP.get(raw, raw)


def fmt_currency(value: float) -> str:
    """Format a number as currency."""
    return f"${value:,.2f}"


def build_gauges(default_prob, fraud_score, rate_sensitivity, loan_amnt=0, annual_inc=0, fico_score=0):
    """Build a 3-gauge Plotly figure with descriptive labels."""

    # Ensure range matches [0, 100] properly with clipping
    dp_val = np.clip((default_prob or 0) * 100, 0, 100)
    fr_val = np.clip((fraud_score  or 0) * 100, 0, 100)
    rs_raw = rate_sensitivity or 0

    # Human-readable labels for what each gauge is showing
    def dp_label(v):
        if v < 20:   return "LOW RISK — Safe to Sanction"
        if v < 40:   return "LOW-MEDIUM — Monitor Closely"
        if v < 70:   return "MEDIUM-HIGH — Caution Advised"
        return       "HIGH RISK — Likely to Default"

    def fr_label(v):
        if v < 7.5:  return "CLEAN — No Fraud Signals"
        if v < 30:   return "LOW — Minor Anomalies"
        if v < 60:   return "MEDIUM — Needs Verification"
        return       "HIGH — Possible Fraud"

    def rs_label(raw):
        if raw >= 0.032:  return "RATE SENSITIVE — Offer Better Rate"
        if raw >= -0.023: return "NEUTRAL — Standard Processing"
        return            "LEAVE UNCHANGED — Rate Offer Unhelpful"

    gauge_axis = dict(tickcolor="#8a9bb0", tickfont=dict(color="#8a9bb0", size=9))

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
    )

    # ── Gauge 1: Default Risk ─────────────────────────────────────────────────
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=dp_val,
        title=dict(
            text=f"<b>Default Risk</b><br><span style='font-size:0.75em;color:#8a9bb0'>{dp_label(dp_val)}</span>",
            font=dict(color="#c9a84c", size=13),
        ),
        number=dict(suffix="%", font=dict(color="#e8eaf0", size=26), valueformat=".1f"),
        gauge=dict(
            axis=dict(**gauge_axis, range=[0, 100],
                      tickvals=[0, 20, 40, 70, 100],
                      ticktext=["0%", "20%\nLow", "40%\nMed", "70%\nHigh", "100%"]),
            bar=dict(color="#c9a84c", thickness=0.28),
            bgcolor="#0f2035",
            borderwidth=1, bordercolor="#1e3a5f",
            steps=[
                dict(range=[0,  20], color="#0d3320"),
                dict(range=[20, 40], color="#1a3010"),
                dict(range=[40, 70], color="#2d2000"),
                dict(range=[70,100], color="#2d0a0a"),
            ],
            threshold=dict(line=dict(color="#ff4444", width=3), thickness=0.8, value=70),
        ),
    ), row=1, col=1)

    # ── Gauge 2: Fraud Risk ───────────────────────────────────────────────────
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=fr_val,
        title=dict(
            text=f"<b>Fraud Risk</b><br><span style='font-size:0.75em;color:#8a9bb0'>{fr_label(fr_val)}</span>",
            font=dict(color="#c9a84c", size=13),
        ),
        number=dict(suffix="%", font=dict(color="#e8eaf0", size=26), valueformat=".2f"),
        gauge=dict(
            axis=dict(**gauge_axis, range=[0, 100],
                      tickvals=[0, 7.5, 30, 60, 85, 100],
                      ticktext=["0%", "7.5%\nClean", "30%", "60%", "85%\nDanger", "100%"]),
            bar=dict(color="#c9a84c", thickness=0.28),
            bgcolor="#0f2035",
            borderwidth=1, bordercolor="#1e3a5f",
            steps=[
                dict(range=[0,   7.5], color="#0d3320"),
                dict(range=[7.5, 30],  color="#1a2a10"),
                dict(range=[30,  60],  color="#2d2000"),
                dict(range=[60,  85],  color="#3d1500"),
                dict(range=[85, 100],  color="#2d0a0a"),
            ],
            threshold=dict(line=dict(color="#ff4444", width=3), thickness=0.8, value=85),
        ),
    ), row=1, col=2)

    # ── Gauge 3: Rate Sensitivity — value is actual ITE, gauge position is normalised ──
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=rs_raw,                          # show actual ITE number
        title=dict(
            text=f"<b>Rate Sensitivity</b><br><span style='font-size:0.75em;color:#8a9bb0'>{rs_label(rs_raw)}</span>",
            font=dict(color="#c9a84c", size=13),
        ),
        number=dict(
            font=dict(color="#e8eaf0", size=20),
            valueformat="+.4f",
            prefix="ITE: ",
        ),
        gauge=dict(
            axis=dict(**gauge_axis, range=[-0.3, 0.4],
                      tickvals=[-0.3, -0.023, 0, 0.032, 0.4],
                      ticktext=["-0.30", "Leave\nUnchanged", "0", "Rate\nSensitive", "+0.40"]),
            bar=dict(color="#c9a84c", thickness=0.28),
            bgcolor="#0f2035",
            borderwidth=1, bordercolor="#1e3a5f",
            steps=[
                dict(range=[-0.3,  -0.023], color="#1a0d2a"),
                dict(range=[-0.023, 0.032], color="#0d1f3c"),
                dict(range=[0.032,  0.4],   color="#0d3a4c"),
            ],
            threshold=dict(line=dict(color="#3498db", width=3), thickness=0.8, value=0.032),
        ),
    ), row=1, col=3)

    # ── Summary annotation below gauges ──────────────────────────────────────
    loan_fmt   = fmt_currency(loan_amnt)   if loan_amnt   else "—"
    income_fmt = fmt_currency(annual_inc)  if annual_inc  else "—"

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        font=dict(color="#e8eaf0", family="Inter"),
        height=320,
        margin=dict(l=30, r=30, t=20, b=60),
        annotations=[
            # Bottom summary bar
            dict(
                text=(
                    f"<b style='color:#c9a84c'>Loan:</b> {loan_fmt} &nbsp;|&nbsp; "
                    f"<b style='color:#c9a84c'>Annual Income:</b> {income_fmt} &nbsp;|&nbsp; "
                    f"<b style='color:#c9a84c'>FICO Score:</b> {fico_score}"
                ),
                x=0.5, y=-0.12, xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="#8a9bb0", size=11, family="Inter"),
                align="center",
            ),
        ],
    )
    return fig


# ── Header ────────────────────────────────────────────────────────────────────
is_online, models_loaded = check_api_health()

col_logo, col_title, col_badge = st.columns([1, 8, 2])
with col_logo:
    st.markdown("<div style='font-size:3rem;padding-top:8px'>🏦</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <div>
      <h1 style='color:#c9a84c;margin:0;font-size:1.8rem;font-weight:700;letter-spacing:1px;'>
        Multi-Agent Loan Evaluation System
      </h1>
      <p style='color:#5a7a9a;margin:4px 0 0 0;font-size:0.85rem;letter-spacing:0.5px;'>
        Powered by LangGraph &nbsp;·&nbsp; LightGBM &nbsp;·&nbsp; T-Learner &nbsp;·&nbsp; SHAP &nbsp;·&nbsp; Gemini AI
      </p>
    </div>
    """, unsafe_allow_html=True)
with col_badge:
    if is_online and models_loaded:
        st.markdown(
            "<div style='padding-top:14px'><span class='status-badge-online'>● API Online</span></div>",
            unsafe_allow_html=True,
        )
    elif is_online:
        st.markdown(
            "<div style='padding-top:14px'><span class='status-badge-offline'>● Models Loading</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='padding-top:14px'><span class='status-badge-offline'>● API Offline</span></div>",
            unsafe_allow_html=True,
        )

st.markdown("<hr style='border:none;border-top:1px solid #1e3a5f;margin:12px 0 20px 0'>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    btn_text = "🌙 Switch to Dark Theme" if st.session_state.theme == "light" else "☀️ Switch to Light Theme"
    st.button(btn_text, on_click=toggle_theme, use_container_width=True)

    st.markdown(
        "<div style='text-align:center;padding:12px 0 4px 0'>"
        "<span style='color:#c9a84c;font-size:1.1rem;font-weight:700;letter-spacing:1px;'>"
        "📋 Loan Application Form</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border:none;border-top:1px solid #1e3a5f;margin:4px 0 12px 0'>", unsafe_allow_html=True)

    # ── Financial Details ────────────────────────────────────────────────────────
    st.markdown("### 💵 Financial Details")

    loan_amnt = st.number_input(
        "Loan Amount ($)", min_value=500.0, max_value=500_000.0,
        value=15_000.0, step=500.0, format="%.0f",
        help="Enter the requested loan amount",
    )

    annual_inc = st.number_input(
        "Annual Income ($)", min_value=1_000.0, max_value=10_000_000.0,
        value=65_000.0, step=1_000.0, format="%.0f",
        help="Gross annual income of the applicant",
    )

    dti = st.number_input(
        "Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0,
        value=18.5, step=0.5,
        help="Total monthly debt obligations as a percentage of gross monthly income",
    )

    fico = st.number_input(
        "FICO Score (low)", min_value=300, max_value=850,
        value=700, step=5,
        help="Applicant's FICO Score",
    )

    int_rate = st.number_input(
        "Interest Rate (%)", min_value=1.0, max_value=40.0,
        value=12.5, step=0.25,
        help="Proposed annual interest rate",
    )

    # ── Loan Characteristics ─────────────────────────────────────────────────────
    st.markdown("### 🏷️ Loan Characteristics")

    grade = st.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"],
        help="Assigned loan grade",
    )

    purpose = st.selectbox(
        "Loan Purpose",
        ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase", "medical", "small_business"],
        help="Select the primary purpose of the loan",
    )

    term = st.selectbox(
        "Loan Term",
        ["36 months", "60 months"],
        help="Select the repayment tenure",
    )

    # ── Demographics & Credit Bureau ──────────────────────────────────────────────
    st.markdown("### 👤 Demographics & Bureau")

    home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"],
        help="Current residential status",
    )

    client_age = st.slider(
        "Client Age (years)", min_value=18, max_value=80, value=35,
        help="Applicant's age in years",
    )

    emp_length = st.slider(
        "Employment Length (years)", min_value=0, max_value=40, value=5,
        help="Number of years in current employment",
    )

    st.markdown("<p style='color:#a8b8cc;font-size:0.85rem;margin-bottom:4px;margin-top:10px;'>External Credit Scores</p>", unsafe_allow_html=True)
    ext_source_1 = st.slider("Bureau Score 1 (Equifax)", 0.0, 1.0, 0.50, 0.01)
    ext_source_2 = st.slider("Bureau Score 2 (Experian)", 0.0, 1.0, 0.50, 0.01)
    ext_source_3 = st.slider("Bureau Score 3 (TransUnion)", 0.0, 1.0, 0.50, 0.01)

    amt_credit = st.number_input(
        "Bureau AMT_CREDIT",
        value=float(loan_amnt), step=500.0, format="%.0f",
    )

    amt_annuity = st.number_input(
        "Bureau AMT_ANNUITY",
        value=round(loan_amnt / 36, 2), step=100.0, format="%.2f",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.button(
        "Evaluate Loan Application 🔍",
        type="primary",
        use_container_width=True,
    )


# ── Main Panel ────────────────────────────────────────────────────────────────
if not submit:
    # ── Dashboard: Agent Cards ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>AI Agent Pipeline</div>", unsafe_allow_html=True)

    card_col1, card_col2, card_col3 = st.columns(3)

    with card_col1:
        st.markdown("""
        <div class='agent-card'>
          <span class='agent-icon'>&#128269;</span>
          <h4>Credit Risk Agent</h4>
          <p>Evaluates default probability using a LightGBM model trained on Home Credit Bureau data.
          Generates SHAP explanations to identify the top risk drivers for each application.</p>
          <br>
          <p style='color:#c9a84c;font-size:0.78rem;'>Model: LightGBM &middot; Explainability: SHAP</p>
        </div>
        """, unsafe_allow_html=True)

    with card_col2:
        st.markdown("""
        <div class='agent-card'>
          <span class='agent-icon'>&#128737;</span>
          <h4>Fraud Detection Agent</h4>
          <p>Detects anomalous transaction patterns using a LightGBM classifier trained on the
          IEEE-CIS Fraud Detection dataset. Flags high-risk applications for additional scrutiny.</p>
          <br>
          <p style='color:#c9a84c;font-size:0.78rem;'>Model: LightGBM &middot; Dataset: IEEE-CIS</p>
        </div>
        """, unsafe_allow_html=True)

    with card_col3:
        st.markdown("""
        <div class='agent-card'>
          <span class='agent-icon'>&#128200;</span>
          <h4>Rate Sensitivity Agent</h4>
          <p>Estimates the causal effect of interest rate changes on repayment behaviour using a
          T-Learner (X-Learner) uplift model. Segments applicants for personalised rate offers.</p>
          <br>
          <p style='color:#c9a84c;font-size:0.78rem;'>Model: T-Learner &middot; Method: Causal ML</p>
        </div>
        """, unsafe_allow_html=True)


    # ── Dashboard: How It Works ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>How It Works</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0f2035;border:1px solid #1e3a5f;border-radius:12px;padding:24px;color:#8a9bb0;font-size:0.88rem;line-height:1.8;'>
      <ol style='margin:0;padding-left:20px;'>
        <li><strong style='color:#c9a84c;'>Application Intake</strong> &mdash; Loan details are validated and normalised.</li>
        <li><strong style='color:#c9a84c;'>Credit Risk Assessment</strong> &mdash; LightGBM model predicts probability of default using bureau data.</li>
        <li><strong style='color:#c9a84c;'>Fraud Screening</strong> &mdash; Transaction patterns are analysed for anomalies using IEEE-CIS trained model.</li>
        <li><strong style='color:#c9a84c;'>Rate Sensitivity Analysis</strong> &mdash; T-Learner estimates causal uplift to personalise interest rate offers.</li>
        <li><strong style='color:#c9a84c;'>Supervisor Decision</strong> &mdash; LangGraph orchestrator synthesises all signals and issues a final decision.</li>
        <li><strong style='color:#c9a84c;'>Audit Report</strong> &mdash; Gemini AI generates a human-readable narrative with SHAP explanations.</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

    if not is_online:
        st.markdown(
            "<div style='background:#2d0a0a;border:1px solid #e74c3c;border-radius:8px;"
            "padding:16px 20px;color:#e8a0a0;font-size:0.88rem;margin-top:20px;'>"
            "API server is offline. Start it with: "
            "<code style='background:#1a0606;padding:2px 8px;border-radius:4px;color:#c9a84c;'>"
            "uvicorn api.main:app --reload</code></div>",
            unsafe_allow_html=True,
        )


else:
    # ── Submission: API Call ──────────────────────────────────────────────────
    payload = {
        "loan_amnt": loan_amnt,
        "fico_range_low": float(fico),
        "dti": dti,
        "annual_inc": annual_inc,
        "int_rate": int_rate,
        "grade": grade,
        "purpose": purpose,
        "term": term,
        "home_ownership": home_ownership,
        "emp_length_num": float(emp_length),
        "DAYS_BIRTH": -1 * int(client_age * 365.25),
        "DAYS_EMPLOYED": -1 * int(emp_length * 365.25),
        "EXT_SOURCE_1": ext_source_1,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "AMT_CREDIT": amt_credit,
        "AMT_INCOME_TOTAL": annual_inc,
        "AMT_ANNUITY": amt_annuity,
        "TransactionAmt": loan_amnt,    # Mapped for Fraud Model
        "installment": amt_annuity,     # Mapped for Uplift Model
    }

    # Debug: Show payload being sent
    with st.expander("🔍 Debug: View Request Payload", expanded=False):
        st.json(payload)
    
    with st.spinner("Running multi-agent evaluation... Please wait."):
        try:
            resp = httpx.post(f"{API_BASE}/evaluate-loan", json=payload, timeout=120.0)
            resp.raise_for_status()
            data = resp.json()
            
            # Debug: Show response
            with st.expander("🔍 Debug: View API Response", expanded=False):
                st.json({
                    "default_probability": data.get("default_probability"),
                    "fraud_score": data.get("fraud_score"),
                    "uplift_score": data.get("uplift_score"),
                    "errors": data.get("errors"),
                })
        except httpx.ReadTimeout:
            st.markdown(
                "<div style='background:#2d0a0a;border:1px solid #e74c3c;border-radius:8px;"
                "padding:16px 20px;color:#e8a0a0;'>"
                "<strong>Request timed out</strong> after 120 seconds. "
                "The server may be overloaded &mdash; please try again.</div>",
                unsafe_allow_html=True,
            )
            st.stop()
        except httpx.ConnectError:
            st.markdown(
                "<div style='background:#2d0a0a;border:1px solid #e74c3c;border-radius:8px;"
                "padding:16px 20px;color:#e8a0a0;'>"
                "<strong>Cannot connect to API.</strong> "
                "Make sure the FastAPI server is running.</div>",
                unsafe_allow_html=True,
            )
            st.stop()
        except Exception as exc:
            st.markdown(
                f"<div style='background:#2d0a0a;border:1px solid #e74c3c;border-radius:8px;"
                f"padding:16px 20px;color:#e8a0a0;'>"
                f"<strong>API error:</strong> {exc}</div>",
                unsafe_allow_html=True,
            )
            st.stop()

    # ── Parse response ────────────────────────────────────────────────────────
    raw_decision = data.get("decision", "error")
    decision_info = DECISION_MAP.get(raw_decision, DECISION_MAP.get(raw_decision.upper(), None))

    if decision_info:
        label, css_class, color, icon = decision_info
    else:
        label, css_class, color, icon = raw_decision.upper(), "decision-committee", "#f39c12", "⚪"

    # ── Decision Banner ───────────────────────────────────────────────────────
    st.markdown(
        f"<div class='{css_class}'>"
        f"<p class='decision-title' style='color:{color};'>{icon} {label}</p>"
        f"<p class='decision-subtitle'>Loan Evaluation Complete</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Decision Reason ───────────────────────────────────────────────────────
    reason = data.get("decision_reason", "")
    if reason:
        st.markdown(
            f"<div class='reason-box'><strong>Decision Rationale:</strong><br>{reason}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 4 Metric Columns ─────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    default_prob = data.get("default_probability")
    fraud_score  = data.get("fraud_score")
    uplift_score = data.get("uplift_score")
    segment_raw  = data.get("segment", "")
    segment      = map_segment(segment_raw)

    with m1:
        dp_pct = f"{np.clip((default_prob or 0)*100, 0, 100):.1f}%" if default_prob is not None else "N/A"
        risk_band = data.get("risk_band", "")
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Default Risk</div>"
            f"<div class='metric-value'>{dp_pct}</div>"
            f"<div class='metric-sub'>{risk_band}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with m2:
        fs_pct = f"{np.clip((fraud_score or 0)*100, 0, 100):.1f}%" if fraud_score is not None else "N/A"
        risk_level = data.get("risk_level", "")
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Fraud Risk Score</div>"
            f"<div class='metric-value'>{fs_pct}</div>"
            f"<div class='metric-sub'>{risk_level}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with m3:
        us_fmt = f"{uplift_score:.4f}" if uplift_score is not None else "N/A"
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Rate Sensitivity Score</div>"
            f"<div class='metric-value'>{us_fmt}</div>"
            f"<div class='metric-sub'>ITE Estimate</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Applicant Segment</div>"
            f"<div class='metric-value' style='font-size:1.1rem;'>{segment or 'N/A'}</div>"
            f"<div class='metric-sub'>Uplift Segment</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk Gauges ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Risk Gauge Dashboard</div>", unsafe_allow_html=True)
    gauge_fig = build_gauges(default_prob, fraud_score, uplift_score, loan_amnt, annual_inc, fico)
    st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})


    # ── Audit Narrative ───────────────────────────────────────────────────────
    narrative = data.get("shap_narrative")
    if narrative:
        st.markdown("<div class='section-header'>AI Audit Narrative</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='narrative-card'>"
            f"<h4>Gemini AI Analysis</h4>"
            f"{narrative}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Agent Pipeline Status ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Agent Pipeline Status</div>", unsafe_allow_html=True)

    agents = [
        ("&#128203;", "Intake",        "Application\nValidated"),
        ("&#128269;", "Credit Risk",   "Default Risk\nAssessed"),
        ("&#128737;", "Fraud Screen",  "Fraud Risk\nScanned"),
        ("&#128200;", "Rate Uplift",   "Sensitivity\nAnalysed"),
        ("&#9878;",   "Supervisor",    "Decision\nIssued"),
    ]

    pipeline_html = "<div class='pipeline-container'>"
    for i, (icon, name, status) in enumerate(agents):
        pipeline_html += (
            f"<div class='pipeline-step'>"
            f"<span class='step-icon'>{icon}</span>"
            f"<span class='step-name'>{name}</span>"
            f"<span class='step-check'>&#10003; {status.split(chr(10))[0]}</span>"
            f"</div>"
        )
        if i < len(agents) - 1:
            pipeline_html += "<span class='pipeline-arrow'>&#8594;</span>"
    pipeline_html += "</div>"

    st.markdown(pipeline_html, unsafe_allow_html=True)

    # ── PDF Download ──────────────────────────────────────────────────────────
    pdf_url = data.get("audit_pdf_url")
    if pdf_url:
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            pdf_resp = httpx.get(f"{API_BASE}{pdf_url}", timeout=10.0)
            pdf_resp.raise_for_status()
            st.download_button(
                label="&#128196; Download Audit Report (PDF)",
                data=pdf_resp.content,
                file_name="loan_audit_report.pdf",
                mime="application/pdf",
                use_container_width=False,
            )
        except Exception:
            st.markdown(
                "<div style='background:#2d2000;border:1px solid #f39c12;border-radius:8px;"
                "padding:12px 16px;color:#f0c060;font-size:0.85rem;'>"
                "PDF audit report is being generated. Please try again shortly.</div>",
                unsafe_allow_html=True,
            )

    # ── Agent Errors (if any) ─────────────────────────────────────────────────
    errors = data.get("errors")
    if errors:
        with st.expander("Agent Diagnostic Logs", expanded=False):
            for err in errors:
                st.markdown(
                    f"<div style='background:#2d0a0a;border-left:3px solid #e74c3c;"
                    f"padding:8px 12px;color:#e8a0a0;font-size:0.82rem;margin:4px 0;"
                    f"border-radius:0 4px 4px 0;'>{err}</div>",
                    unsafe_allow_html=True,
                )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>"
    "Multi-Agent Loan Evaluation System &nbsp;|&nbsp; BML Munjal University &nbsp;|&nbsp; "
    "Powered by Google Gemini AI"
    "</div>",
    unsafe_allow_html=True,
)

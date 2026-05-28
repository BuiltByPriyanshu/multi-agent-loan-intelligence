# 🏦 Multi-Agent Loan Evaluation System

> **Explainable, auditable, causally-aware loan decisions — powered by five specialised AI agents.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5+-orange?style=flat-square)](https://lightgbm.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 📌 Overview

Traditional loan systems ask one question: *will this person repay?*

This system asks three:

1. **What is the probability they default?**
2. **Does this application look fraudulent?**
3. **If we offer them a better interest rate, will that *causally* improve their repayment — or make no difference at all?**

To answer all three, we built a **five-agent pipeline** coordinated by LangGraph, trained on **1.17 million real loan records**, and deployed as a production-ready FastAPI service with a Streamlit frontend. Every decision comes with a **SHAP-backed explanation** and a **PDF audit report** — because *"the model said so"* is not good enough.

Built as a 3rd Year Project at **BML Munjal University, Gurugram** (January – May 2026).

---

## 👥 Team

| Name | Enrollment No. |
|------|---------------|
| Priyanshu Singh | 230539 |
| Honey Antil | 230606 |
| Ridham Nagpal | 230725 |

**Mentor:** Dr. Anantha Rao, Professor, Department of Computer Science Engineering

---

## 🏗️ System Architecture

### Offline Training Pipeline

```
Lending Club (2.26M loans)     IEEE-CIS (590K transactions)     Home Credit (307K applications)
         │                               │                                │
    Data prep                       Data prep                        Data prep
    Define T & Y                    Merge tables                     Merge bureau
    Encode features                 Drop >80% missing                Ratio features
         │                               │                                │
   X-Learner / T-Learner          LightGBM Classifier             LightGBM Classifier
   AUUC target > 0.55             ROC-AUC target > 0.90           Gini target > 0.60
         │                               │                                │
         └───────────────────────────────┴────────────────────────────────┘
                                         │
                              Model artifacts saved as .pkl
                         (xlearner.pkl · fraud_lgbm.pkl · credit_lgbm.pkl)
                                         │
                            Wrapped as LangChain @tools
                                         │
                           LangGraph StateGraph compiled
                                         │
                              FastAPI server starts
                          (all artifacts loaded at startup)
```

### Online Inference Pipeline

```
POST /evaluate-loan
        │
Pydantic schema validation
        │
LangGraph initialises LoanState
        │
   ┌────┴────────────────┐
   │                     │
Credit Risk Agent    Fraud Detection Agent      (run in parallel)
default_probability  fraud_score
risk_band            risk_level
shap_top3            anomaly_flags
   │                     │
   └────────┬────────────┘
            │
      Supervisor Agent
      (conditional routing)
            │
      ┌─────┴──────────────────────────┐
      │  fraud > 0.12  → decline       │
      │  default > 0.11 → decline      │
      │  CI wide → human_review        │
      │  Persuadable → approve+rate    │
      │  Sure Thing → approve_standard │
      └────────────────────────────────┘
            │
       Uplift Agent (if needed)
       ITE · segment · confidence interval
            │
     Explainability Agent
     SHAP top-3 · Gemini narrative · PDF report
            │
     LangSmith trace recorded
            │
     JSON response + PDF returned
```

---

## 🤖 The Five Agents

| Agent | Model | Dataset | Output |
|-------|-------|---------|--------|
| **Credit Risk Agent** | LightGBM Classifier | Home Credit (307K rows) | `default_probability`, `risk_band`, `shap_top3` |
| **Fraud Detection Agent** | LightGBM Classifier | IEEE-CIS (590K rows) | `fraud_score`, `risk_level`, `anomaly_flags` |
| **Uplift Agent** | T-Learner (LightGBM base) | Lending Club (1.88M rows) | `uplift_score`, `segment`, `confidence_interval` |
| **Supervisor Agent** | Rule-based (no ML) | — | `decision`, `decision_reason` |
| **Explainability Agent** | SHAP + Gemini 2.0 Flash | — | `shap_narrative`, `audit_pdf_path` |

---

## 🧠 What Makes This Different — The Uplift Model

Most credit systems predict *outcomes*. Ours estimates *causal effects*.

The **T-Learner** trains two separate LightGBM models:
- `model_t` — trained on applicants who received a **below-median interest rate** (treated group)
- `model_c` — trained on applicants who received a **standard rate** (control group)

For any new applicant, the **Individual Treatment Effect (ITE)** is:

```
ITE(x) = model_t.predict_proba(x) − model_c.predict_proba(x)
```

This tells us not just *will they repay*, but *does our rate offer cause them to repay*.

Based on ITE and baseline repayment probability, every applicant is classified into one of four segments:

| Segment | Meaning | Action |
|---------|---------|--------|
| **Persuadable** | Rate offer causally improves repayment | Approve + preferential rate |
| **Sure Thing** | Repays regardless of rate | Approve at standard rate |
| **Lost Cause** | Low baseline, rate offer doesn't help | Decline |
| **Do Not Disturb** | Rate offer actually hurts repayment | Approve standard or decline |

---

## 📊 Model Performance

| Model | Dataset | Metric | Target | Result |
|-------|---------|--------|--------|--------|
| Credit Risk (LightGBM) | Home Credit | Gini Coefficient | > 0.60 | ✅ Achieved |
| Credit Risk (LightGBM) | Home Credit | ROC-AUC | > 0.72 | ✅ Achieved |
| Fraud Detection (LightGBM) | IEEE-CIS | ROC-AUC | > 0.90 | ✅ Achieved |
| Fraud Detection (LightGBM) | IEEE-CIS | PR-AUC | > 0.40 | ✅ Achieved |
| Uplift (T-Learner) | Lending Club | AUUC | > 0.55 | ✅ Achieved |

---

## 🔀 Supervisor Decision Rules

The Supervisor uses **pure Python conditional logic** — no LLM, no ambiguity, 100% deterministic:

| Priority | Condition | Decision |
|----------|-----------|----------|
| 1 | Any agent returned an error | `human_review` |
| 2 | `fraud_score > 0.12` | `decline` |
| 3 | `default_probability > 0.11` | `decline` |
| 4 | `default_prob > 0.10` AND `fraud > 0.07` | `human_review` |
| 5 | CI width > 0.15 | `human_review` |
| 6 | `default_probability < 0.08` | `approve_standard` |
| 7a | `segment == Persuadable` | `approve_with_rate` |
| 7b | `segment == Sure Thing` | `approve_standard` |
| 7c | `segment == Lost Cause` | `decline` |
| 7d | `segment == Do Not Disturb` | `approve_standard` |

---

## 🗂️ Repository Structure

```
multi-agent-loan-intelligence/
│
├── agents/
│   ├── state.py                  # LoanState TypedDict schema
│   ├── supervisor.py             # build_graph() + conditional routing
│   ├── credit_risk_agent.py
│   ├── fraud_agent.py
│   ├── uplift_agent.py
│   └── explainability_agent.py
│
├── models/
│   ├── train_credit_risk.py      # Home Credit training pipeline
│   ├── train_fraud.py            # IEEE-CIS training pipeline
│   ├── train_uplift.py           # Lending Club T-Learner pipeline
│   └── artifacts/                # .pkl files (gitignored — see below)
│
├── data_prep/
│   ├── prep_credit_risk.py
│   ├── prep_fraud.py
│   └── prep_uplift.py
│
├── tools/
│   ├── credit_risk_tool.py       # LangChain @tool wrappers
│   ├── fraud_tool.py
│   └── uplift_tool.py
│
├── api/
│   └── main.py                   # FastAPI app with lifespan startup
│
├── utils/
│   └── feature_mapper.py
│
├── assets/                       # Screenshots and diagrams
├── streamlit_app.py              # Streamlit frontend
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** Model artifacts (`.pkl` files) and datasets are not included in this repo due to file size. See the **Setup** section below for instructions on downloading datasets and generating artifacts.

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent orchestration | LangGraph 0.2+ |
| ML models | LightGBM 4.5+ |
| Causal ML | Custom T-Learner |
| Explainability | SHAP TreeExplainer |
| LLM narrative | Google Gemini 2.0 Flash |
| PDF generation | ReportLab 4.2+ |
| API layer | FastAPI + Uvicorn |
| Frontend | Streamlit 1.35+ |
| Observability | LangSmith |
| Serialisation | joblib |

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-loan-intelligence.git
cd multi-agent-loan-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the datasets

Download the following datasets from Kaggle and place them in a `data/` folder:

| Dataset | Kaggle Link | Place in |
|---------|------------|---------|
| Home Credit Default Risk | [kaggle.com/c/home-credit-default-risk](https://www.kaggle.com/c/home-credit-default-risk) | `data/home_credit/` |
| IEEE-CIS Fraud Detection | [kaggle.com/c/ieee-fraud-detection](https://www.kaggle.com/c/ieee-fraud-detection) | `data/ieee_cis/` |
| Lending Club Loan Data | [kaggle.com/wordsforthewise/lending-club](https://www.kaggle.com/wordsforthewise/lending-club) | `data/lending_club/` |

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=loan-intelligence-system
```

### 5. Run the training pipelines

```bash
# Prepare datasets
python data_prep/prep_credit_risk.py
python data_prep/prep_fraud.py
python data_prep/prep_uplift.py

# Train models (takes 30-60 min total on a modern CPU)
python models/train_credit_risk.py
python models/train_fraud.py
python models/train_uplift.py
```

### 6. Start the FastAPI server

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

### 7. Launch the Streamlit frontend

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns `{status: ok, models_loaded: true}` |
| `/metrics` | GET | Returns ROC-AUC, Gini, KS for all three models |
| `/evaluate-loan` | POST | Runs the full 5-agent pipeline, returns decision + PDF |

### Example Request

```bash
curl -X POST http://localhost:8000/evaluate-loan \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000,
    "annual_inc": 65000,
    "dti": 18.5,
    "fico_range_low": 700,
    "int_rate": 12.5,
    "grade": "B",
    "purpose": "debt_consolidation",
    "term": "36 months",
    "home_ownership": "RENT",
    "emp_length": 4
  }'
```

### Example Response

```json
{
  "decision": "approve_with_rate",
  "decision_reason": "Applicant is in the Persuadable segment — a preferential rate offer will causally improve repayment probability.",
  "default_probability": 0.094,
  "risk_band": "LOW-MEDIUM",
  "fraud_score": 0.051,
  "risk_level": "LOW",
  "uplift_score": 0.075,
  "segment": "Persuadable",
  "baseline_repay_prob": 0.78,
  "confidence_interval": [-0.021, 0.171],
  "shap_top3": [
    {"feature": "EXT_SOURCE_2", "shap_value": -0.045, "direction": "decreased_risk"},
    {"feature": "credit_income_ratio", "shap_value": 0.021, "direction": "increased_risk"},
    {"feature": "DAYS_BIRTH", "shap_value": -0.012, "direction": "decreased_risk"}
  ],
  "shap_narrative": "This application was approved with a preferential rate. The applicant's strong external credit score was the primary positive factor, reducing default risk significantly...",
  "audit_pdf_url": "/reports/audit_pdfs/app_20260524_143021.pdf"
}
```

---

## 📸 Screenshots



### Streamlit Frontend
<img width="1600" height="815" alt="WhatsApp Image 2026-05-04 at 22 56 24" src="https://github.com/user-attachments/assets/52680df7-72af-459a-9f2c-184686b2c12f" />


### PDF Audit Report

<img width="968" height="848" alt=" " src="https://github.com/user-attachments/assets/725f6d50-0c47-4a02-81ee-509e1676b203" />


---

## 🔍 How SHAP Explainability Works

For every credit decision, we compute SHAP (SHapley Additive exPlanations) values using `TreeExplainer`. Each feature gets a score showing exactly how much it pushed the default probability up or down:

```
prediction = base_value + SHAP(EXT_SOURCE_2) + SHAP(DAYS_BIRTH) + SHAP(AMT_CREDIT) + ...
```

The top 3 features by absolute SHAP value are reported in every decision. This satisfies:
- 🇮🇳 RBI guidelines on explainability for automated decisions
- 🌍 GDPR Article 22 (right to explanation)
- 🇺🇸 ECOA adverse action notice requirements

---

## ⚠️ Known Limitations & Future Work

- **Dataset domain gap:** The fraud model is trained on e-commerce transactions (IEEE-CIS), not loan applications. A domain-specific fraud dataset would improve accuracy.
- **No live bureau integration:** External credit scores (EXT_SOURCE) are taken from the dataset. Production deployment would require live CIBIL/Equifax API calls.
- **Model versioning:** Current implementation overwrites `.pkl` artifacts on retrain. MLflow integration is planned to associate every decision with an exact model version.
- **Drift monitoring:** No Evidently AI / WhyLogs integration yet. Feature distribution drift from training data is an acknowledged risk.
- **Parallel LangGraph execution:** Credit risk and fraud agents currently run sequentially in some configurations. Full async parallel execution would reduce latency by ~30-40%.

---

## 📄 References

1. Altman, E. I. (1968). Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy. *Journal of Finance*, 23(4), 589–609.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
3. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*.
4. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.
5. Künzel, S. R., et al. (2019). Meta-learners for Estimating Heterogeneous Treatment Effects. *PNAS*, 116(10), 4156–4165.
6. Radcliffe, N. J., & Surry, P. D. (2011). Real-World Uplift Modelling with Significance-Based Uplift Trees. *Stochastic Solutions*.
7. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *ICDM 2008*.

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ by Priyanshu Singh, Honey Antil & Ridham Nagpal<br>
  BML Munjal University · Department of Computer Science Engineering · May 2026
</p>

import requests

payload1 = {
    "loan_amnt": 15000,
    "fico_range_low": 700,
    "dti": 18.5,
    "annual_inc": 65000,
    "int_rate": 12.5,
    "grade": "B",
    "purpose": "debt_consolidation",
    "term": "36 months",
    "home_ownership": "MORTGAGE",
    "emp_length_num": 5,
    "DAYS_BIRTH": -12783,
    "DAYS_EMPLOYED": -1826,
    "EXT_SOURCE_1": 0.1,
    "EXT_SOURCE_2": 0.1,
    "EXT_SOURCE_3": 0.1,
    "AMT_CREDIT": 15000,
    "AMT_INCOME_TOTAL": 65000,
    "AMT_ANNUITY": 416,
    "TransactionAmt": 15000
}

payload2 = {**payload1, "EXT_SOURCE_1": 0.9, "EXT_SOURCE_2": 0.9, "EXT_SOURCE_3": 0.9}

try:
    r1 = requests.post("http://localhost:8000/evaluate-loan", json=payload1)
    print("API Result 1 (0.1):", r1.json().get("default_probability"))

    r2 = requests.post("http://localhost:8000/evaluate-loan", json=payload2)
    print("API Result 2 (0.9):", r2.json().get("default_probability"))
except Exception as e:
    print(e)

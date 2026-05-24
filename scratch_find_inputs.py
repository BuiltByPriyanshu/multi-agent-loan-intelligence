import sys
import os

# Add current directory to path so we can import from tools
sys.path.append(os.getcwd())

from tools.uplift_tool import run_uplift_model

def search_segments():
    segments_found = {}
    
    # Base applicant profile
    base_applicant = {
        "loan_amnt": 15000.0,
        "fico_range_low": 700.0,
        "dti": 18.5,
        "annual_inc": 65000.0,
        "int_rate": 12.5,
        "grade": "B",
        "purpose": "debt_consolidation",
        "term": "36 months",
        "home_ownership": "MORTGAGE",
        "emp_length_num": 3.0,
        "AMT_CREDIT": 15000.0,
        "AMT_INCOME_TOTAL": 65000.0,
        "AMT_ANNUITY": 416.0,
        "TransactionAmt": 15000.0,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.5,
        "DAYS_BIRTH": 35 * -365,
        "DAYS_EMPLOYED": 5 * -365
    }

    # Generate a bunch of variations
    ficos = [500, 600, 680, 750, 800]
    dtis = [5, 15, 25, 35]
    incomes = [30000, 65000, 150000]
    loans = [5000, 15000, 35000]
    int_rates = [5.0, 15.0, 25.0]

    for fico in ficos:
        for dti in dtis:
            for income in incomes:
                for loan in loans:
                    for rate in int_rates:
                        app = dict(base_applicant)
                        app["fico_range_low"] = float(fico)
                        app["dti"] = float(dti)
                        app["annual_inc"] = float(income)
                        app["loan_amnt"] = float(loan)
                        app["int_rate"] = float(rate)
                        
                        # also scale ext_sources roughly with fico (500=0.1, 800=0.9)
                        scaled_ext = (fico - 500) / 300 * 0.8 + 0.1
                        app["EXT_SOURCE_1"] = scaled_ext
                        app["EXT_SOURCE_2"] = scaled_ext
                        app["EXT_SOURCE_3"] = scaled_ext
                        
                        try:
                            result = run_uplift_model.invoke({"applicant_features": app})
                            if result["status"] == "success":
                                seg = result["segment"]
                                if seg not in segments_found:
                                    segments_found[seg] = {
                                        "FICO": fico,
                                        "DTI": dti,
                                        "Income": income,
                                        "Loan": loan,
                                        "IntRate": rate,
                                        "tau": result["uplift_score"],
                                        "p0": result["baseline_repay_prob"]
                                    }
                                if len(segments_found) == 4:
                                    break
                        except Exception as e:
                            pass
                    if len(segments_found) == 4: break
                if len(segments_found) == 4: break
            if len(segments_found) == 4: break
        if len(segments_found) == 4: break
        
    for k, v in segments_found.items():
        print(f"Segment: {k} -> {v}")

search_segments()

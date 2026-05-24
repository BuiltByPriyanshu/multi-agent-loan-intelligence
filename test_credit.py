import json
from tools.credit_risk_tool import run_credit_risk_model

def test():
    feat1 = {"EXT_SOURCE_1": 0.1, "EXT_SOURCE_2": 0.1, "EXT_SOURCE_3": 0.1}
    feat2 = {"EXT_SOURCE_1": 0.9, "EXT_SOURCE_2": 0.9, "EXT_SOURCE_3": 0.9}

    res1 = run_credit_risk_model.invoke({"applicant_features": feat1})
    res2 = run_credit_risk_model.invoke({"applicant_features": feat2})

    print("Result 1:", res1["default_probability"])
    print("Result 2:", res2["default_probability"])

if __name__ == "__main__":
    test()

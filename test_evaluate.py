import asyncio
from api.schemas import LoanApplicationRequest
from api.main import evaluate_loan, lifespan
from fastapi import FastAPI

app = FastAPI()

async def run_test():
    async with lifespan(app):
        req = LoanApplicationRequest(
            loan_amnt=15000.0,
            fico_range_low=700.0,
            dti=18.5,
            annual_inc=65000.0,
            int_rate=12.5,
            grade="B",
            purpose="debt_consolidation",
            term="36 months",
            home_ownership="MORTGAGE",
            emp_length_num=3.0,
            AMT_CREDIT=15000.0,
            AMT_INCOME_TOTAL=65000.0,
            AMT_ANNUITY=416.0,
            TransactionAmt=15000.0
        )
        try:
            resp = evaluate_loan(req)
            print("Decision:", resp.decision)
        except Exception as e:
            print("Error during evaluate_loan:", e)

if __name__ == "__main__":
    asyncio.run(run_test())

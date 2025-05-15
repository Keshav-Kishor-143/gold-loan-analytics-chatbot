import os
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Import the updated orchestration entrypoint
from crew.crew_orchestrator import run_analysis

# Load .env
load_dotenv()

# FastAPI setup
app = FastAPI(
    title="Analytics Chatbot API",
    description="API for running analytical queries via a multi-agent pipeline."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Lock this down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default CSV locations
CUSTOMER_SUMMARY_PATH = "csv/customer_summary.csv"
PAYMENT_SUMMARY_PATH  = "csv/payment_summary.csv"

# Request / response models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    html_document: str
    status: str = "success"

@app.get("/")
async def root():
    return {
        "message": "Analytics Chatbot API is running.",
        "example_queries": [
            "Analyze loan distribution by customer type and branch",
            "Find average loan amount by scheme",
            "Compare NPA rates across branches"
        ]
    }

@app.post("/chatbot", response_model=QueryResponse)
async def chatbot(request: QueryRequest = Body(...)):
    # 1) Validate the query
    if not request.query or len(request.query.strip()) < 5:
        raise HTTPException(status_code=400, detail="Your query is too short or empty.")
    
    # 2) Resolve absolute CSV paths and verify they exist
    customer_path = os.path.abspath(CUSTOMER_SUMMARY_PATH)
    payment_path  = os.path.abspath(PAYMENT_SUMMARY_PATH)
    missing = [p for p in (customer_path, payment_path) if not os.path.exists(p)]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Data files not found: {', '.join(missing)}"
        )
    
    csv_files = {
        "loan_df": customer_path,
        "payment_df": payment_path
    }

    # 3) Delegate to the agent crew
    try:
        html_doc = run_analysis(request.query, csv_files=csv_files)
        # Expecting run_analysis to return the html_document string (or an error message)
        return QueryResponse(html_document=html_doc)
    except Exception as e:
        # Log internally, then return 500
        print(f"[Error in /chatbot] {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8010, reload=True)

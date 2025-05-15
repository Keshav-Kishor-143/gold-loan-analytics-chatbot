import os
import sys
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from io import StringIO
import contextlib
import locale
import asyncio
import logging
# from crew.crew_orchestrator import run_analysis
from fastapi import FastAPI, HTTPException, Depends, Request, Body
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import re
from pydantic import BaseModel, Field
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the database connection
from connection.dbConnection import execute_query, DatabaseConnection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables from .env file
load_dotenv()


app = FastAPI()

# Initialize database connection on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database connection
    try:
        logger.info("Initializing database connection...")
        db = DatabaseConnection()
        await db.warm_up_pool()
        logger.info("Database connection initialized and warmed up successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database connection: {str(e)}")
        raise
    
    yield  # This is where the app runs
    
    # Shutdown: Clean up resources
    try:
        logger.info("Shutting down database connections...")
        # Any cleanup needed for database connections
        logger.info("Database connections shut down successfully")
    except Exception as e:
        logger.error(f"Error shutting down database connections: {str(e)}")
        
class QueryRequest(BaseModel):
    query: str



@app.get("/")
def read_root():
    return {"message": "Gold Loan Analytics API is running."}


@app.post("/execute-query")
async def execute_sql_query(request: QueryRequest):
    """
    Execute a SQL query against the database and return the results.
    
    Args:
        request: SQLQueryRequest containing the query and optional parameters
    
    Returns:
        JSON response with query results or error message
    """
    try:
        logger.info(f"Executing SQL query: {request.query}")
        
        # Execute the query using the database connection
        result = await execute_query(request.query)
        
        # Return the results
        return {
            "status": "success",
            "error": None,
            "results": result,
            "row_count": len(result) if isinstance(result, list) else 0
        }
    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "results": None,
                "row_count": 0
            }
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
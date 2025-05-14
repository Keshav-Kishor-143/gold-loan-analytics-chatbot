import requests
from typing import Dict, Any, Optional
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import logging
import pandas as pd
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseQueryInput(BaseModel):
    query: str = Field(
        ..., 
        description="SQL query to execute against the database"
    )
    save_csv: bool = Field(
        False,
        description="Whether to save the results to a CSV file"
    )
    csv_path: Optional[str] = Field(
        None,
        description="Path where to save the CSV file (if save_csv is True)"
    )
    file_suffix: Optional[str] = Field(
        "",
        description="Optional suffix to add to the CSV filename"
    )
    model_config = {
        'extra': 'forbid',
        'validate_assignment': True
    }

def call_api(query: str) -> Dict[str, Any]:
    """Make API call to execute the query"""
    url = "http://localhost:8000/execute-query"
    params = {"query": query}
    response = requests.post(url, json=params)
    return response.json()

def execute_query(query: str, save_csv: bool = False, csv_path: Optional[str] = None, file_suffix: str = "") -> Dict[str, Any]:
    """Execute a SQL query using the database API and optionally save results to CSV."""
    try:
        logger.info(f"Executing query: {query}")
        filtered_query = query.strip()
        result = call_api(filtered_query)
        logger.info("Query executed successfully")
        
        # If save_csv is True and we have results, save to CSV
        if save_csv and result.get("status") == "success" and result.get("results"):
            try:
                # Create DataFrame from results
                df = pd.DataFrame(result["results"])
                
                # Determine the CSV file path
                if csv_path:
                    # Create directory if it doesn't exist
                    directory = Path(csv_path).parent
                    os.makedirs(directory, exist_ok=True)
                    
                    # Add suffix to filename if provided
                    filename = Path(csv_path).stem + file_suffix + ".csv"
                    filepath = directory / filename
                else:
                    # Use a default path if none provided
                    import tempfile
                    directory = Path(tempfile.gettempdir())
                    filename = f"query_results{file_suffix}.csv"
                    filepath = directory / filename
                
                # Save to CSV
                df.to_csv(filepath, index=False)
                
                # Add CSV info to result
                result["csv_path"] = str(filepath)
                result["csv_saved"] = True
                result["csv_rows"] = len(df)
                result["csv_columns"] = list(df.columns)
                
                logger.info(f"Query results saved to CSV: {filepath}")
            except Exception as e:
                logger.error(f"Error saving results to CSV: {str(e)}")
                result["csv_saved"] = False
                result["csv_error"] = str(e)
        
        return result
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return {"status": "error", "error": str(e)}

# Create the database query tool
db_query_tool = StructuredTool(
    name="database_query_tool",
    description="A tool to execute SQL queries against the database, retrieve results, and optionally save them to CSV files.",
    func=execute_query,
    args_schema=DatabaseQueryInput
)

if __name__ == "__main__":
    # Test the tool
    test_query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE';"
    result = execute_query(test_query, save_csv=True)
    print("Test Query Result:", result)

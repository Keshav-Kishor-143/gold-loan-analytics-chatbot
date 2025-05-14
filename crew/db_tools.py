import requests
from typing import Dict, Any
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseQueryInput(BaseModel):
    query: str = Field(
        ..., 
        description="SQL query to execute against the database"
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

def execute_query(query: str) -> Dict[str, Any]:
    """Execute a SQL query using the database API."""
    try:
        logger.info(f"Executing query: {query}")
        filtered_query = query.strip()
        result = call_api(filtered_query)
        logger.info("Query executed successfully")
        return result
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return {"status": "error", "error": str(e)}

# Create the database query tool
db_query_tool = StructuredTool(
    name="database_query_tool",
    description="A tool to execute SQL queries against the database and retrieve results.",
    func=execute_query,
    args_schema=DatabaseQueryInput
)

if __name__ == "__main__":
    # Test the tool
    test_query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE';"
    result = execute_query(test_query)
    print("Test Query Result:", result)

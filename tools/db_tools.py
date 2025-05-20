import requests
from typing import Dict, Any, Optional
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import logging
import pandas as pd
from tools.state_manager import AnalysisStateManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseQueryInput(BaseModel):
    query: str = Field(
        ..., 
        description="SQL query to execute against the database"
    )
    save_to_state: bool = Field(
        True,
        description="Whether to save the results to state manager"
    )
    result_name: str = Field(
        "query_result",
        description="Name to use when saving the result DataFrame"
    )
    description: str = Field(
        "",
        description="Description of the query for analysis tracking"
    )

def call_api(query: str) -> Dict[str, Any]:
    """Make API call to execute the query"""
    url = "http://localhost:8000/execute-query"
    params = {"query": query}
    response = requests.post(url, json=params)
    return response.json()

class DatabaseQueryTool:
    def __init__(self, state_manager: Optional[AnalysisStateManager] = None):
        self.state_manager = state_manager or AnalysisStateManager()

    def execute_query(
        self,
        query: str,
        save_to_state: bool = True,
        result_name: str = "query_result",
        description: str = ""
    ) -> Dict[str, Any]:
        """Execute a SQL query and optionally save results to state manager."""
        try:
            logger.info(f"Executing query: {query}")
            filtered_query = query.strip()
            result = call_api(filtered_query)
            
            if result.get("status") == "success" and result.get("results"):
                # Create DataFrame from results
                df = pd.DataFrame(result["results"])
                
                if save_to_state:
                    # Save DataFrame to state
                    self.state_manager.save_dataframe(result_name, df)
                    
                    # Add query to code history
                    self.state_manager.add_code(
                        f"# SQL Query:\n{query}",
                        description or f"Database query: {result_name}"
                    )
                    
                    # Add analysis step
                    self.state_manager.add_analysis_step(
                        description or "Database query execution",
                        {
                            "query": query,
                            "result_name": result_name,
                            "row_count": len(df),
                            "columns": list(df.columns)
                        }
                    )

                # Create metadata response
                metadata = {
                    "status": "success",
                    "saved_to_state": save_to_state,
                    "result_name": result_name,
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "columns": list(df.columns),
                    "sample_data": df.head(5).to_dict(orient='records'),
                    "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024)  # in MB
                }
                
                return metadata
            
            return result
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {"status": "error", "error": str(e)}

# Create the database query tool
state_manager = AnalysisStateManager()
db_query_tool = StructuredTool(
    name="database_query_tool",
    description="""A tool to execute SQL queries and save results to state manager.
    Returns metadata about query results and maintains analysis history.
    Results are automatically saved as DataFrames for further analysis.""",
    func=DatabaseQueryTool(state_manager).execute_query,  # Pass state_manager instance
    args_schema=DatabaseQueryInput
)

if __name__ == "__main__":
    # Test the tool
    state_manager = AnalysisStateManager()
    db_tool = DatabaseQueryTool(state_manager)
    test_query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE';"
    result = db_tool.execute_query(
        query=test_query,
        result_name="table_list",
        description="Get list of all tables"
    )
    print("Test Query Result:", result)
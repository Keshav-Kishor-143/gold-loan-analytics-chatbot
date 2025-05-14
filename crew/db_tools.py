import os
import sys
from typing import Dict, Any
import asyncio
import logging
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import nest_asyncio

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the database connection
from connection.dbConnection import execute_query, DatabaseConnection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Define the input schema for the database query tool
class DatabaseQueryInput(BaseModel):
    query: str = Field(..., description="SQL query to execute")

# Initialize and warm up the database connection pool
async def init_db_connection():
    """Initialize and warm up the database connection pool."""
    try:
        db = DatabaseConnection()
        await db.warm_up_pool()
        logger.info("Database connection pool warmed up successfully")
        return db
    except Exception as e:
        logger.error(f"Error warming up database connection pool: {str(e)}")
        raise

async def _execute_db_query(query: str) -> Dict[str, Any]:
    """Execute a database query asynchronously."""
    try:
        logger.info(f"Executing query: {query}")
        result = await execute_query(query)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error executing query: {query}\n{str(e)}")
        return {"error": str(e)}

def _run_db_query(query: str) -> Dict[str, Any]:
    """Run the database query with proper event loop handling"""
    try:
        # Get the current event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the query
        return asyncio.run_coroutine_threadsafe(_execute_db_query(query), loop).result()
    except Exception as e:
        logger.error(f"Error in _run_db_query: {str(e)}")
        return {"error": str(e)}

# Create the database query tool
db_query_tool = StructuredTool(
    name="database_query_tool",
    description="A tool to execute SQL queries against the database and retrieve results.",
    func=_run_db_query,
    args_schema=DatabaseQueryInput
)

if __name__ == "__main__":
    # Example query to fetch table names
    test_query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE';"
    
    # Test the tool using a clean approach
    async def run_test():
        # Initialize the connection
        await init_db_connection()
        
        # Execute the test query
        result = await _execute_db_query(test_query)
        print("Test Query Result:", result)
        
        # Return to ensure the function completes
        return result
    
    # Create a new event loop and run the test
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    
    try:
        result = new_loop.run_until_complete(run_test())
    except Exception as e:
        print("Error while executing test query:", str(e))
    finally:
        # Clean up resources
        new_loop.run_until_complete(new_loop.shutdown_asyncgens())
        new_loop.close()
        print("Test completed and resources cleaned up")

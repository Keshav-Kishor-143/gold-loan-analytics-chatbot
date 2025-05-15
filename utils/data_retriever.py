from dbConnection import execute_query, DatabaseConnection
import logging
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataRetrieverAgent():
    def __init__(self):
        """Initialize the DataRetrieverAgent with database connection"""
        # Initialize the database connection
        self.db_connection = DatabaseConnection()
        logger.info("DataRetrieverAgent initialized")
        
    async def initialize(self):
        """Asynchronous initialization method to warm up the connection pool"""
        await self.db_connection.warm_up_pool()
        logger.info("DataRetrieverAgent connection pool warmed up")
        
    async def retrieve_data(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve data from the database using the provided query and parameters
        Args:
            query: SQL query string
            params: Optional parameters for the query
        Returns:
            List of dictionaries containing the query results
        """
        try:
            logger.info(f"Executing query: {query}")
            results = await execute_query(query, params)
            logger.info(f"Query executed successfully. Retrieved {len(results)} rows.")
            return results
        except Exception as e:
            logger.error(f"Error retrieving data: {str(e)}")
            return []


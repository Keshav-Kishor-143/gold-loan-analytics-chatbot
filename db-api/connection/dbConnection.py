import os
import sqlalchemy
from sqlalchemy import text
import urllib.parse
from datetime import datetime
from dotenv import load_dotenv
import logging
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    _instance = None
    _engine = None
    _pool_size = 5
    _connection_pool: List[sqlalchemy.engine.Connection] = []
    _connection_locks: List[Lock] = []
    _pool_lock = Lock()
    _executor = ThreadPoolExecutor(max_workers=10)
    _is_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        if not self._is_initialized:
            self.connection_health_checks = {}
            self._create_engine()
            self._initialize_connection_pool()
            self._is_initialized = True

    def _create_engine(self):
        try:
            db_user = "sa"
            db_password = urllib.parse.quote_plus(os.getenv('DB_PASSWORD', ''))
            db_host = "103.224.243.71"
            db_port = "1700"
            db_name = "GoldLoanChatbot"
            
            if not db_password:
                logger.warning("DB_PASSWORD environment variable not set")
            
            connection_string = (
                f"mssql+pymssql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            )
            self._engine = sqlalchemy.create_engine(
                connection_string,
                pool_timeout=30,
                pool_recycle=3600,
                pool_pre_ping=True
            )
            logger.info(f"Database engine created successfully for {db_host}/{db_name}")
        except Exception as e:
            logger.error(f"Failed to create database engine: {str(e)}")
            raise

    def _initialize_connection_pool(self):
        try:
            with self._pool_lock:
                if not self._connection_pool:
                    for i in range(self._pool_size):
                        try:
                            connection = self._engine.connect()
                            # Set DATEFORMAT to ensure consistent date handling
                            connection.execute(text("SET DATEFORMAT ymd"))
                            self._connection_pool.append(connection)
                            self._connection_locks.append(Lock())
                            logger.debug(f"Initialized connection {i+1}/{self._pool_size}")
                        except Exception as e:
                            logger.error(f"Failed to initialize connection {i+1}: {str(e)}")
                            continue
                    
                    if not self._connection_pool:
                        raise Exception("Failed to establish any database connections")
                        
                    logger.info(f"Connection pool initialized with {len(self._connection_pool)} connections")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {str(e)}")
            raise

    async def get_available_connection(self) -> Tuple[Optional[sqlalchemy.engine.Connection], int]:
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            for i, lock in enumerate(self._connection_locks):
                if lock.acquire(blocking=False):
                    try:
                        if not self._connection_pool[i].closed:
                            return self._connection_pool[i], i
                        else:
                            logger.warning(f"Connection {i} was closed, recreating")
                            connection = self._engine.connect()
                            connection.execute(text("SET DATEFORMAT ymd"))
                            self._connection_pool[i] = connection
                            return self._connection_pool[i], i
                    except Exception as e:
                        logger.error(f"Error checking connection {i}: {str(e)}")
                        lock.release()
            
            attempt += 1
            await asyncio.sleep(0.2)
        
        logger.warning("No connections available in pool, creating temporary connection")
        try:
            temp_connection = self._engine.connect()
            temp_connection.execute(text("SET DATEFORMAT ymd"))
            return temp_connection, -1
        except Exception as e:
            logger.error(f"Failed to create temporary connection: {str(e)}")
            raise Exception("No database connections available")

    def release_connection(self, connection_index: int):
        if connection_index == -1:
            try:
                self._connection_pool[0].close()
            except Exception as e:
                logger.error(f"Error closing temporary connection: {str(e)}")
        else:
            try:
                self._connection_locks[connection_index].release()
            except Exception as e:
                logger.error(f"Error releasing connection lock {connection_index}: {str(e)}")

    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        connection = None
        connection_index = None
        
        # Modify the query to handle date formatting properly
        modified_query = self._modify_query_for_date_handling(query)
       
        try:
            connection, connection_index = await self.get_available_connection()
            
            result = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._execute_query_with_connection,
                connection,
                modified_query,
                params
            )
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}\nQuery: {modified_query}")
            raise
        finally:
            if connection_index is not None:
                self.release_connection(connection_index)
    
    def _modify_query_for_date_handling(self, query: str) -> str:
        """
        Modify the query to handle date formatting properly by using CONVERT instead of FORMAT
        where appropriate
        """
        # Replace FORMAT([LoanDisbursementDate], 'yyyy-MM') with 
        # CONVERT(varchar(7), [LoanDisbursementDate], 120)
        modified_query = query.replace(
            "FORMAT([LoanDisbursementDate], 'yyyy-MM')",
            "CONVERT(varchar(7), [LoanDisbursementDate], 120)"
        )
        return modified_query

    def _execute_query_with_connection(self, connection, query, params=None):
        try:
            if params:
                result = connection.execute(text(query), params)
            else:
                result = connection.execute(text(query))
                
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}\nQuery: {query}")
            raise

    async def warm_up_pool(self):
        try:
            logger.info("Warming up connection pool...")
            for _ in range(min(len(self._connection_pool), self._pool_size)):
                connection, index = await self.get_available_connection()
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        lambda: connection.execute(text("SELECT 1 AS connection_test"))
                    )
                    logger.debug(f"Connection {index} warmed up successfully")
                finally:
                    self.release_connection(index)
            logger.info("Connection pool warm-up completed")
        except Exception as e:
            logger.error(f"Error during connection pool warm-up: {str(e)}")

async def execute_query(query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    db = DatabaseConnection()
    return await db.execute_query(query, params)


# Add this at the end of the file
async def test_connection():
    """Test the database connection and retrieve schema information"""
    try:
        # Initialize database connection
        db = DatabaseConnection()
        await db.warm_up_pool()
        
        # Get schema for loan_customers table
        customers_schema_query = """
        SELECT COLUMN_NAME, DATA_TYPE 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'Loan_Customer_Summary'
        """
        
        # Get schema for loan_payments table
        payments_schema_query = """
        SELECT COLUMN_NAME, DATA_TYPE 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'Loan_Payment_Summary'
        """
        
        print("Executing query for customer schema...")
        customers_schema = await execute_query(customers_schema_query)
        print(f"Retrieved {len(customers_schema)} columns for Loan_Customer_Summary")
        
        print("Executing query for payment schema...")
        payments_schema = await execute_query(payments_schema_query)
        print(f"Retrieved {len(payments_schema)} columns for Loan_Payment_Summary")
        
        # Print sample of schema
        if customers_schema:
            print("\nSample columns from Loan_Customer_Summary:")
            for col in customers_schema[:5]:  # Print first 5 columns
                print(f"  {col['COLUMN_NAME']}: {col['DATA_TYPE']}")
        
        if payments_schema:
            print("\nSample columns from Loan_Payment_Summary:")
            for col in payments_schema[:5]:  # Print first 5 columns
                print(f"  {col['COLUMN_NAME']}: {col['DATA_TYPE']}")
        
        return True
    except Exception as e:
        print(f"Error testing connection: {str(e)}")
        return False

if __name__ == "__main__":
    import asyncio
    
    # Create and run the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run the test
    success = loop.run_until_complete(test_connection())
    
    # Close the loop
    loop.close()
    
    if success:
        print("\nDatabase connection test successful!")
    else:
        print("\nDatabase connection test failed!")

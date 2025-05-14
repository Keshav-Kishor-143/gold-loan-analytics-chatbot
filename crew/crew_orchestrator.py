import os
import sys
from pathlib import Path
import tempfile
import pandas as pd
import asyncio
from crewai import Crew, Agent, Task
from crewai_tools import CodeInterpreterTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import logging

# Import the correct path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crew.db_tools import db_query_tool, init_db_connection
from connection.dbConnection import execute_query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_schema_info():
    """Get schema information for the database tables"""
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
    
    customers_schema = await execute_query(customers_schema_query)
    payments_schema = await execute_query(payments_schema_query)
    
    customers_schema_str = "\n".join([f"{col['COLUMN_NAME']}: {col['DATA_TYPE']}" for col in customers_schema])
    payments_schema_str = "\n".join([f"{col['COLUMN_NAME']}: {col['DATA_TYPE']}" for col in payments_schema])
    
    return customers_schema_str, payments_schema_str


def create_analysis_crew(user_query, llm, customers_schema_str, payments_schema_str):
    """Create and configure the analysis crew with all necessary agents and tasks"""
    
    code_interpreter = CodeInterpreterTool()
    
    # 1. Planner Agent - Creates a two-step plan
    planner = Agent(
        role='Query Planner',
        goal='Create a detailed two-step plan for data retrieval and code generation',
        backstory="""
            You are an expert data analyst who excels at planning complex data analyses.
            You create detailed plans that specify what data to retrieve and how to process it.
            Your plans are precise, executable, and focused on answering specific queries efficiently.
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    # 2. Data Retriever Agent - Retrieves filtered data based on plan
    data_retriever = Agent(
        role='Data Retriever',
        goal='Retrieve filtered data from database based on STEP 1 of the plan',
        backstory="""
            You are a database expert who specializes in efficient data retrieval.
            You take a plan and translate it into SQL queries that filter data at a basic level.
            You ensure only the necessary data is retrieved to answer the query.
            You return the SQL query and its results.
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[db_query_tool]  # Pass the StructuredTool directly
    )
    
    # Task 1: Planning Task - Create two-step plan
    planning_task = Task(
        description=f"""
        Based on the user query: "{user_query}"
        
        Create a detailed two-step plan that includes:
        
        STEP 1 - DATA RETRIEVAL PLAN:
        - Specify what basic filtering should be done (e.g., date ranges, specific customer types)
        - Identify which tables are needed ([Loan_Customer_Summary], [Loan_Payment_Summary], or both)
        - Do NOT include any aggregations, complex logic, or joins in this step
        - The goal is to retrieve a filtered subset of data that contains everything needed for further processing
        
        STEP 2 - CODE GENERATION PLAN:
        - Specify what detailed operations should be performed (filtering, grouping, joining, etc.)
        - List the exact columns to use from each table
        - Detail any calculations or transformations needed
        - Outline how to join the tables using LoanId when necessary
        - Describe what the final output should look like
        
        Database Schema Information:
        
        [Loan_Customer_Summary] table:
        {customers_schema_str}
        
        [Loan_Payment_Summary] table:
        {payments_schema_str}
        
        Return your plan as a structured document with clearly labeled STEP 1 and STEP 2 sections.
        """,
        agent=planner,
        expected_output="Detailed two-step plan for data retrieval and code generation"
    )
    
    # Task 2: Data Retrieval Task - Retrieve filtered data
    data_retrieval_task = Task(
        description="""
        Using the DATA RETRIEVAL PLAN from the Planner (especially STEP 1), generate and execute SQL queries.
        
        IMPORTANT: When using the database_query_tool, follow these exact steps:
        1. Formulate your SQL query based on STEP 1 of the plan
        2. Use the database_query_tool with your SQL query
        3. Wait for the results before proceeding
        
        For example, if you need to retrieve customer data, you might use:
        SELECT * FROM Loan_Customer_Summary LIMIT 10
        
        Return both the SQL query you used and the results you obtained.
        """,
        agent=data_retriever,
        expected_output="SQL query and retrieved data",
        context=[planning_task],
    )
    
    # Create the crew - start with just planning and data retrieval
    crew = Crew(
        agents=[planner, data_retriever],
        tasks=[planning_task, data_retrieval_task],
        verbose=True,
        process_type="sequential"  # Change from async to sequential
    )
    
    # Execute the crew
    result = crew.kickoff(
        inputs={
            "user_query": user_query
        }
    )
    
    return result


async def run_analysis(user_query):
    """Run the analysis using the crew and return the result"""
    load_dotenv()
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY", ''))
    
    # Initialize database connection first
    logger.info("Initializing database connection...")
    await init_db_connection()
    logger.info("Database connection initialized successfully")
    
    # Get schema info
    logger.info("Retrieving schema information...")
    customers_schema_str, payments_schema_str = await get_schema_info()
    logger.info("Schema information retrieved successfully")
    
    # Then create and run the crew
    logger.info("Creating and running analysis crew...")
    result = create_analysis_crew(user_query, llm, customers_schema_str, payments_schema_str)
    logger.info("Analysis crew completed successfully")
    return result

if __name__ == "__main__":
    load_dotenv()
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY", ''))
    user_query = "Analyze loan distribution by customer type and branch for the last month"
    
    # Create and run the event loop properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the analysis
        logger.info("Starting analysis...")
        result = loop.run_until_complete(run_analysis(user_query))
        print("Analysis Result:")
        print(result)
    except Exception as e:
        logger.error(f"Error running analysis: {str(e)}")
        print(f"Error running analysis: {str(e)}")
    finally:
        # Clean up resources
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        logger.info("Analysis completed and resources cleaned up")
        print("Analysis completed and resources cleaned up")

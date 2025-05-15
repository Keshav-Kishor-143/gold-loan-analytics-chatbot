# from crewai import Agent, Task
# from textwrap import dedent

from textwrap import dedent

from crewai import Agent, Task
from utils.db_tools import db_query_tool

def create_data_retriever_agent():
    return Agent(
        role='Data Retriever',
        goal='Retrieve and filter relevant data based on user query intent',
        backstory=dedent("""
            You are an expert in data management and preprocessing. You specialize in 
            understanding user intent from natural language queries and retrieving only
            the most relevant data needed for analysis. You excel at filtering large datasets
            to provide focused, relevant information that directly addresses the user's needs.
            You can call external Python functions to load or retrieve data as needed.
        """),
        verbose=True,
        allow_delegation=False,
        tools=[db_query_tool]  # Register the external function as a tool
    )

def create_retrieval_task(sql_query,agent):
    """Create the data retrieval task"""
    return Task(
        description=f"""
        Using the DATA RETRIEVAL PLAN from the Planner (especially STEP 1):
       
        2. Execute each query {sql_query} using the database_query_tool with save_csv=True
        4. Save the file like filtered_data_<suffix>.csv
       
        If you need data from multiple tables:
        - Use 'loan_customers' suffix for Loan_Customer_Summary data
        - Use 'loan_payments' suffix for Loan_Payment_Summary data
       
        For each query, provide:
        1. The SQL query used
        2. The metadata about the retrieved data (row count, columns, etc.)
        3. The path to the saved CSV file
       
        IMPORTANT: Do NOT include the full dataset in your response, only metadata.
        The actual data is saved to CSV files that will be used by the Code Generator.
        """,
        agent=agent,
        expected_output="SQL queries, metadata, and CSV file paths",

    )
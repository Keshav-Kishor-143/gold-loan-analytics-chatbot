
import json
import os
import sys
from pathlib import Path
import tempfile
from textwrap import dedent
import pandas as pd
import asyncio
from crewai import Crew, Agent, Task
from crewai_tools import CodeInterpreterTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Import the correct path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from crew.db_tools import db_query_tool
# from connection.dbConnection import execute_query


# async def get_schema_info():
    # """Get schema information for the database tables"""
    # # Get schema for loan_customers table
    # customers_schema_query = """
    # SELECT COLUMN_NAME, DATA_TYPE 
    # FROM INFORMATION_SCHEMA.COLUMNS 
    # WHERE TABLE_NAME = 'Loan_Customer_Summary'
    # """
    
    # # Get schema for loan_payments table
    # payments_schema_query = """
    # SELECT COLUMN_NAME, DATA_TYPE 
    # FROM INFORMATION_SCHEMA.COLUMNS 
    # WHERE TABLE_NAME = 'Loan_Payment_Summary'
    # """
    
    # customers_schema = await execute_query(customers_schema_query)
    # payments_schema = await execute_query(payments_schema_query)
    
    # customers_schema_str = "\n".join([f"{col['COLUMN_NAME']}: {col['DATA_TYPE']}" for col in customers_schema])
    # payments_schema_str = "\n".join([f"{col['COLUMN_NAME']}: {col['DATA_TYPE']}" for col in payments_schema])
    
    # return customers_schema_str, payments_schema_str

def load_schema():
    # Load JSON data from a file
    with open('./schema/views_schema.json', 'r') as file:
        schema = json.load(file)
        return schema


def create_analysis_crew(user_query, llm, schema):
    """Create and configure the analysis crew with all necessary agents and tasks"""
    temp_dir = Path('./temp_dir')
    
    code_interpreter = CodeInterpreterTool()
    # # 1. Planner Agent - Creates a two-step plan
    # planner = Agent(
    #     role='Query Planner',
    #     goal='Create a detailed two-step plan for data retrieval and code generation',
    #     backstory="""
    #         You are an expert data analyst who excels at planning complex data analyses.
    #         You create detailed plans that specify what data to retrieve and how to process it.
    #         Your plans are precise, executable, and focused on answering specific queries efficiently.
    #     """,
    #     verbose=True,
    #     allow_delegation=False,
    #     llm=llm
    # )

    sql_generator=Agent(
        role='SQL Generator',
        goal='Generate optimal SQL queries based on user intent and database schema',
        backstory=dedent("""
          You are an expert Data Scientist who able to understand user intent and precisely do the data modelling (selecting only required columns from complete dataset) from database schema.
          You only focus on the segregate the required columns only as per user intent.Main focus is to generate query to select only column.
            """),
        verbose=True,
        allow_delegation=False,
        tools=[]  # Add any specific tools needed for SQL generation
    )
 
    
    # 2. Data Retriever Agent - Retrieves filtered data based on plan
    data_retriever = Agent(
    role='Data Retriever',
    goal='Retrieve filtered data from database and save to CSV files',
    backstory="""
        You are a database expert who specializes in efficient data retrieval.
        You take a sql query from sql_generator agent and return metadata.
        You ensure only the necessary data is retrieved and saved to CSV files for further analysis.
        You return both the SQL queries used and the paths to the saved CSV files.
    """,
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[]
    )
    
    # 3. Code Generator Agent - Creates detailed Python code
    code_generator = Agent(
        role='Code Generator',
        goal='Generate Python code for detailed data processing',
        backstory="""
            You are a Python expert who specializes in data analysis code generation.
            You take a plan and create efficient Python code that processes data to answer specific queries.
            Your code is clean, well-commented, and focused on producing exactly what's needed.
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    # 4. Code Executor Agent - Executes the generated code
    code_executor = Agent(
        role='Code Executor',
        goal='Execute generated Python code and return results',
        backstory="""
            You are a Python execution specialist who runs code to process data.
            You take generated code and filtered data and execute it to produce final results.
            You ensure all operations are performed correctly and return clean, formatted results.
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[code_interpreter]
    )
    
    # # Task 1: Planning Task - Create two-step plan
    # planning_task = Task(
    #     description=f"""
    #     Based on the user query: "{user_query}"
        
    #     Create a detailed two-step plan that includes:
        
    #     STEP 1 - DATA RETRIEVAL PLAN:
    #     - Specify what basic filtering should be done (e.g., date ranges, specific customer types)
    #     - Identify which tables are needed ([Loan_Customer_Summary], [Loan_Payment_Summary], or both)
    #     - Do NOT include any aggregations, complex logic, or joins in this step
    #     - The goal is to retrieve a filtered subset of data that contains everything needed for further processing
        
    #     STEP 2 - CODE GENERATION PLAN:
    #     - Specify what detailed operations should be performed (filtering, grouping, joining, etc.)
    #     - List the exact columns to use from each table
    #     - Detail any calculations or transformations needed
    #     - Outline how to join the tables using LoanId when necessary
    #     - Describe what the final output should look like
        
    #     Database Schema Information:
        
    #     [Loan_Customer_Summary] table:
    #     {customers_schema_str}
        
    #     [Loan_Payment_Summary] table:
    #     {payments_schema_str}
        
    #     Return your plan as a structured document with clearly labeled STEP 1 and STEP 2 sections.
    #     """,
    #     agent=planner,
    #     expected_output="Detailed two-step plan for data retrieval and code generation"
    # )
    
     
    sql_generator_task=Task(
            description=f"""
        Based on the provided user query {user_query} and database schema {schema}:
    Understand the user's intent: Analyze the user's query to determine the required columns needed to fulfill the request. This involves identifying the specific fields or entities that the user is asking for, without any aggregate operations or calculations.
    Extract relevant columns: From the provided {schema}, identify the exact column names related to the user’s query, ensuring that only the essential columns for the user's intent are selected (without any unnecessary columns).
    Generate a SQL SELECT statement: Construct a simple SQL query that selects only the necessary columns from the relevant tables based on the user's intent, formatted for use in SQL Server Management Studio (SSMS). The query should be focused solely on retrieving the column names that are directly related to the user's request (i.e., no aggregation, joins, or transformations).
    Output: A SQL SELECT query with the required columns, which can then be used for further analysis in Python or other tools like Pandas.
    **Return only a valid Python dictionary with the following keys:**
    - 'sql_query': The executable SQL query as a string
    - 'intent': A clear description of the user's intent
    - 'user_query': The original user query
    **Do not return markdown, python dict, or any explanation. Only output the JSON.**
    """,
            agent=sql_generator,
            expected_output="""A dictionary containing:
    1. 'sql_query': The executable SQL query as a string
    2. 'intent': A clear description of the user's intent
    3. 'user_query': The original user query""",
        )

    # Task 2: Data Retrieval Task - Retrieve filtered data
    data_retrieval_task = Task(
        description=f"""
        Using the DATA RETRIEVAL from the sql_generator execute query using the database_query_tool with save_csv=True
        Use the temp_dir path provided in the inputs: "{temp_dir}"
        
        If you need data from multiple tables:
        - Use '_customers' suffix for Loan_Customer_Summary data
        - Use '_payments' suffix for Loan_Payment_Summary data
        
        For each query, provide:
        1. The SQL query used
        2. The number of rows retrieved
        3. The path to the saved CSV file
        
        This information will be passed to the Code Generator agent.
        """,
        agent=data_retriever,
        expected_output="SQL queries, row counts, and CSV file paths",
        context=[sql_generator_task],
    )
    
    # Create the crew - start with just planning and data retrieval
    crew = Crew(
        agents=[sql_generator,data_retriever],
        tasks=[sql_generator_task,data_retrieval_task],
        verbose=True,
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
    
    # Get schema info first
    # customers_schema_str, payments_schema_str = await get_schema_info()
    schema=load_schema()
    # Then create and run the crew
    result = create_analysis_crew(user_query, llm, schema)
    return result


if __name__ == "__main__":
    load_dotenv()
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY", ''))
    user_query = "Analyze loan distribution by customer type and branch"
    
    # Create and run the event loop properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the analysis
        result = loop.run_until_complete(run_analysis(user_query))
        print(result)
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
    finally:
        # Clean up resources
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        print("Analysis completed and resources cleaned up")
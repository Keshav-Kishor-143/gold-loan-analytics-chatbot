
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

# Import the correct path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crew.db_tools import db_query_tool
from connection.dbConnection import execute_query


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
    temp_dir = Path('./temp_dir')
    
    code_interpreter = CodeInterpreterTool(unsafe_mode=True)
    
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
    goal='Retrieve filtered data from database and save to CSV files',
    backstory="""
        You are a database expert who specializes in efficient data retrieval.
        You take a plan and translate it into SQL queries that filter data at a basic level.
        You ensure only the necessary data is retrieved and saved to CSV files for further analysis.
        You return metadata about the saved data, not the full dataset itself.
    """,
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[db_query_tool]
    )
    
    # 3. Code Generator Agent - Creates detailed Python code
    code_generator = Agent(
        role='Data Analyst & Code Generator',
        goal='Generate analytical Python code that discovers insights and patterns from data',
        backstory="""
        You are an expert data analyst and Python programmer who specializes in discovering hidden patterns and trends.
        You have a deep understanding of statistical analysis, data visualization, and machine learning techniques.
        Your specialty is creating clear, well-structured code that produces actionable intelligence and presents
        findings in an easy-to-understand tabular format.
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    # 4. Code Executor Agent - Executes the generated code
    code_executor = Agent(
        role='Analytics Code Executor',
        goal='Execute analytical code and present insights in a structured format',
        backstory="""
        You are a data science execution specialist who excels at running analytical code and interpreting results.
        You have deep expertise in pandas, numpy, seaborn, and other data analysis libraries.
        Your specialty is executing code and presenting insights in a clear, actionable, and
        professional format. You ensure that all insights are properly formatted and directly address the
        business question at hand.
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[code_interpreter]
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
        description=f"""
        Using the DATA RETRIEVAL PLAN from the Planner (especially STEP 1):
        
        1. Generate SQL queries that implement the basic filtering described in STEP 1
        2. Execute each query using the database_query_tool with save_csv=True
        3. Use the temp_dir path provided in the inputs: "{temp_dir}"
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
        agent=data_retriever,
        expected_output="SQL queries, metadata, and CSV file paths",
        context=[planning_task],
    )
    
    # Task 3: Code Generation Task - Creates the execution plan
    code_generation_task = Task(
        description=f"""
        As a data analyst, your task is to generate Python code that analyzes data to answer: "{user_query}"
        STEP 1: UNDERSTAND THE AVAILABLE DATA
        - First, dynamically inspect the CSV files in the temp_dir that were created by the Data Retriever
        - Examine the metadata provided by the Data Retriever (file paths, column names, data types, sample data)
        - Understand the structure and content of each dataset
        STEP 2: DEVELOP ANALYTICAL APPROACH
        - Based on the user query, determine the appropriate analytical techniques to apply
        - Consider statistical methods, time series analysis, grouping, aggregation, or correlation analysis
        - Write code to identify hidden patterns, trends, anomalies, or relationships in the data
        STEP 3: GENERATE COMPREHENSIVE PYTHON CODE
        Your code should:
        
        1. DATA LOADING AND EXPLORATION:
        - Load the CSV files from the paths provided by the Data Retriever(if multiple files, load them separately and understand their structure/columns before combining them)
        - Include basic exploratory data analysis to understand the data
        - Handle data cleaning, missing values, and type conversions
        - Combine the datasets carefully and only if necessary(use LoanId as common key)
        - This is sample data 
        
        2. ADVANCED ANALYSIS:
        - Apply appropriate statistical methods to uncover insights
        - Use techniques like aggregation, pivoting, correlation, or time-based analysis
        - Identify patterns, trends, outliers, or other significant findings
        - Calculate relevant metrics and KPIs that address the user query
        
        3. VISUALIZATION AND PRESENTATION:
        - Generate summary tables that present results in a clear, structured format
        - Ensure all tables  have proper titles and labels
        
        4. INSIGHTS AND INTERPRETATION:
        - Include code that interprets the results and extracts business insights
        - Highlight the most important findings related to the user query
        - Format the final output as a well-structured table with supporting visualizations
        
        5. CODE QUALITY:
        - Write clean, well-commented code with clear section headers
        - Include error handling for robustness
        - Optimize for performance with potentially large datasets
        - Make your code modular and easy to understand
        
        IMPORTANT: 
        - Your code must be complete and ready to execute
        - Use ONLY standard Python libraries that are already installed: pandas, numpy
        - DO NOT use libraries like 'glob' or 'os' for file operations - use pathlib instead
        - Ensure your code has proper syntax with no unmatched brackets or parentheses
        - All file paths should use the Path object from pathlib
        - The code must directly address the user's query with analytical insights presented in a tabular format
        """,
        agent=code_generator,
        expected_output="Complete Python code for advanced data analysis",
        context=[planning_task, data_retrieval_task],
    )
    
    # Task 4: Code Execution Task - Execute the code
    code_execution_task = Task(
        description=f"""
        Execute the analytical Python code provided by the Data Analyst and present the results:
        
        1. EXECUTION:
        - Run the provided Python code using the code_interpreter tool
        - The code will load data from CSV files in the temp_dir
        - If any errors occur, troubleshoot and fix them, then re-run the code
        
        2. RESULTS CAPTURE:
        - Capture all outputs including:
            * Numerical results and statistics
            * Data tables and summary information
            * Visualizations (as descriptions or base64-encoded images)
            * Analytical insights and interpretations
        - Save visualizations to the {temp_dir} with descriptive filenames
        
        3. PRESENTATION:
        - Format the results in a professional, easy-to-understand structure
        - Ensure tables are properly formatted with clear headers
        - Make sure all monetary values are formatted with the Indian Rupee symbol (₹)
        - Present visualizations with clear titles and descriptions
        
        4. INSIGHTS SUMMARY:
        - Highlight the key findings that directly answer the user query: "{user_query}"
        - Emphasize any discovered patterns, trends, or anomalies
        - Provide context and interpretation for the analytical results
        - Summarize the most important insights in a concise, business-friendly manner
        
        Return the complete execution results with a focus on presenting actionable insights
        in a tabular format that directly addresses the user's query.
        """,
        agent=code_executor,
        expected_output="Execution results with tabular insights and visualizations",
        context=[code_generation_task],
    )
    
    # Create the crew - start with just planning and data retrieval
    crew = Crew(
        agents=[planner, data_retriever, code_generator, code_executor],
        tasks=[planning_task, data_retrieval_task, code_generation_task, code_execution_task],
        verbose=True
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
    customers_schema_str, payments_schema_str = await get_schema_info()
    
    # Then create and run the crew
    result = create_analysis_crew(user_query, llm, customers_schema_str, payments_schema_str)
    return result


if __name__ == "__main__":
    load_dotenv()
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY", ''))
    user_query = "What type of customers take loan at what frequency"
    
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
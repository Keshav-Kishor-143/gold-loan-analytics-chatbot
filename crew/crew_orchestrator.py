import json
import os
from pathlib import Path
from textwrap import dedent
import asyncio
from crewai import Crew, Agent, Task
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from db_tools import db_query_tool

def load_schema():
    with open('schema/views_schema.json', 'r') as file:
        schema = json.load(file)
        return schema

def create_analysis_crew(user_query, llm, schema):
    temp_dir = Path('./temp_dir')
    os.makedirs(temp_dir, exist_ok=True)
    
    sql_generator = Agent(
        role='SQL Generator',
        goal='Generate optimal SQL queries based on user intent and database schema',
        backstory=dedent("""
            You are an expert Data Scientist who specializes in SQL query generation.
            You analyze user requirements and create precise SQL queries that select
            only the necessary columns from the database schema.
        """),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[]
    )
    # 2. Data Retriever Agent - Retrieves filtered data based on plan
    data_retriever = Agent(
    role='Data Retriever',
    goal='Retrieve filtered data from database and save to CSV files',
    backstory="""
        You are a database expert who specializes in efficient data retrieval.
        You take SQL query from sql_generator and retrieve filtered data from the database.
        You ensure only the necessary data is retrieved and saved to CSV files for further analysis.
        You return metadata about the saved data, not the full dataset itself.
    """,
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[db_query_tool]
    )
    
    sql_generator_task = Task(
        description=dedent(f"""
            Generate a SQL query based on this user request: "{user_query}"
            
            Available tables and their schema {schema}
            
            Requirements:
            1. Select only necessary columns
            2. Focus on columns directly related to the user's request
            3. Keep the query simple - avoid complex joins or aggregations
            
            Return a JSON object with these exact keys:
            - sql_query: The SQL SELECT statement
            - intent: Brief description of what the query aims to analyze
            - user_query: The original user request
        """),
        agent=sql_generator,
        expected_output="sql query",
    )

    # Task 2: Data Retrieval Task - Retrieve filtered data
    data_retrieval_task = Task(
        description=f"""
                
        1. Use the SQL query provided by the SQL Generator
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
        context=[sql_generator_task],
    )
    crew = Crew(
        agents=[sql_generator,data_retriever],
        tasks=[sql_generator_task,data_retrieval_task],
        verbose=True
    )
    
    return crew.kickoff()

async def run_analysis(user_query):
    load_dotenv()
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
    schema = load_schema()
    # print(schema)
    # return schema
    return create_analysis_crew(user_query, llm, schema)

if __name__ == "__main__":
    user_query = "Analyze loan distribution by customer type and branch"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(run_analysis(user_query))
        print(result)
    finally:
        loop.close()

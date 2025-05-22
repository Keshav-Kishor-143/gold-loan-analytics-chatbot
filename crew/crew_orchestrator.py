import json
import os
from pathlib import Path
from textwrap import dedent
import asyncio
from crewai import Crew, Agent, Task
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from db_tools import db_query_tool
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def load_schema():
    with open('schema/views_schema.json', 'r') as file:
        schema = json.load(file)
        return schema

def create_analysis_crew(user_query, llm, schema):
    temp_dir = Path('temp_dir')
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
    verbose=False,
    allow_delegation=False,
    llm=llm,
    tools=[db_query_tool]
    )
    
    trend_analyzer = Agent(
    role="Exploratory Data Analyst",
    goal="Perform a comprehensive exploratory data analysis on the input dataset",
    backstory="You are an expert data scientist skilled in quickly generating insights from data using statistical techniques, visualizations, and automated reporting.",
    llm=llm,
    memory=True,
)


    # trend_analyzer = Agent(
    #     role='Data Analyst',
    #     goal='Analyze data and present clear, actionable insights with actual values',
    #     backstory=dedent("""
    #         You are an expert data analyst who excels at:
    #         1. Understanding business data
    #         2. Finding meaningful patterns
    #         3. Presenting clear results with actual values
    #         4. Providing practical insights
            
    #         For every analysis, you:
    #         - Read and understand the data thoroughly
    #         - Focus on relevant metrics for the query
    #         - Show actual numbers and percentages
    #         - Explain what the numbers mean
    #     """),
    #     verbose=False,
    #     allow_delegation=False,
    #     llm=llm,
    #     tools=[]
    # )
    
    sql_generator_task = Task(
        description=dedent(f"""
            Generate a SQL query based on this user request: "{user_query}"
            
            Available tables and their schema {schema}
            
            Requirements:
            1. Select only necessary columns
            2. Focus on columns directly related to the user's request
            3. Keep the query simple - avoid complex joins or aggregations
            4. 
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

#     trend_analysis_task = Task(
#     description=(
#         "Conduct a complete exploratory data analysis (EDA) on the avoidable days dataset. "
#         "Use the tools in the following strict sequence to ensure consistent insights:\n\n"
#         "1. **Use `load_data_tool`** to load the dataset.\n"
#         "2. **Use `missing_value_analysis`** to inspect and summarize missing data.\n"
#         "3. **Use `univariate_analysis`** to analyze distributions of each numeric column.\n"
#         "4. **Use `correlation_analysis`** to find relationships between numeric variables.\n"
#         "5. **Use `outlier_detection`** to detect and summarize outliers.\n"
#         "6. **Use `target_relationship`** to analyze relationships between predictors and the target variable.\n"
#         "7. **Use `generate_visualizations`** to create supporting charts.\n"
#         "8. **Use `export_report`** to generate a final PDF or HTML report of all findings. "
#         "The report will include a summary of key insights and visualizations.\n\n"
#         "Strict adherence to the tool sequence is required. Summarize each tool's output where applicable."
#     ),
#     expected_output=(
#         """
#         1. A comprehensive summary of EDA insights and the path to the saved final report.
#         2. A PDF or HTML report generated by `export_report_tool` containing all findings, including missing values, univariate analysis,
#         correlation analysis, outlier detection, target relationships, and visualizations.
#         3. Ensure the report is well-structured with clear headings and sections for each analysis.
#         4. Include links to all generated visualizations within the report.
#         5. Ensure the report is saved in the specified output directory `./temp_dir`.
#         6. The report should be named `eda_report.pdf` or `eda_report.html` depending on the chosen format.
#         7. Print the path and status of the generated PDF or HTML file.
#         """
#     ),
#     agent=trend_analyzer
# )
    
    trend_analysis_task = Task(
        
        description=f"""
        Analyze the CSV files in: "{temp_dir}" to answer: "{user_query}"
        
        1. First, read and understand the data:
           - Load the CSV files
           - Check what columns are available
           - Understand the data types and values
        
        2. Then analyze based on the query:
           - Calculate relevant metrics
           - Find important patterns
           - Identify key insights
        
        3. Present results in clear tables with actual data:
        
           Data Summary:
           ```
           | Column Name | Data Type | Non-Null Count | Example Values |
           |------------|-----------|----------------|----------------|
           | [actual]   | [actual]  | [actual]       | [actual]       |
           ```
           
           Key Metrics:
           ```
           | Metric Name | Current Value | Previous Value | % Change |
           |------------|---------------|----------------|----------|
           | [actual]   | [actual]      | [actual]       | [actual] |
           ```
           
           Important Patterns:
           ```
           | Pattern Description | Supporting Numbers | Significance |
           |-------------------|-------------------|--------------|
           | [actual finding]  | [actual values]   | [actual %]   |
           ```
           
           Recommendations:
           ```
           | Finding | Supporting Data | Action Item |
           |---------|----------------|-------------|
           | [actual]| [actual values]| [actual]    |
           ```
        
        4. For each result:
           - Use actual values from the data
           - Include real percentages
           - Show supporting numbers
           - Explain what they mean
        
        Return the analysis in this structure:
        {{
            "data_summary": {{
                "columns": ["actual column names"],
                "rows": "actual number",
                "period": "actual dates"
            }},
            "metrics": [
                {{
                    "name": "actual metric name",
                    "value": "actual value",
                    "meaning": "what this means"
                }}
            ],
            "patterns": [
                {{
                    "type": "actual pattern found",
                    "evidence": "actual supporting values",
                    "impact": "actual impact"
                }}
            ],
            "actions": [
                {{
                    "insight": "actual finding",
                    "support": "actual numbers",
                    "recommendation": "specific action"
                }}
            ]
        }}
        
        Important:
        - Always use actual values from the data
        - Show real numbers and percentages
        - Provide context for what the numbers mean
        - Make findings specific to the user's query
        """,
        agent=trend_analyzer,
        expected_output="Analysis results with actual data values and clear explanations",
        context=[data_retrieval_task]
    )

    crew = Crew(
        agents=[sql_generator, data_retriever, trend_analyzer],
        tasks=[sql_generator_task, data_retrieval_task, trend_analysis_task],
        verbose=True,
        max_iterations=2
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

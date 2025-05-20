from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import pandas as pd
from textwrap import dedent
from pathlib import Path
from crewai_tools import CodeInterpreterTool

load_dotenv()

def create_data_retriever_agent(llm=None):
    return Agent(
        role='Data Retriever',
        goal='Load, verify and filter relevant data based on user query intent',
        backstory=dedent("""
            You are an expert data engineer specializing in data preparation and filtering.
            Your primary responsibility is to ensure data quality, proper loading, and
            precise filtering of datasets to match user requirements. You thoroughly verify
            data integrity before passing it to analysis stages.
        """),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        input_schema={
            "user_query": str,
            "csv_files": dict
        }
    )

def create_enhanced_code_generator_agent(llm=None):
    return Agent(
        role='Code Generator',
        goal='Generate comprehensive Python code for detailed data analysis',
        backstory=dedent("""
            You are an expert Python programmer specializing in data analysis and transformation.
            You excel at creating insightful analyses that go beyond basic aggregations.
            You know how to create meaningful comparisons, identify patterns, and provide
            actionable insights through intelligent data transformations.
            
            Your task is to generate a complete Python script that performs detailed analysis
            of loan distribution data. The script should include:
            1. Data loading and validation
            2. Multiple analysis perspectives (totals, averages, percentages)
            3. Statistical measures (mean, median, standard deviation)
            4. Trend analysis for date data
            5. Proper error handling and data validation
            6. Currency formatting with ₹ symbol
            7. Clear output formatting
            
            Return ONLY the Python script as a string, with no additional text or formatting.
        """),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        input_schema={
            "dataframe_info": dict,
            "sample_rows": list,
            "schema": dict,
            "query_intent": str
        }
    )

def create_code_executor_agent(llm=None):
    return Agent(
        role='Code Executor',
        goal='Execute and validate Python code for tabular data analysis',
        backstory=dedent("""
            You are a Python expert specializing in data analysis and code execution.
            You ensure that all code runs correctly, handles errors gracefully, and produces
            accurate tabular results. You have deep knowledge of pandas, numpy, and data
            manipulation techniques, and you excel at debugging and optimizing code to
            produce clear, insightful tables that answer business questions.
        """),
        verbose=True,
        allow_delegation=False,
        tools=[CodeInterpreterTool()],
        llm=llm,
        input_schema={
            "python_script": str,
            "required_packages": list
        }
    )

def create_response_formatter_agent(llm=None):
    return Agent(
        role='Response Formatter',
        goal='Format tabular analysis results into clear, professional HTML documents with business insights',
        backstory=dedent("""
            You are an expert in data presentation and business communication. You excel at
            transforming complex analytical results into clear, visually appealing HTML tables
            that highlight key insights. You have a strong background in financial reporting
            and know how to present tabular data insights that drive business decision-making.
        """),
        verbose=True,
        allow_delegation=False,
        tools=[],  # formatting with LLM only
        llm=llm,
        input_schema={
            "html_table": str,
            "raw_output": str,
            "query_intent": str
        }
    )

def create_retrieval_task(user_query: str, csv_files: dict, agent: Agent) -> Task:
    """
    Creates a retrieval Task for loading and filtering data based on user query.
    """
    # Convert absolute paths to relative paths
    relative_csv_files = {}
    for key, path in csv_files.items():
        if path.startswith('D:\\') or path.startswith('D:/'):
            parts = path.replace('\\', '/').split('gl_analytics_backend/')
            if len(parts) > 1:
                relative_csv_files[key] = parts[1]
        else:
            relative_csv_files[key] = path

    # Load and validate the primary CSV file
    primary_csv_path = relative_csv_files.get('loan_df')
    if not primary_csv_path:
        raise ValueError("Primary CSV file (loan_df) not found in csv_files")

    try:
        # Read the CSV file
        df = pd.read_csv(primary_csv_path)
        
        # Get actual data info
        columns = df.columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        sample_rows = df.head(5).to_dict('records')
        
        # Create schema with actual column descriptions
        schema = {}
        for col in columns:
            schema[col] = {
                "dtype": str(df[col].dtype),
                "description": f"Column containing {col.replace('_', ' ')} data"
            }
        
        # Create the expected output structure
        expected_output = {
            "dataframe_info": {
                "path": primary_csv_path,
                "columns": columns,
                "dtypes": dtypes
            },
            "sample_rows": sample_rows,
            "schema": schema,
            "query_intent": user_query.lower()
        }
        
        # Print the actual data for verification
        print("\nActual CSV Data:")
        print("="*80)
        print(f"File: {primary_csv_path}")
        print("\nColumns:", columns)
        print("\nSample Data:")
        print(df.head().to_string())
        print("="*80)

    except Exception as e:
        raise Exception(f"Error loading CSV file: {str(e)}")

    description = dedent(f"""
    TASK: Given the user query:
      "{user_query}"

    You receive a dictionary of CSV files to process:
    {json.dumps(relative_csv_files, indent=2)}

    Steps:
    1. Load and validate each CSV file:
       - Use pandas to read the CSV files using relative paths
       - Verify file existence and readability
       - Check for required columns
       - Handle any encoding or format issues
       - Convert data types appropriately
       - Handle missing values

    2. Analyze the data:
       - Understand the data structure
       - Identify relevant columns for the analysis
       - Check data quality and completeness
       - Prepare data for analysis

    3. Return exactly this JSON structure:
    {json.dumps(expected_output, indent=2)}

    Important:
    - Handle all errors gracefully
    - Validate data quality
    - Ensure proper data types
    - Handle missing values appropriately
    - Provide clear error messages if something goes wrong
    - Use relative paths for file access (e.g., 'csv/customer_summary.csv')
    """)

    return Task(
        description=description,
        agent=agent,
        expected_output = """
        {
            "type": "object",
            "required": ["dataframe_info", "sample_rows", "schema", "query_intent"],
            "properties": {
                "dataframe_info": {
                    "type": "object",
                    "required": ["path", "columns", "dtypes"],
                    "properties": {
                        "path": { "type": "string" },
                        "columns": { 
                            "type": "array",
                            "items": { "type": "string" }
                        },
                        "dtypes": {
                            "type": "object",
                            "additionalProperties": { "type": "string" }
                        }
                    }
                },
                "sample_rows": {
                    "type": "array",
                    "items": { "type": "object" }
                },
                "schema": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "dtype": { "type": "string" },
                            "description": { "type": "string" }
                        },
                        "required": ["dtype"]
                    }
                },
                "query_intent": { "type": "string" }
            }
        }
        """
    )

def create_enhanced_code_generation_task(agent: Agent) -> Task:
    description = dedent("""
    Generate a comprehensive Python script to analyze loan distribution by customer type and branch.
    
    The script should:
    1. Load and validate the data:
       import pandas as pd
       import numpy as np
       from pathlib import Path
       
       # Load the DataFrame
       df = pd.read_csv("csv/customer_summary.csv")
       print("Loaded DataFrame successfully")
       
       # Print sample data
       print("SAMPLE_DATA_BEGIN")
       print(df.head(5).to_json(orient="records"))
       print("SAMPLE_DATA_END")
    
    2. Perform comprehensive analysis:
       - Calculate total loan amounts by customer type and branch
       - Compute average loan amounts
       - Calculate percentage distributions
       - Add statistical measures (mean, median, std dev)
       - Format currency values with ₹ symbol
       - Handle missing values appropriately
       - Convert data types as needed
    
    3. Generate HTML output:
       result_html = results_df.to_html(index=False, classes="data-table", border=0)
       print("RESULT_HTML_BEGIN")
       print(result_html)
       print("RESULT_HTML_END")
    
    4. Add verification checks:
       print("\nVERIFICATION:")
       print("Columns:", df.columns.tolist())
       print("Total records:", len(df))
    
    Return ONLY the complete Python script as a string, with no additional text or formatting.
    """)

    return Task(
        description=description,
        agent=agent,
        expected_output = """
        {
            "type": "string",
            "description": "Complete Python script for loan distribution analysis"
        }
        """
    )

def create_execution_task(agent: Agent) -> Task:
    description = dedent("""
    Execute the Python analysis script and extract both sample data and the HTML table.

    Steps:
    1. First, ensure all required packages are installed:
       - Install pandas, numpy, and other required packages
       - Handle any installation errors gracefully

    2. Execute the provided python_script:
       - Capture all stdout and stderr
       - Handle any execution errors gracefully
       - Ensure proper resource cleanup
       - Set a reasonable timeout (5 minutes)

    3. From the execution output, extract:
       - sample_rows: JSON between "SAMPLE_DATA_BEGIN" and "SAMPLE_DATA_END"
       - html_table: HTML between "RESULT_HTML_BEGIN" and "RESULT_HTML_END"

    4. Return exactly this JSON structure:
       {
         "raw_output": "<complete stdout/stderr>",
         "sample_rows": [ ... parsed JSON array ... ],
         "html_table": "<extracted HTML or empty string>"
       }

    Important:
    - Handle all errors gracefully
    - Ensure proper resource cleanup
    - Validate the output format
    - Provide clear error messages if something goes wrong
    """)

    return Task(
        description=description,
        agent=agent,
        expected_output = """
        {
            "type": "object",
            "required": ["raw_output", "sample_rows", "html_table"],
            "properties": {
                "raw_output": { 
                    "type": "string",
                    "description": "Complete stdout and stderr from execution"
                },
                "sample_rows": { 
                    "type": "array",
                    "items": { "type": "object" },
                    "description": "Sample data rows from execution"
                },
                "html_table": { 
                    "type": "string",
                    "description": "HTML table output from execution"
                }
            },
            "additionalProperties": false
        }
        """
    )

def create_formatting_task(agent: Agent) -> Task:
    description = dedent("""
    Create a professional HTML document from the analysis results.

    Steps:
    1. Analyze the input:
       - Review the html_table for data patterns
       - Check raw_output for any errors or warnings
       - Understand the query_intent for context

    2. Generate a complete HTML5 document with:
       - Proper DOCTYPE and meta tags
       - Responsive CSS styling
       - Clear headings and structure
       - The provided table (or fallback if empty)
       - Key insights section
       - Error handling if needed

    3. Return exactly this JSON structure:
       {
         "html_document": "<!DOCTYPE html>..."
       }

    Important:
    - Ensure valid HTML5 structure
    - Include responsive design
    - Handle empty or invalid tables gracefully
    - Provide clear error messages if needed
    - Focus on business insights
    - Use professional styling
    """)

    return Task(
        description=description,
        agent=agent,
        expected_output = """
        {
            "type": "object",
            "required": ["html_document"],
            "properties": {
                "html_document": { "type": "string" }
            },
            "additionalProperties": false
        }
        """
    )

def test_crew_agents():
    """Test the data retriever, code generator, code executor, and response formatter agents in sequence."""
    try:
        # Initialize the LLM with temperature=0 for more deterministic output
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            temperature=0
        )

        # Create the agents
        data_retriever = create_data_retriever_agent(llm=llm)
        code_generator = create_enhanced_code_generator_agent(llm=llm)
        code_executor = create_code_executor_agent(llm=llm)
        response_formatter = create_response_formatter_agent(llm=llm)

        # Sample query and CSV files
        user_query = "What is the frequency and trend of loan disbursement by customer types and branch?"
        csv_files = {
            "loan_df": "csv/customer_summary.csv",
            "payment_df": "csv/payment_summary.csv"
        }

        # Verify CSV files exist
        missing = [p for p in csv_files.values() if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Data files not found: {', '.join(missing)}")

        # Create the tasks
        retrieval_task = create_retrieval_task(user_query, csv_files, data_retriever)
        code_generation_task = create_enhanced_code_generation_task(code_generator)
        execution_task = create_execution_task(code_executor)
        formatting_task = create_formatting_task(response_formatter)

        # Connect the tasks in sequence and pass required data
        code_generation_task.context = [retrieval_task]
        execution_task.context = [code_generation_task]
        formatting_task.context = [execution_task]

        # Create a crew with all four agents
        crew = Crew(
            agents=[data_retriever, code_generator, code_executor, response_formatter],
            tasks=[retrieval_task, code_generation_task, execution_task, formatting_task],
            llm=llm,
            verbose=True,
            process="sequential"
        )

        # Run the crew and get the result
        result = crew.kickoff(inputs={"user_query": user_query, "csv_files": csv_files})
        
        # Extract and print the formatted HTML document
        if hasattr(result, 'raw'):
            output = result.raw
            print("\n" + "="*80)
            print("FORMATTED HTML DOCUMENT")
            print("="*80)
            
            if isinstance(output, str):
                # Try to parse the string as JSON if it's wrapped in ```json
                if output.startswith("```json"):
                    try:
                        json_str = output.replace("```json", "").replace("```", "").strip()
                        output = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            
            if isinstance(output, dict):
                html_doc = output.get('html_document', '')
                if html_doc:
                    print("\nHTML Document:")
                    print("-"*20)
                    print(html_doc)
                else:
                    
                    print("\nNo HTML document found in output")
            else:
                print("\nUnexpected output format from formatter:")
                print(output)
        else:
            print("\nUnexpected result type:")
            print(type(result))

    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    test_crew_agents() 
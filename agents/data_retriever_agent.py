from crewai import Agent, Task
from textwrap import dedent
from crewai_tools import CodeInterpreterTool
import pandas as pd
import os
import json
from pathlib import Path

def validate_csv_path(path: str) -> Path:
    """Validate and return absolute path for CSV file"""
    # Convert to relative path if absolute path is provided
    if path.startswith('D:\\') or path.startswith('D:/'):
        # Extract just the relative part after 'gl_analytics_backend'
        parts = path.replace('\\', '/').split('gl_analytics_backend/')
        if len(parts) > 1:
            path = parts[1]
    
    # Ensure path uses forward slashes
    path = path.replace('\\', '/')
    abs_path = Path(path).resolve()
    
    if not abs_path.exists():
        raise FileNotFoundError(f"CSV file not found: {abs_path}")
    if not abs_path.is_file():
        raise ValueError(f"Path is not a file: {abs_path}")
    return abs_path

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
        tools=[],
        llm=llm,
        input_schema={
            "user_query": str,
            "csv_files": dict
        }
    )

def create_retrieval_task(user_query: str, csv_files: dict, agent: Agent) -> Task:
    """
    Creates a retrieval Task for loading and filtering data based on user query.
    
    Args:
        user_query: The analysis query from the user
        csv_files: Dict mapping DataFrame names to file paths
        agent: The agent that will execute the task
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

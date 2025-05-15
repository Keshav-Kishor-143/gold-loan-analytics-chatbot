from crewai import Agent, Task
from textwrap import dedent

def create_code_generator_agent(llm=None):
    return Agent(
        role='Code Generator',
        goal='Generate optimal Python code for data analysis based on user intent',
        backstory=dedent("""
            You are an expert Python programmer specializing in data analysis and transformation.
            You excel at translating analytical requirements into efficient, well-structured
            Python code. You have deep knowledge of pandas, numpy, and data manipulation libraries,
            and you know how to create insightful tabular analyses that answer specific business 
            questions through intelligent data transformations.
        """),
        verbose=False,
        allow_delegation=False,
        tools=[],  # no direct execution, only code emission
        llm=llm,
        input_schema={
            "dataframe_info": dict,
            "sample_rows": list,
            "schema": dict,
            "query_intent": str
        }
    )

def create_code_generation_task(agent: Agent) -> Task:
    """
    Task to generate analysis code.

    INPUT JSON:
      {
        "dataframe_info": { "path": str, "columns": list, "dtypes": dict },
        "sample_rows": [ {...}, ... ],
        "schema": { col: { "dtype": "...", ... }, … },
        "query_intent": "..."
      }

    OUTPUT JSON:
      {
        "python_script": "<full_source_code>",
        "required_packages": ["pandas", ...]
      }
    """
    description = dedent("""
    You receive exactly one JSON input with these keys:
      • dataframe_info: object containing path, columns, and dtypes
      • sample_rows: array of up to 5 JSON objects (for context)
      • schema: object describing each column's dtype
      • query_intent: the user's analytical goal

    Your output MUST be a single JSON object (no extra text) with:

    1. python_script (string):
       - Start by loading the DataFrame:
         ```python
         import pandas as pd
         from pathlib import Path
         
         # Load the DataFrame
         df = pd.read_csv(r"<dataframe_info.path>")
         print("Loaded DataFrame successfully")
         ```
       - **Immediately** print the first 5 rows to stdout for verification:
         ```python
         print("SAMPLE_DATA_BEGIN")
         print(df.head(5).to_json(orient="records"))
         print("SAMPLE_DATA_END")
         ```
       - Add analysis code using `df` to satisfy `query_intent`:
         • No plotting—only pandas transformations/aggregations
         • Format any currency columns with the ₹ symbol
         • Include proper error handling and data validation
         • Handle missing values appropriately
         • Convert data types as needed
       - Convert your main results DataFrame (named `results_df`) to HTML:
         ```python
         result_html = results_df.to_html(index=False, classes="data-table", border=0)
         print("RESULT_HTML_BEGIN")
         print(result_html)
         print("RESULT_HTML_END")
         ```
       - Append verification checks at the end, e.g.:
         ```python
         print("\\nVERIFICATION:")
         print("Columns:", df.columns.tolist())
         ```
    2. required_packages (array of strings):
       - List any Python packages your script depends on, e.g. ["pandas","numpy"].

    **Do not** emit any explanatory text—only the JSON below.
    """)

    return Task(
        description=description,
        agent=agent,
        expected_output = """
                        {
                        "type": "object",
                        "required": ["python_script", "required_packages"],
                        "properties": {
                            "python_script": { 
                                "type": "string",
                                "description": "Complete Python script to execute"
                            },
                            "required_packages": {
                                "type": "array",
                                "items": { 
                                    "type": "string",
                                    "description": "Package name to install"
                                },
                                "description": "List of Python packages required for execution"
                            }
                        },
                        "additionalProperties": false
                        }
                        """
    )

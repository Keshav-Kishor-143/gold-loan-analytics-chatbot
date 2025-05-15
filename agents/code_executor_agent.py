from crewai import Agent, Task
from textwrap import dedent
from crewai_tools import CodeInterpreterTool
import tempfile
import os
import subprocess
import sys

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
        verbose=False,
        allow_delegation=False,
        tools=[CodeInterpreterTool()],
        llm=llm,
        input_schema={
            "python_script": str,
            "required_packages": list
        }
    )

def create_execution_task(agent: Agent) -> Task:
    """
    Task to execute the Python analysis script and extract both sample data and the HTML table.

    INPUT JSON:
      {
        "python_script": "<full Python code to run>",
        "required_packages": ["pandas", ...]
      }

    OUTPUT JSON:
      {
        "raw_output": "<complete stdout and stderr from execution>",
        "sample_rows": [ {...}, ... ],
        "html_table": "<HTML table string between markers or empty string>"
      }
    """
    description = dedent("""
    You receive exactly one JSON input with these keys:
      • python_script: the complete code to run (string)
      • required_packages: list of required Python packages (array of strings)

    Steps:
    1. First, ensure all required packages are installed using the CodeInterpreterTool:
       - Use the provided required_packages list from the input
       - Install each package using pip
       - Handle any installation errors gracefully

    2. Execute the provided python_script using CodeInterpreterTool:
       - Capture all stdout and stderr
       - Handle any execution errors gracefully
       - Ensure proper resource cleanup
       - Set a reasonable timeout (e.g., 5 minutes)

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

def execute_code(self, code: str) -> str:
    """Execute the generated Python code with proper error handling."""
    try:
        # Create a temporary directory for execution
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary Python file
            script_path = os.path.join(temp_dir, "analysis_script.py")
            with open(script_path, "w") as f:
                f.write(code)
            
            # Install required packages
            required_packages = [
                "pandas",
                "numpy",
                "matplotlib",
                "seaborn"
            ]
            
            for package in required_packages:
                try:
                    subprocess.check_call([
                        sys.executable, 
                        "-m", 
                        "pip", 
                        "install", 
                        package
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError as e:
                    print(f"Warning: Failed to install {package}: {str(e)}")
            
            # Execute the script with timeout
            try:
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5-minute timeout
                )
                
                if result.returncode != 0:
                    error_msg = f"Script execution failed with return code {result.returncode}"
                    if result.stderr:
                        error_msg += f"\nError output:\n{result.stderr}"
                    return error_msg
                
                return result.stdout
                
            except subprocess.TimeoutExpired:
                return "Error: Script execution timed out after 5 minutes"
            except Exception as e:
                return f"Error executing script: {str(e)}"
                    
    except Exception as e:
        return f"Error in code execution: {str(e)}"

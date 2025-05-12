from crewai import Agent, Task
from textwrap import dedent

def create_code_executor_agent():
    return Agent(
        role='Code Executor',
        goal='Execute and validate Python code for data analysis',
        backstory=dedent("""
            You are a Python expert specializing in data analysis and code execution.
            You ensure that all code runs correctly, handles errors gracefully, and produces
            accurate results. You have deep knowledge of pandas, numpy, and data visualization
            libraries, and you excel at debugging and optimizing code.
        """),
        verbose=True,
        allow_delegation=False,
        tools=[]  # Add any specific tools needed for code execution
    )

def create_execution_task(agent):
    """Create the code execution task"""
    return Task(
        description="""
        Execute the Python code provided by the Code Generator agent:
        
        1. Run the code exactly as provided
        2. Capture all outputs, including:
           - Computed values and statistics
           - Generated visualizations
           - Any printed results
        
        3. If any errors occur:
           - Identify the root cause
           - Fix the code if possible
           - Document the issue if not fixable
        
        4. Ensure the execution results directly address the user's original query
        
        Provide the complete execution results to the Response Formatter agent.
        """,
        agent=agent,
        expected_output="Executed code with validated results."
    ) 
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
        allow_delegation=True,
        tools=[]  # Add any specific tools needed for code execution
    )

def create_execution_task(agent):
    """Create the code execution task"""
    return Task(
        description="""
        Execute the analysis code and validate the results.
        Ensure all code runs without errors and produces accurate results.
        Handle any data quality issues or edge cases appropriately.
        """,
        agent=agent,
        expected_output="Executed code with validated results."
    ) 
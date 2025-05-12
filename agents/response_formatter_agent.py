from crewai import Agent, Task
from textwrap import dedent

def create_response_formatter_agent():
    return Agent(
        role='Response Formatter',
        goal='Format analysis results into clear, professional reports',
        backstory=dedent("""
            You are an expert in technical writing and data presentation.
            You excel at transforming complex analysis results into clear, well-structured
            reports that are easy to understand and actionable. You have a strong background
            in financial reporting and know how to present data insights effectively.
        """),
        verbose=True,
        allow_delegation=True,
        tools=[]  # Add any specific tools needed for response formatting
    )

def create_formatting_task(agent):
    """Create the response formatting task"""
    return Task(
        description="""
        Format the analysis results into a clear, professional report.
        Include:
        1. Executive summary
        2. Key findings
        3. Supporting data and visualizations
        4. Recommendations
        5. Next steps
        """,
        agent=agent,
        expected_output="Formatted analysis report."
    ) 
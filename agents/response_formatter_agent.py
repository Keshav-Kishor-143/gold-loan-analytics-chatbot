from crewai import Agent, Task
from textwrap import dedent

def create_response_formatter_agent():
    return Agent(
        role='Response Formatter',
        goal='Format analysis results into clear, professional HTML tables with business insights',
        backstory=dedent("""
            You are an expert in data presentation and business communication. You excel at
            transforming complex analytical results into clear, visually appealing HTML tables
            that highlight key insights. You have a strong background in financial reporting 
            and know how to present data insights that drive business decision-making.
        """),
        verbose=True,
        allow_delegation=False,
        tools=[]  # Add any specific tools needed for response formatting
    )

def create_formatting_task(agent):
    """Create the response formatting task"""
    return Task(
        description="""
        Format the analysis results from the Code Executor into a professional HTML table:
        
        1. Extract the key insights from the analysis results
        
        2. Create a well-structured HTML table that:
           - Has a clear header with the analysis title
           - Presents data in a logical, easy-to-read format
           - Uses appropriate styling for readability
           - Highlights the most important findings
        
        3. Include a brief business insight summary below the table that:
           - Explains the key takeaways in business terms
           - Points out trends, patterns, or anomalies
           - Suggests potential business implications
        
        The output should be in proper HTML format that can be directly embedded in a web page,
        with proper styling for the table and appropriate formatting for the business insight.
        """,
        agent=agent,
        expected_output="Formatted HTML table with business insights."
    ) 
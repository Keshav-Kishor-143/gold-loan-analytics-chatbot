from crewai import Agent, Task
from textwrap import dedent

def create_data_analyst_agent():
    return Agent(
        role='Data Analyst',
        goal='Analyze loan data to provide actionable insights and recommendations',
        backstory=dedent("""
            You are an expert data analyst specializing in financial data analysis.
            You have extensive experience in analyzing loan portfolios, identifying patterns,
            and providing strategic recommendations. You excel at using Python for data analysis
            and creating clear, actionable insights from complex datasets.
        """),
        verbose=True,
        allow_delegation=True,
        tools=[]  # Add any specific tools needed for data analysis
    )

def create_analysis_task(user_query, loading_instructions, agent):
    """Create the analysis task for the data analyst"""
    return Task(
        description=f"""
        Analyze the loan data files to answer:

        {user_query}

        Start by loading the available datasets with pandas:
        ```python
        import pandas as pd
        {loading_instructions}
        ```
        
        The loan_df contains customer loan information from the customer_summary.csv file.
        If your analysis requires merging datasets, you can join them on common keys like LoanId.
        
        Write analytical code to directly answer the query. Format all monetary values with the Indian Rupee symbol (₹).
        
        Provide a complete analysis answering the query with:
        1. Quantitative insights based on actual data
        2. Visualizations if helpful
        3. Business implications
        4. Recommendations
        """,
        agent=agent,
        expected_output="Data analysis with insights and recommendations."
    ) 
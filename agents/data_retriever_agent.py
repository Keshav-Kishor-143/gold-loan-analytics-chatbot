from crewai import Agent, Task
from textwrap import dedent

def create_data_retriever_agent():
    return Agent(
        role='Data Retriever',
        goal='Retrieve and filter relevant data based on user query intent',
        backstory=dedent("""
            You are an expert in data management and preprocessing. You specialize in 
            understanding user intent from natural language queries and retrieving only
            the most relevant data needed for analysis. You excel at filtering large datasets
            to provide focused, relevant information that directly addresses the user's needs.
        """),
        verbose=True,
        allow_delegation=False,
        tools=[]  # Add any specific tools needed for data retrieval
    )

def create_retrieval_task(user_query, loading_instructions, agent):
    """Create the data retrieval task"""
    return Task(
        description=f"""
        Based on the following user query:
        
        {user_query}
        1. Load the data using:
        ```python
        import pandas as pd
        {loading_instructions}
        ```
        2. Understand the intent of the user's query and identify what data is relevant.
        3. Filter the dataframes to include only the columns and rows that are directly relevant 
           to answering the query.
        4. Clean and preprocess the data as needed (handling missing values, data type conversions, etc.).
        5. Provide:
           - A concise description of what the user is asking for (the query intent)
           - The filtered dataframe(s) with only the relevant data
           - Any context about the data that would be helpful for analysis
        """,
        agent=agent,
        expected_output="Filtered dataframe with relevant data and query intent explanation."
    ) 
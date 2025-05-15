from crewai import Agent, Task
from textwrap import dedent

def create_code_generator_agent():
    return Agent(
        role='Code Generator',
        goal='Generate optimal Python code for data analysis based on user intent',
        backstory=dedent("""
            You are an expert Python programmer specializing in data analysis and visualization.
            You excel at translating analytical requirements into efficient, well-structured
            Python code. You have deep knowledge of pandas, numpy, matplotlib, seaborn and other
            data analysis libraries, and you know how to create insightful analyses that answer
            specific business questions.
        """),
        verbose=True,
        allow_delegation=False,
        tools=[]  # Add any specific tools needed for code generation
    )

def create_code_generation_task(intent,table_schema,agent):
    """Create the code generation task"""
    return Task(
        description="""
        Based on the provided filtered dataframe and query intent:
        
        1. Generate efficient Python code that performs a deep analysis addressing the user's query.
        
        2. Your code should:
           - Use the dataframe provided by the Data Retriever agent
           - Apply appropriate statistical methods, aggregations, and calculations
           - Create relevant visualizations if helpful (using matplotlib or seaborn)
           - Format all monetary values with the Indian Rupee symbol (₹)
        
        3. Make sure the code:
           - Is clean, well-structured, and commented
           - Handles potential errors gracefully
           - Produces clear, actionable insights
           - Is optimized for performance
        
        4. If appropriate, include code to generate insightful visualizations that help answer the query.
        
        You will receive a dataframe and query intent from the Data Retriever agent. Use these to 
        generate Python code that will be executed by the Code Executor agent.
        """,
        agent=agent,
        expected_output="Python code for data analysis addressing the user's query."
    ) 
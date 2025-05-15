import os
from crewai import Crew, Agent, Task
from textwrap import dedent
from crewai_tools import CodeInterpreterTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
import pandas as pd
 
def create_analysis_crew(user_query, loan_customers_df, loan_payment_df, code_interpreter, llm):
    """Create and configure the analysis crew with all necessary agents and tasks"""
   
    # Convert DataFrames to string representations
    loan_customers_head = loan_customers_df.head().to_string()
    loan_customers_dtypes = str(loan_customers_df.dtypes)
    loan_customers_shape = str(loan_customers_df.shape)
   
    loan_payment_head = loan_payment_df.head().to_string()
    loan_payment_dtypes = str(loan_payment_df.dtypes)
    loan_payment_shape = str(loan_payment_df.shape)
   
    # Code Generator Agent - Creates execution plan
    code_generator = Agent(
        role='Analysis Planner',
        goal='Create detailed execution plan for data analysis based on user query',
        backstory="""
            You are an expert data analyst who excels at planning complex data analyses.
            You examine data structures and create detailed execution plans that specify
            exactly what operations to perform, what columns to use, and what insights to extract.
            Your plans are precise, executable, and focused on answering the specific query.
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[code_interpreter]
    )
 
    # Code Executor Agent - Executes the plan
    code_executor = Agent(
        role='Plan Executor',
        goal='Execute analysis plans and return filtered DataFrame results',
        backstory="""
            You are a Python expert specializing in pandas execution. You take analysis
            plans and transform them into efficient code that produces exactly the results
            needed. You ensure all operations are performed correctly and return clean,
            filtered DataFrames that directly answer the user's query.
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[code_interpreter]
    )
   
    # Code Generation Task - Creates the execution plan
    code_generation_task = Task(
        description=f"""
        Based on the user query: "{user_query}"
       
        Examine the provided DataFrame information and create a detailed execution plan that:
       
        1. Specifies exactly what operations to perform (filtering, grouping, joining, etc.)
        2. Lists the exact columns to use from each DataFrame
        3. Details any calculations or transformations needed
        4. Outlines how to join the DataFrames using LoanId when necessary
        5. Describes what the final output should look like
       
        DataFrame Information:
       
        loan_customers_df shape: {loan_customers_shape}
        loan_customers_df head:
        {loan_customers_head}
       
        loan_customers_df dtypes:
        {loan_customers_dtypes}
       
        loan_payment_df shape: {loan_payment_shape}
        loan_payment_df head:
        {loan_payment_head}
       
        loan_payment_df dtypes:
        {loan_payment_dtypes}
       
        Your plan should be specific enough that someone else could implement it exactly.
        DO NOT write actual code - just create a clear, step-by-step execution plan.
       
        Return your plan as a structured document with numbered steps.
        """,
        agent=code_generator,
        expected_output="Detailed execution plan for data analysis"
    )
   
    # Code Execution Task - Executes the plan and returns DataFrame
    execution_task = Task(
        description="""
        Take the execution plan from the Analysis Planner and implement it using pandas.
       
        You need to load the data first with:
        ```python
        import pandas as pd
        loan_customers_df = pd.read_csv('csv/customer_summary.csv')
        loan_payment_df = pd.read_csv('csv/payment_summary.csv')
        ```
       
        Then:
        1. Follow the plan exactly, using the specified operations and columns
        2. Write clean, efficient pandas code that implements each step
        3. Join DataFrames using LoanId when needed
        4. Return the top 5 rows of the final filtered DataFrame
        5. Format any monetary values with the Indian Rupee symbol (₹)
       
        Your code should be lightweight and focused on producing exactly what's needed.
        Return only the final DataFrame head() that answers the user query.
        """,
        agent=code_executor,
        expected_output="DataFrame with top 5 rows answering the query",
        context=[code_generation_task]
    )
   
    # Create and return the crew with the correct task order
    crew = Crew(
        agents=[code_generator, code_executor],
        tasks=[code_generation_task, execution_task],
        verbose=True
    )
   
    # Execute the crew with string representations of the DataFrames
    result = crew.kickoff(
        inputs={
            "user_query": user_query
        }
    )
   
    return result
 
if __name__ == "__main__":
    code_interpreter = CodeInterpreterTool()  
    load_dotenv()
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY", ''))
    user_query = "Analyze loan distribution by customer type and branch"
    # print(loan_df.head())
    # loading_instructions = """
    try:
        loan_customers_df = pd.read_csv('csv/customer_summary.csv')
        loan_payment_df = pd.read_csv('csv/payment_summary.csv')
        print(f"Data loaded successfully. Customer DF shape: {loan_customers_df.shape}, Payment DF shape: {loan_payment_df.shape}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit(1)
    # """
    result = create_analysis_crew(user_query, loan_customers_df, loan_payment_df, code_interpreter,llm)
    print(result)
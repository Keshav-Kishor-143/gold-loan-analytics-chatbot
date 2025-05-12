from crewai import Crew, Agent, Task
from textwrap import dedent
from crewai_tools import CodeInterpreterTool

# Initialize the tool
code_interpreter = CodeInterpreterTool(code_execution_mode="unsafe")  

def create_analysis_crew():
    """Create and configure the analysis crew with all necessary agents and tasks"""
    # Data Retriever Agent (commented for future use)
    # data_retriever = Agent(
    #     role='Data Retriever',
    #     goal='Retrieve and filter relevant data based on user query intent',
    #     backstory=dedent("""
    #         You are an expert in data management and preprocessing. You specialize in 
    #         understanding user intent from natural language queries and retrieving only
    #         the most relevant data needed for analysis. You excel at filtering large datasets
    #         to provide focused, relevant information that directly addresses the user's needs.
    #     """),
    #     verbose=True,
    #     allow_delegation=False,
    #     tools=[]
    # )
    # Code Generator Agent
    code_generator = Agent(
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
        tools=[code_interpreter]
    )
  
    # Code Executor Agent
    code_executor = Agent(
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
        tools=[code_interpreter]
    )
    # Response Formatter Agent (commented out for now)
    # response_formatter = Agent(
    #     role='Response Formatter',
    #     goal='Format analysis results into clear, professional HTML tables with business insights',
    #     backstory=dedent("""
    #         You are an expert in data presentation and business communication. You excel at
    #         transforming complex analytical results into clear, visually appealing HTML tables
    #         that highlight key insights. You have a strong background in financial reporting 
    #         and know how to present data insights that drive business decision-making.
    #     """),
    #     verbose=True,
    #     allow_delegation=False,
    #     tools=[]
    # )
    # Data Retrieval Task (commented for future use)
    # retrieval_task = Task(
    #     description=f"""
    #     Based on the following user query:
    #     \n{user_query}
    #     1. Load the data using:
    #     ```python
    #     import pandas as pd
    #     {loading_instructions}
    #     ```
    #     2. Understand the intent of the user's query and identify what data is relevant.
    #     3. Filter the dataframes to include only the columns and rows that are directly relevant 
    #        to answering the query.
    #     4. Clean and preprocess the data as needed (handling missing values, data type conversions, etc.).
    #     5. Provide:
    #        - A concise description of what the user is asking for (the query intent)
    #        - The filtered dataframe(s) with only the relevant data
    #        - Any context about the data that would be helpful for analysis
    #     """,
    #     agent=data_retriever,
    #     expected_output="Filtered dataframe with relevant data and query intent explanation."
    # )
    # Code Generation Task
    code_generation_task = Task(
        description="""
        Based on the provided user_query and loading_instructions:
        User Query: {user_query}
        Loading Instructions: {loading_instructions}
        
        You are only allowed to use the following Python libraries for code generation and execution:
        - pandas
        - numpy
        - matplotlib
        - seaborn
        - scikit-learn
        
        Do NOT use any other libraries. If a required library is not in this list, do not use it.
        
        IMPORTANT: After loading the DataFrames, you MUST always programmatically inspect their columns and schema (e.g., using df.columns, df.info(), df.head()) before referencing any columns. Never assume or guess column names. Always dynamically adapt your code to the actual DataFrame structure.
        
        When you output code, ensure it is a valid, plain Python string (not JSON, not double-escaped, and with single backslashes for newlines). The code should be directly executable and not wrapped or escaped multiple times. Do not use quadruple or double backslashes for newlines.
        
        1. Perform exploratory data analysis (EDA) to understand the data and the user's query intent.
        2. Generate efficient Python code that performs a deep analysis addressing the user's query.
        3. Your code should:
           - Load the data using the provided loading_instructions
           - Dynamically inspect the DataFrame(s) to determine available columns and types
           - Filter and preprocess the data as needed to address the query
           - Apply appropriate statistical methods, aggregations, and calculations
           - Create relevant visualizations if helpful (using matplotlib or seaborn)
           - Format all monetary values with the Indian Rupee symbol (₹)
        4. Make sure the code:
           - Is clean, well-structured, and commented
           - Handles potential errors gracefully
           - Produces clear, actionable insights
           - Is optimized for performance
        5. If appropriate, include code to generate insightful visualizations that help answer the query.
        Output both the generated code and the filtered dataframe(s) relevant to the query.
        """,
        agent=code_generator,
        expected_output="A dictionary with keys 'code' (the generated Python code as a string) and 'filtered_df' (the relevant filtered dataframe(s)).",
        input_variables=["user_query", "loading_instructions"]
    )
    # Code Execution Task
    execution_task = Task(
        description="""
        Take the generated code and filtered dataframe(s) from the Code Generator agent:
        1. Execute the provided Python code using the filtered dataframe(s) as input
        2. Capture all outputs, including:
           - Computed values and statistics
           - Generated visualizations
           - Any printed results
        3. If any errors occur:
           - Identify the root cause
           - Fix the code if possible
           - Document the issue if not fixable
        4. Ensure the execution results directly address the user's original query
        Return the raw output of the code execution.
        """,
        agent=code_executor,
        expected_output="Raw output of the executed code."
    )
    # Response Formatting Task (commented out for now)
    # formatting_task = Task(
    #     description="""
    #     Format the analysis results from the Code Executor into a professional HTML table:
    #     1. Extract the key insights from the analysis results
    #     2. Create a well-structured HTML table that:
    #        - Has a clear header with the analysis title
    #        - Presents data in a logical, easy-to-read format
    #        - Uses appropriate styling for readability
    #        - Highlights the most important findings
    #     3. Include a brief business insight summary below the table that:
    #        - Explains the key takeaways in business terms
    #        - Points out trends, patterns, or anomalies
    #        - Suggests potential business implications
    #     The output should be in proper HTML format that can be directly embedded in a web page,
    #     with proper styling for the table and appropriate formatting for the business insight.
    #     """,
    #     agent=response_formatter,
    #     expected_output="Formatted HTML table with business insights."
    # )
    # Set up task dependencies
    # code_generation_task.context = [retrieval_task]  # For future use
    execution_task.context = [code_generation_task]
    # formatting_task.context = [execution_task]  # For future use
    # Create and return the crew with the correct task order
    return Crew(
        agents=[code_generator, code_executor],
        tasks=[code_generation_task, execution_task],
        verbose=True
    )

def run_analysis(user_query, loading_instructions):
    """Run the analysis using the crew and return both result and usage metrics"""
    crew = create_analysis_crew()
    try:
        # Pass user_query and loading_instructions as input to kickoff
        result = crew.kickoff(inputs={
            "user_query": user_query,
            "loading_instructions": loading_instructions
        })
        usage_metrics = crew.usage_metrics
        metrics = {
            "total_tokens": int(getattr(usage_metrics, 'total_tokens', 0)),
            "prompt_tokens": int(getattr(usage_metrics, 'prompt_tokens', 0)),
            "completion_tokens": int(getattr(usage_metrics, 'completion_tokens', 0)),
            "successful_requests": int(getattr(usage_metrics, 'successful_requests', 0))
        }
        return result, metrics
    except Exception as e:
        return f"Error in crew execution: {str(e)}", {}

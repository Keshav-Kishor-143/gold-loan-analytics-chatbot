from crewai import Agent, Task
from textwrap import dedent
import json

def create_sql_generator_agent():
    return Agent(
        role='SQL Generator',
        goal='Generate optimal SQL queries based on user intent and database schema',
        backstory=dedent("""
          You are an expert Data Scientist who able to understand user intent and precisely do the data modelling (selecting only required columns from complete dataset) from database schema.
         You only focus on the segregate the required columns only as per user intent.Main focus is to generate query to select only column.
            """),
        verbose=True,
        allow_delegation=False,
        tools=[]  # Add any specific tools needed for SQL generation
    )

# def create_sql_generation_task(user_query,schema,agent):
#     """Create the SQL generation task"""
#     return Task(
#         description=f"""
#       Based on the provided user query {user_query} and database schema {schema}:

# Understand the user's intent: Analyze the user’s query to determine the required columns needed to fulfill the request. This involves identifying the specific fields or entities that the user is asking for, without any aggregate operations or calculations.

# Extract relevant columns: From the provided {schema}, identify the exact column names related to the user’s query, ensuring that only the essential columns for the user's intent are selected (without any unnecessary columns).

# Generate a SQL SELECT statement: Construct a simple SQL query that selects only the necessary columns from the relevant tables based on the user's intent, formatted for use in SQL Server Management Studio (SSMS). The query should be focused solely on retrieving the column names that are directly related to the user's request (i.e., no aggregation, joins, or transformations).

# Output: A SQL SELECT query with the required columns, which can then be used for further analysis in Python or other tools like Pandas.""",
#         agent=agent,
#         expected_output="""A dictionary containing:
# 1. 'sql_query': The executable SQL query as a string
# 2. 'intent': A clear description of the user's intent
# 3. 'user_query': The original user query""",
#     )

def create_sql_generation_task(user_query, schema, agent):
    """Create the SQL generation task"""
    return Task(
        description=f"""
      Based on the provided user query {user_query} and database schema {schema}:
Understand the user's intent: Analyze the user’s query to determine the required columns needed to fulfill the request. This involves identifying the specific fields or entities that the user is asking for, without any aggregate operations or calculations.
Extract relevant columns: From the provided {schema}, identify the exact column names related to the user’s query, ensuring that only the essential columns for the user's intent are selected (without any unnecessary columns).
Generate a SQL SELECT statement: Construct a simple SQL query that selects only the necessary columns from the relevant tables based on the user's intent, formatted for use in SQL Server Management Studio (SSMS). The query should be focused solely on retrieving the column names that are directly related to the user's request (i.e., no aggregation, joins, or transformations).
Output: A SQL SELECT query with the required columns, which can then be used for further analysis in Python or other tools like Pandas.
**Return only a valid Python dictionary with the following keys:**
- 'sql_query': The executable SQL query as a string
- 'intent': A clear description of the user's intent
- 'user_query': The original user query
**Do not return markdown, python dict, or any explanation. Only output the JSON.**
""",
        agent=agent,
        expected_output="""A dictionary containing:
1. 'sql_query': The executable SQL query as a string
2. 'intent': A clear description of the user's intent
3. 'user_query': The original user query""",
    )
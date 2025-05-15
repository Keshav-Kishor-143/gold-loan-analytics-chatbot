import pyodbc
import pandas as pd
from crewai import Agent, Task
from textwrap import dedent

def create_database_data_retriever_agent():
    return Agent(
        role='Database Data Retriever',
        goal='Execute provided SQL queries on the database and fetch relevant data.',
        backstory=dedent("""
            You are an expert in database connectivity and data extraction. 
            You receive a SQL query and a database connection string, execute the query, 
            and return the results as a pandas DataFrame for further analysis.
        """),
        verbose=True,
        allow_delegation=False,
        tools=[]
    )

def create_database_retrieval_task(sql_query, db_connection_string, agent):
    """Create a task to execute a SQL query and fetch data from the database."""
    return Task(
        description=f"""
        Given the following SQL query:
        {sql_query}

        And the following database connection string:
        {db_connection_string}

        1. Connect to the SQL Server database using pyodbc.
        2. Execute the provided SQL query.
        3. Fetch the results and return them as a pandas DataFrame.
        4. Handle any exceptions and ensure the connection is closed after execution.
        """,
        agent=agent,
        expected_output="A pandas DataFrame containing the query results."
    )

class DatabaseDataRetriever:
    def __init__(self, db_connection_string):
        self.connection_string = db_connection_string
        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            self.connection = pyodbc.connect(self.connection_string)
            self.cursor = self.connection.cursor()
        except Exception as e:
            print(f"Failed to connect to the database: {e}")
            return None

    def fetch_data(self, sql_query):
        try:
            self.cursor.execute(sql_query)
            rows = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            data = pd.DataFrame(rows, columns=columns)
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def close_connection(self):
        if self.connection:
            self.connection.close()
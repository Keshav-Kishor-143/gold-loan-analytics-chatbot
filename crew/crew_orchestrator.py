import os
from crewai import Crew, Agent, Task
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from textwrap import dedent
from dotenv import load_dotenv
from tools.code_execution_tool import CodeExecutionTool
from tools.db_tools import DatabaseQueryTool, db_query_tool
from tools.state_manager import AnalysisStateManager

class AnalysisCrew:
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY", ''))
        self.state_manager = AnalysisStateManager()
        self.code_executor = CodeExecutionTool(state_manager=self.state_manager)
        self.db_tool = DatabaseQueryTool(state_manager=self.state_manager)
        self.message_history = ChatMessageHistory()


    def create_data_analyst(self):
        return Agent(
            role='Data Analyst',
            goal='Break down analysis tasks and guide the exploration process',
            backstory=dedent("""
                You are an expert data analyst who excels at breaking down complex analyses into
                smaller, logical steps. You do not execute code or queries yourself, but you plan
                and assign all technical tasks to the Coder. Your output should be a clear, structured list of steps for the Coder to execute (SQL, Python, visualization, etc.).
                
                Separation of concerns:
                - All data fetching must be done using SQL queries compatible with SQL Server Management Studio (SSMS). Use only queries like:
                  - SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES;
                  - SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'Loan_Customer_Summary';
                  - SELECT * FROM [table_name] WHERE ...
                - Do NOT use MySQL/Postgres-specific syntax such as LIMIT, SHOW TABLES, or non-existent schema names.
                - All data processing, analysis, and visualization must be done using Python code (pandas, matplotlib, etc.).
                - Never mix SQL and Python in the same step.
            """),
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[],  # Analyst has no tools
            memory=self.message_history
        )

    def create_coder(self):
        return Agent(
            role='Data Engineer',
            goal='Execute all assigned data queries and analysis code while maintaining state',
            backstory=dedent("""
                You are an expert Python developer specialized in data engineering and analysis.
                You execute all SQL and Python tasks assigned by the Analyst, using the available tools.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.code_executor, db_query_tool],
            memory=self.message_history
        )

    def create_tasks(self, user_query):
        analyst = self.create_data_analyst()
        coder = self.create_coder()

        planning_task = Task(
            description=dedent(f"""
                Plan the analysis for: "{user_query}"
                1. If you do not know the schema, ask the Coder to provide it first using a SQL Server compatible query (e.g., SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES;).
                2. Once you have the schema, break down the analysis into a clear, structured list of steps for the Coder to execute.
                3. For each step:
                   - If it is data fetching, provide the actual SQL query (for SQL Server, using only SSMS-compatible syntax as above).
                   - If it is data processing, analysis, or visualization, provide the actual Python code (using pandas, matplotlib, etc.).
                   - Never mix SQL and Python in the same step.
                4. Do not execute any code or queries yourself.
                5. Output only the list of steps (with code/queries) for the Coder.
            """),
            agent=analyst,
            expected_output="A structured list of analysis steps, each with either an actual SQL query (for data fetching, using only SSMS-compatible syntax) or Python code (for processing/visualization), for the Coder to execute."
        )

        execution_task = Task(
            description=dedent("""
                For each step provided by the Analyst:
                - If the step is a SQL query, use db_query_tool to fetch data (do not use code_executor for SQL).
                - If the step is Python code, use code_executor to process/analyze/visualize data (do not use db_query_tool for Python).
                - Save results and visualizations as appropriate.
                - Return outputs and explanations for each step.
                - Maintain strict separation: db_query_tool for SQL, code_executor for Python only.
            """),
            agent=coder,
            expected_output="Results and explanations for each analysis step provided by the Analyst, with strict separation of data fetching (SQL) and processing (Python)."
        )

        return [planning_task, execution_task]

    def run_analysis(self, user_query):
        tasks = self.create_tasks(user_query)
        
        crew = Crew(
            agents=[self.create_data_analyst(), self.create_coder()],
            tasks=tasks,
            verbose=True
        )

        try:
            result = crew.kickoff()
            return {
                'result': result,
                'analysis_state': self.state_manager.get_current_state()
            }
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            return None

def main():
    crew = AnalysisCrew()
    user_query = "Analyze loan distribution by customer type and branch"
    result = crew.run_analysis(user_query)
    print("\n\n======== ANALYSIS RESULTS ========")
    print(result)

if __name__ == "__main__":
    main()
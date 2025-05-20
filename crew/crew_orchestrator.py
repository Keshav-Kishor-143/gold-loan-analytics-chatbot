import os
from crewai import Crew, Agent, Task
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from textwrap import dedent
from dotenv import load_dotenv
from tools.code_execution_tool import CodeExecutionTool
from tools.db_tools import db_query_tool
from tools.state_manager import AnalysisStateManager

class AnalysisCrew:
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY", ''))
        self.code_executor = CodeExecutionTool()
        self.state_manager = AnalysisStateManager()
        self.message_history = ChatMessageHistory()

    def create_data_analyst(self):
        return Agent(
            role='Data Analyst',
            goal='Break down analysis tasks and guide the exploration process',
            backstory=dedent("""
                You are an expert data analyst who excels at breaking down complex analyses into
                smaller, logical steps. You work like a data scientist in a Jupyter notebook,
                planning each step carefully and providing clear instructions to the coder.
                
                Your approach:
                1. First understand the data schema
                2. Break analysis into smaller tasks
                3. Request exploratory queries first
                4. Build complexity gradually
                5. Validate results at each step
            """),
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[db_query_tool],
            memory=self.message_history
        )

    def create_coder(self):
        return Agent(
            role='Data Engineer',
            goal='Execute data queries and analysis code while maintaining state',
            backstory=dedent("""
                You are an expert Python developer specialized in data engineering and analysis.
                You maintain state between executions and build analysis incrementally.
                You store intermediate results and can reference previous computations.
                
                Available tools:
                1. db_query_tool: For database queries
                2. code_executor: For Python code execution
                3. state_manager: To save/load analysis state
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
                
                1. First understand available tables and their structure
                2. Break down the analysis into small, logical steps
                3. Create a sequence of tasks for the coder
                4. Start with basic exploration before complex analysis
            """),
            agent=analyst
        )

        execution_task = Task(
            description=dedent("""
                Execute the analysis steps provided by the analyst:
                1. Use db_query_tool for database queries
                2. Save results using state_manager
                3. Build on previous results
                4. Create visualizations when needed
                5. Provide clear explanations of findings
            """),
            agent=coder
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
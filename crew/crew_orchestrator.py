# Filename: enhanced_crew_orchestrator.py

import os
from crewai import Crew, Agent, Task
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from textwrap import dedent
from dotenv import load_dotenv
from crewai_tools import CodeInterpreterTool
from tools.db_tools import db_query_tool
from tools.state_manager import AnalysisStateManager
# from tools.data_bootstrap import bootstrap_database_data

# Load environment and initialize LLM
load_dotenv()
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY", ''))

# Shared conversational memory
message_history = ChatMessageHistory()
state_manager = AnalysisStateManager()
code_interpreter = CodeInterpreterTool()

# ----------------------------------------
# Agent Definitions
# ----------------------------------------

master_analyst = Agent(
    role='Master Analyst',
    goal='Coordinate EDA, hypothesis generation, coding and narrative reporting',
    backstory=dedent("""
        You are a master analyst who coordinates the entire analysis. You break down the user query into tasks for your specialized team: EDA agent, hypothesis generator, coder, and narrator.
    """),
    verbose=True,
    allow_delegation=True,
    tools=[],
    llm=llm,
    memory=message_history
)

eda_agent = Agent(
    role='EDA Agent',
    goal='Explore the database schema and data content to support analysis',
    backstory="You specialize in exploring database schemas and extracting key summaries to support further analysis.",
    verbose=True,
    allow_delegation=False,
    tools=[code_interpreter],
    llm=llm,
    memory=message_history
)

hypothesis_agent = Agent(
    role='Hypothesis Generator',
    goal='Identify patterns, trends, and possible explanations from the data',
    backstory="You specialize in exploring data for insights and forming data-driven hypotheses.",
    verbose=True,
    allow_delegation=False,
    tools=[code_interpreter],
    llm=llm,
    memory=message_history
)

coder = Agent(
    role='Coder',
    goal='Execute Python code for EDA, visualization, and statistical analysis',
    backstory="You are a skilled Python developer who runs all code needed to answer analytical questions.",
    verbose=True,
    allow_delegation=False,
    tools=[code_interpreter],
    llm=llm,
    memory=message_history
)

narrator = Agent(
    role='Narrator',
    goal='Write a coherent markdown story based on the findings and insights',
    backstory="You are an expert in crafting compelling data narratives and generating user-friendly reports.",
    verbose=True,
    allow_delegation=False,
    tools=[],
    llm=llm,
    memory=message_history
)

# ----------------------------------------
# Tasks
# ----------------------------------------

def run_crew_analysis_v2(user_query):
    # Bootstrap schema and table data into state for all agents
    # bootstrap_database_data(state_manager)

    analysis_task = Task(
        description=f"""
            Break down the user's request: "{user_query}" into smaller steps.
            Assign EDA, hypothesis, coding, and narrative writing to appropriate agents.
            Use all available insights from the schema and bootstrap to guide the flow.
        """,
        agent=master_analyst,
        expected_output="Delegated tasks assigned to relevant agents."
    )

    eda_task = Task(
        description="""
            Using the schema and preloaded data, summarize key stats, row counts, column types, and distributions.
            Suggest any anomalies or key descriptive observations.
        """,
        agent=eda_agent,
        expected_output="Key findings from EDA including tables, schema insights, and summaries."
    )

    hypothesis_task = Task(
        description="""
            Using data available from EDA, propose hypotheses or patterns relevant to the user query.
        """,
        agent=hypothesis_agent,
        expected_output="List of patterns, trends, and supported hypotheses from the data."
    )

    coding_task = Task(
        description="""
            Write and run Python code (pandas, matplotlib, seaborn) to validate hypotheses and generate visualizations.
        """,
        agent=coder,
        expected_output="Results from hypothesis testing, charts, and statistical outputs."
    )

    narration_task = Task(
        description="""
            Craft a final markdown report that presents:
            - EDA findings
            - Hypotheses
            - Validations with visuals
            - Answer to the user's query in a story-like fashion
        """,
        agent=narrator,
        expected_output="Markdown report that synthesizes the full analysis in plain English."
    )

    crew = Crew(
        agents=[master_analyst, eda_agent, hypothesis_agent, coder, narrator],
        tasks=[analysis_task, eda_task, hypothesis_task, coding_task, narration_task],
        verbose=True
    )

    try:
        result = crew.kickoff()
        return {
            'result': result,
            'analysis_state': state_manager.get_current_state()
        }
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None

if __name__ == "__main__":
    query = "Analyze customer behavior by region and suggest segmentation strategies."
    result = run_crew_analysis_v2(query)
    print("\n\n======== FINAL REPORT ========")
    print(result['result'])

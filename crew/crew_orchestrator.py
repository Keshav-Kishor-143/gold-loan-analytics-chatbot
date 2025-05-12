from crewai import Crew
from agents.data_analyst_agent import create_data_analyst_agent, create_analysis_task
from agents.code_executor_agent import create_code_executor_agent, create_execution_task
from agents.response_formatter_agent import create_response_formatter_agent, create_formatting_task

def create_analysis_crew(user_query, loading_instructions):
    """Create and configure the analysis crew with all necessary agents and tasks"""
    # Create agents
    data_analyst = create_data_analyst_agent()
    code_executor = create_code_executor_agent()
    response_formatter = create_response_formatter_agent()
    
    # Create tasks using the agent-specific task creation functions
    analysis_task = create_analysis_task(user_query, loading_instructions, data_analyst)
    execution_task = create_execution_task(code_executor)
    formatting_task = create_formatting_task(response_formatter)
    
    # Create and return the crew
    return Crew(
        agents=[data_analyst, code_executor, response_formatter],
        tasks=[analysis_task, execution_task, formatting_task],
        verbose=True
    )

def run_analysis(user_query, loading_instructions):
    """Run the analysis using the crew"""
    crew = create_analysis_crew(user_query, loading_instructions)
    return crew.kickoff()

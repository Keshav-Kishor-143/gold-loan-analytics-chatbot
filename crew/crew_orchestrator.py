from crewai import Crew
from agents.data_retriever_agent import create_data_retriever_agent, create_retrieval_task
from agents.code_generator_agent import create_code_generator_agent, create_code_generation_task
from agents.code_executor_agent import create_code_executor_agent, create_execution_task
from agents.response_formatter_agent import create_response_formatter_agent, create_formatting_task

def create_analysis_crew(user_query, loading_instructions):
    """Create and configure the analysis crew with all necessary agents and tasks"""
    # Create agents
    data_retriever = create_data_retriever_agent()
    code_generator = create_code_generator_agent()
    code_executor = create_code_executor_agent()
    response_formatter = create_response_formatter_agent()
    
    # Create tasks using the agent-specific task creation functions
    retrieval_task = create_retrieval_task(user_query, loading_instructions, data_retriever)
    code_generation_task = create_code_generation_task(code_generator)
    execution_task = create_execution_task(code_executor)
    formatting_task = create_formatting_task(response_formatter)
    
    # Set up task dependencies
    code_generation_task.context = """
    Use the filtered dataframe and query intent provided by the Data Retriever agent.
    """
    execution_task.context = """
    Execute the Python code provided by the Code Generator agent.
    """
    formatting_task.context = """
    Format the results from the Code Executor agent into a meaningful HTML table with business insights.
    """
    
    # Create and return the crew with the correct task order
    return Crew(
        agents=[data_retriever, code_generator, code_executor, response_formatter],
        tasks=[retrieval_task, code_generation_task, execution_task, formatting_task],
        verbose=True
    )

def run_analysis(user_query, loading_instructions):
    """Run the analysis using the crew"""
    crew = create_analysis_crew(user_query, loading_instructions)
    return crew.kickoff()

from crewai import Crew
from agents.data_retriever_agent import create_data_retriever_agent, create_retrieval_task
from agents.code_generator_agent import create_code_generator_agent, create_code_generation_task
from agents.code_executor_agent import create_code_executor_agent, create_execution_task
from agents.response_formatter_agent import create_response_formatter_agent, create_formatting_task
from crewai_tools import CodeInterpreterTool
import os
import traceback
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def create_analysis_crew(user_query, csv_files=None, dataframes=None, loading_instructions=None):
    """Create and configure the multi‐agent analysis crew."""
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini",
                         openai_api_key=os.getenv("OPENAI_API_KEY", ""))

        if not any([csv_files, dataframes, loading_instructions]):
            raise ValueError("Provide csv_files, dataframes, or loading_instructions.")

        # Instantiate agents
        data_retriever   = create_data_retriever_agent(llm=llm)
        code_generator   = create_code_generator_agent(llm=llm)
        code_executor    = create_code_executor_agent(llm=llm)
        response_formatter = create_response_formatter_agent(llm=llm)

        # Create their tasks
        retrieval_task       = create_retrieval_task(user_query, csv_files, data_retriever)
        code_generation_task = create_code_generation_task(code_generator)
        execution_task       = create_execution_task(code_executor)
        formatting_task      = create_formatting_task(response_formatter)

        # Wire contexts: each task only sees the preceding task's output
        code_generation_task.context = [retrieval_task]
        execution_task.context      = [code_generation_task]
        formatting_task.context     = [execution_task]

        return Crew(
            agents=[data_retriever, code_generator, code_executor, response_formatter],
            tasks=[retrieval_task, code_generation_task, execution_task, formatting_task],
            llm=llm,
            verbose=True,
            process="sequential"
        )

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Error creating crew] {e}\n{tb}")
        raise

def run_analysis(user_query, csv_files=None, dataframes=None, loading_instructions=None):
    """Kick off the crew and return the final HTML document as a string."""
    try:
        if not user_query or len(user_query.strip()) < 5:
            return "Error: Query too short or empty."

        # Verify CSV paths upfront
        if csv_files:
            missing = [p for p in csv_files.values() if not os.path.exists(p)]
            if missing:
                return f"Error: Missing CSV files: {missing}"

        crew = create_analysis_crew(user_query, csv_files, dataframes, loading_instructions)
        result = crew.kickoff(inputs={"user_query": user_query})

        # Handle the response based on its type
        if isinstance(result, dict):
            return result.get("html_document", str(result))
        elif hasattr(result, "raw"):
            if isinstance(result.raw, dict):
                return result.raw.get("html_document", str(result.raw))
            return str(result.raw)
        else:
            return str(result)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Error during analysis] {e}\n{tb}")
        return f"Error during analysis: {e}"

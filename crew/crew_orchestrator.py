import json
from crewai import Crew
from agents.sql_generator_agent import create_sql_generator_agent, create_sql_generation_task
from agents.database_data_retriever_agent import create_database_data_retriever_agent,create_database_retrieval_task
from agents.data_retriever_agent import create_data_retriever_agent, create_retrieval_task
from agents.code_generator_agent import create_code_generator_agent, create_code_generation_task
from agents.code_executor_agent import create_code_executor_agent, create_execution_task
from agents.response_formatter_agent import create_response_formatter_agent, create_formatting_task

def create_analysis_crew(user_query, schema):
    """Create and configure the analysis crew with all necessary agents and tasks"""
    # Create agents
    sql_generator = create_sql_generator_agent()
    # database_retriever = create_database_data_retriever_agent()
    data_retriever = create_data_retriever_agent()
    # code_generator = create_code_generator_agent()
    # code_executor = create_code_executor_agent()
    # response_formatter = create_response_formatter_agent()
    
    # Create tasks using the agent-specific task creation functions
    sql_generation_task = create_sql_generation_task(user_query, schema, sql_generator)
    # Pass the generated SQL query to the data retriever agent using CrewAI's variable interpolation
    retrieval_task = create_retrieval_task(sql_query="{{sql_query}}", agent=data_retriever)
    retrieval_task.context = [sql_generation_task]
    
    # # Create and return the crew with the correct task order
    # return Crew(
    #     agents=[data_retriever, code_generator, code_executor, response_formatter],
    #     tasks=[retrieval_task, code_generation_task, execution_task, formatting_task],
    #     verbose=True
    # )

    return Crew(
        agents=[sql_generator, data_retriever],
        tasks=[sql_generation_task, retrieval_task],
        verbose=True
    )


# def run_analysis(user_query, schema):
#     """Run the analysis using the crew"""
#     crew = create_analysis_crew(user_query, schema)
#     result= crew.kickoff()
#      # If result is a dict with 'output' or similar, return just that
#     if isinstance(result, dict) and "output" in result:
#         return result["raw"]
#     # Otherwise, return the result as is
#     return result


import ast
import re

def run_analysis(user_query, schema):
    crew = create_analysis_crew(user_query, schema)
    result = crew.kickoff()
    print("DEBUG: Raw crew result:", result)
    return json.loads(result.raw)
    # # If result is already a dict with the expected keys, return it directly
    # if isinstance(result, dict) and all(k in result for k in ("sql_query", "user_query", "intent")):
    #     return result

    # # If result is nested, try to extract the dict
    # if isinstance(result, dict):
    #     for v in result.values():
    #         if isinstance(v, dict) and all(k in v for k in ("sql_query", "user_query", "intent")):
    #             return v

    # # Try to extract from 'raw' or 'output'
    # raw_output = None
    # if isinstance(result, dict):
    #     raw_output = result.get("raw") or result.get("output")
    # elif isinstance(result, str):
    #     raw_output = result

    # if not raw_output:
    #     return {
    #         "error": "No output from agent",
    #         "raw": raw_output
    #     }

    # # Extract the first Python dict from the string using regex
    # dict_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    # if dict_match:
    #     dict_str = dict_match.group(0)
    #     try:
    #         parsed = ast.literal_eval(dict_str)
    #         if all(k in parsed for k in ("sql_query", "user_query", "intent")):
    #             return parsed
    #         else:
    #             return {"error": "Output missing expected keys", "raw": raw_output}
    #     except Exception as e:
    #         return {"error": str(e), "raw": dict_str}
    # else:
    #     return {"error": "No dictionary found in output", "raw": raw_output}
import json
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import os
from dotenv import load_dotenv
from tools.idrp_tools import IDRPTool
 
def execute_crew_query(query: str, llm_instance: ChatOpenAI, inventory_tools: list, sales_tools: list) -> str:
    """
    Execute crew workflow with proper validation and separation of concerns:
    1. Analyzer+Planner validates query and creates plan if valid
    2. Executor uses tools to get raw data
    3. Formatter presents the data in a user-friendly way
    4. Master manages the flow
    """
    # Generate tool schema for analyzer and planner
    tool_schema = {
        "tools": [
            {"name": tool.name, "description": tool.description}
            for tool in (inventory_tools + sales_tools)
        ]
    }
 
 
    # Query Analyzer and Planner Agent
    analyzer_planner = Agent(
        role="Query Analyzer & Planner",
        goal="Validate queries and create structured execution plans for valid ones",
        backstory=f"""You have two responsibilities:
        1. Analyze if queries are valid and within system scope
        2. If valid, create a plan for execution
       
        Available Tools Schema:
        {json.dumps(tool_schema, indent=2)}
       
        Validation Rules:
        1. Query must be about inventory or sales data
        2. Product codes must be in SPGR format
        3. Location names must be valid
        4. Query must be clear and actionable
       
        If query is invalid, return a clear error message.
        If valid, create and return a detailed plan(sequential tool execution plan and data processing plan) for the executor.""",
        allow_delegation=False,
        verbose=True,
        llm=llm_instance
    )
 
    # Executor Agent
    executor_agent = Agent(
        role="DRP Executor",
        goal="Execute plans using tools and return raw data",
        backstory="""You execute plans by using the appropriate tools.
        Your job is to get the data and return it as it.""",
        tools=inventory_tools + sales_tools,
        allow_delegation=False,
        verbose=True,
        llm=llm_instance
    )
 
    # Formatter Agent
    formatter_agent = Agent(
        role="Response Formatter",
        goal="Format responses in a clear and consistent manner",
        backstory="""You take raw data or error messages and format them for end users.
       
        Formatting Rules:
        1. Use markdown for structure
        2. Use bullet points for lists
        3. Format product codes in single quotes
        4. Keep metrics in original numeric form
        5. Create clear error messages for invalid queries""",
        allow_delegation=False,
        verbose=True,
        llm=llm_instance
    )
 
    # Analysis and Planning Task
    analysis_planning_task = Task(
        description=f"""
        For this query: {query}
       
        1. First, analyze if the query is valid:
           - Must be about inventory or sales or both
           - Must have valid input parameters for tools (if any)
           - Must be clear and actionable
       
        2. If valid, create an tool execution plan first:
           - What tools to use and in what order
           - Which tools to use
           - What parameters to pass
           - What is expected output
        3. If query requires any processing create a separate plan for processing(if any filtration, sorting, aggregation, etc required)
        4. If invalid, return an error message explaining why
        """,
        expected_output="if valid, return a detailed plan for the executor, if invalid, return an error message for formatter",
        agent=analyzer_planner
    )
 
    # Execution Task (only created if query is valid)
    execution_task = Task(
        description=f"""
        Execute the plan for query: {query}
        IMPORTANT:
        1. Call tools in the EXACT order specified in the plan
        2. Use ONLY the parameters mentioned in the plan
        3. Apply any filtering/sorting EXACTLY as specified
        4. DO NOT skip any processing steps
        5. Return complete and processed data
        """,
        expected_output="Complete processed data after all tool calls and processing steps",
        agent=executor_agent,
        context=[analysis_planning_task],
       
    )
 
    # Formatting Task
    formatting_task = Task(
        description=f"""
        Format the response for: {query}
       
        If you receive an error message, format it clearly.
        If you receive raw data, format it according to the rules:
        1. Use markdown
        2. Use bullet points for lists
        3. Format product codes in single quotes
        4. Keep numeric values as is
       
        Return the formatted response to the user.
        """,
        expected_output="return formatted response to user",
        context=[analysis_planning_task, execution_task],
        agent=formatter_agent
    )
 
    # Create crew with sequential process
    crew = Crew(
        agents=[analyzer_planner, executor_agent, formatter_agent],
        tasks=[analysis_planning_task, execution_task, formatting_task],
        verbose=True
    )
 
    try:
        result = crew.kickoff()
        usage_metrics = crew.usage_metrics
       
        # Ensure we have numeric values for metrics
        metrics = {
            "total_tokens": int(getattr(usage_metrics, 'total_tokens', 0)),
            "prompt_tokens": int(getattr(usage_metrics, 'prompt_tokens', 0)),
            "completion_tokens": int(getattr(usage_metrics, 'completion_tokens', 0)),
            "successful_requests": int(getattr(usage_metrics, 'successful_requests', 0))
        }
        return result,metrics
    except Exception as e:
        return f"Error in crew execution: {str(e)}",{}
 
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", '')
 
    # Load API specifications
    with open("openapi.json", "r") as f:
        openapi_spec = json.load(f)
 
    # Initialize tools
    idrp_tool = IDRPTool(openapi_spec)
   
    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)
   
    # Test query
    query = "What is the current stock at hand for top 3 overselling products in Delhi?"
    result,metrics = execute_crew_query(
        query=query,
        llm_instance=llm,
        inventory_tools=idrp_tool.inventory_tools,
        sales_tools=idrp_tool.sales_tools
    )
    print("new crew",metrics)
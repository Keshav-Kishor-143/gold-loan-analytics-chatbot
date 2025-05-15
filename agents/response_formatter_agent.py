from crewai import Agent, Task
from textwrap import dedent

def create_response_formatter_agent(llm=None):
    return Agent(
        role='Response Formatter',
        goal='Format tabular analysis results into clear, professional HTML documents with business insights',
        backstory=dedent("""
            You are an expert in data presentation and business communication. You excel at
            transforming complex analytical results into clear, visually appealing HTML tables
            that highlight key insights. You have a strong background in financial reporting
            and know how to present tabular data insights that drive business decision-making.
        """),
        verbose=False,
        allow_delegation=False,
        tools=[],  # formatting with LLM only
        llm=llm,
        input_schema={
            "html_table": str,
            "raw_output": str,
            "query_intent": str
        }
    )

def create_formatting_task(agent: Agent) -> Task:
    """
    Task to wrap analysis table into a full HTML document with business insights.

    INPUT JSON:
      {
        "html_table": "<table>...</table>",  # may be empty
        "raw_output": "<full raw stdout/stderr>",
        "query_intent": "<original user query>"
      }

    OUTPUT JSON:
      {
        "html_document": "<!DOCTYPE html>..."  # full HTML5 document
      }
    """
    description = dedent("""
    You receive one JSON input with keys:
      • html_table (string containing the table markup or empty string)
      • raw_output (string with execution logs or errors)
      • query_intent (the original user query)

    Your task is to create a professional HTML document that:
    1. Has a clear, business-focused structure
    2. Highlights key insights from the data
    3. Presents information in a visually appealing way
    4. Is responsive and accessible

    Steps:
    1. Analyze the input:
       - Review the html_table for data patterns
       - Check raw_output for any errors or warnings
       - Understand the query_intent for context

    2. Generate a complete HTML5 document with:
       - Proper DOCTYPE and meta tags
       - Responsive CSS styling
       - Clear headings and structure
       - The provided table (or fallback if empty)
       - Key insights section
       - Error handling if needed

    3. Return exactly this JSON structure:
       {
         "html_document": "<!DOCTYPE html>..."
       }

    Important:
    - Ensure valid HTML5 structure
    - Include responsive design
    - Handle empty or invalid tables gracefully
    - Provide clear error messages if needed
    - Focus on business insights
    - Use professional styling
    """)

    return Task(
        description=description,
        agent=agent,
        expected_output = """
            {
            "type": "object",
            "required": ["html_document"],
            "properties": {
               "html_document": { "type": "string" }
            },
            "additionalProperties": false
            }
            """
    )

import os
import sys
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import CodeInterpreterTool, SerperDevTool
from langchain_openai import ChatOpenAI
from io import StringIO
import contextlib
import locale

# Load environment variables from .env file
load_dotenv()

# Set default CSV file path
DEFAULT_CSV_PATH = "goldloan.csv"

# Set Indian locale for number formatting (if available)
try:
    locale.setlocale(locale.LC_MONETARY, 'en_IN')
except:
    pass  # Fall back to default if Indian locale not available

# Create a capture context for capturing print output
@contextlib.contextmanager
def capture_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def format_indian_currency(amount):
    """Format amount in Indian currency format (₹xx,xx,xxx.xx)"""
    try:
        # Convert to float first
        amount = float(amount)
        
        # Get integer and decimal parts
        integer_part = int(amount)
        decimal_part = int(round((amount - integer_part) * 100))
        
        # Format the integer part with commas for Indian format
        s = str(integer_part)
        result = s[-3:]
        s = s[:-3]
        
        # Add commas for thousands and beyond
        i = 0
        while s:
            if i % 2 == 0:
                result = s[-2:] + "," + result if len(s[-2:]) == 2 else s[-1:] + "," + result
            else:
                result = s[-2:] + "," + result if len(s[-2:]) == 2 else s[-1:] + "," + result
            s = s[:-2]
            i += 1
        
        # Remove leading comma if present
        if result.startswith(","):
            result = result[1:]
            
        # Add decimal part and ₹ symbol
        if decimal_part:
            return f"₹{result}.{decimal_part:02d}"
        else:
            return f"₹{result}"
    except:
        return f"₹{amount}"  # Fallback for any errors

def execute_analysis_code(code_str):
    """Execute the analysis code and capture both results and printed output"""
    try:
        # Create a local namespace for execution
        local_vars = {}
        
        # Capture printed output during execution
        with capture_output() as (out, err):
            # Execute the code
            exec(code_str, globals(), local_vars)
        
        # Get the printed output
        output = out.getvalue()
        
        # Get the dataframe or series results
        results = {}
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, (pd.DataFrame, pd.Series)):
                results[var_name] = var_value
        
        # If there are no dataframe/series results but there is printed output,
        # return the printed output as the main result
        if not results and output:
            return {"output": output}
        
        # Otherwise return both results and output
        results["output"] = output
        return results
    except Exception as e:
        return {"error": str(e)}

def get_dataset_info(df):
    """Get dynamic information about the dataset"""
    info = {}
    try:
        # Basic info
        info['rows'] = len(df)
        info['columns'] = list(df.columns)
        
        # Numeric columns
        info['numeric_columns'] = df.select_dtypes(include=['number']).columns.tolist()
        
        # Categorical columns
        info['categorical_columns'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Date columns (heuristic)
        date_columns = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_columns.append(col)
        info['date_columns'] = date_columns
        
        # Missing values
        info['missing_values'] = df.isnull().sum().to_dict()
        
        # Column types
        info['column_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Sample statistics for numeric columns
        info['statistics'] = {}
        for col in info['numeric_columns']:
            try:
                info['statistics'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std())
                }
            except:
                pass  # Skip columns that can't be summarized
                
        # Unique values for categorical columns (if not too many)
        info['unique_values'] = {}
        for col in info['categorical_columns']:
            try:
                unique_vals = df[col].unique().tolist()
                if len(unique_vals) <= 10:  # Only if there aren't too many unique values
                    info['unique_values'][col] = unique_vals
            except:
                pass
    
    except Exception as e:
        info['error'] = str(e)
    
    return info

def format_output(result):
    """Format the analysis results in a concise, report-style format"""
    formatted_result = str(result)
    
    # Replace verbose headers with concise ones
    formatted_result = formatted_result.replace("# Agent: Data Analyst", "")
    formatted_result = formatted_result.replace("## Final Answer:", "")
    formatted_result = formatted_result.replace("# Agent:", "")
    formatted_result = formatted_result.replace("## Task:", "")
    formatted_result = formatted_result.replace("## Thought:", "")
    
    # Fix any $ symbols to ₹
    formatted_result = formatted_result.replace("$", "₹")
    
    return formatted_result

def run_loan_analysis():
    try:
        # Use default CSV file path
        data_file = DEFAULT_CSV_PATH
        
        # Validate file exists
        if not os.path.exists(data_file):
            print(f"Error: File '{data_file}' not found.")
            return
            
        # Load data
        try:
            df = pd.read_csv(data_file)
            print("\nPreview of your data:")
            print(df.head(3))
            print(f"\nTotal rows: {len(df)}")
            print(f"Columns: {', '.join(df.columns)}")
            
            # Get dataset info for dynamic prompting
            dataset_info = get_dataset_info(df)
            
        except Exception as e:
            print(f"Error analyzing data: {e}")
            return
        
        # Configure OpenAI as the LLM
        openai_llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
        )
        
        # Initialize tools
        code_interpreter = CodeInterpreterTool(
            execute_code=execute_analysis_code
        )
        search_tool = SerperDevTool()
        
        # Define the Data Analyst Agent
        data_analyst = Agent(
            role="Data Analyst",
            goal="Analyze data dynamically to extract meaningful insights and patterns.",
            backstory=f"""You are an expert financial data analyst with proficiency in Python and pandas.
                      Your role is to dynamically analyze loan data based on user queries.
                      
                      The dataset has the following characteristics:
                      - Rows: {dataset_info['rows']}
                      - Columns: {', '.join(dataset_info['columns'])}
                      - Numeric columns: {', '.join(dataset_info['numeric_columns'])}
                      - Categorical columns: {', '.join(dataset_info['categorical_columns'])}
                      """,
            tools=[code_interpreter],
            verbose=True,
            allow_delegation=False
        )
        
        # Show example queries
        print("\nExample queries you can try:")
        print("1. Analyze loan distribution by customer type and branch")
        print("2. Find the average loan amount by scheme")
        print("3. Identify NPA patterns across different branches")
        print("4. Compare loan amounts between customer types")
        
        # Get user query
        user_query = input("\nEnter your loan data analysis query: ")
        
        # Create analysis task
        analysis_task = Task(
            description=f"""
            Analyze the loan data in the file '{data_file}' to answer:
        
        {user_query}
        
            Start by loading the data with pandas:
            ```python
            import pandas as pd
            df = pd.read_csv('{data_file}')
            ```
            
            Then write analytical code to directly answer the query. Format all monetary values with the Indian Rupee symbol (₹).
            
            Provide a complete analysis answering the query with:
            1. Quantitative insights based on actual data
            2. Visualizations if helpful
            3. Business implications
            4. Recommendations
            """,
            agent=data_analyst,
            expected_output="Data analysis with insights and recommendations."
        )
        
        # Create and run the crew
        analytics_crew = Crew(
            agents=[data_analyst],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=True
        )
        
        print("\nAnalyzing your query. This may take a few moments...")
        result = analytics_crew.kickoff()
        
        # Format the output
        formatted_result = format_output(result)
        
        print("\n" + "="*50)
        print("LOAN ANALYSIS REPORT")
        print("="*50)
        print(formatted_result)
        
        return result
    
    except Exception as e:
        print(f"\nAn error occurred during analysis: {str(e)}")
        print("Please check your API keys and ensure all dependencies are properly installed.")
        return None

def main():
    print("="*50)
    print("LOAN DATA ANALYSIS TOOL WITH OPENAI")
    print("="*50)
    
    # Check if OpenAI API key is set correctly
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Run the analysis
    run_loan_analysis()

if __name__ == "__main__":
    main()
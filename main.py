import os
import sys
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from io import StringIO
import contextlib
import locale
from crew.crew_orchestrator import run_analysis

# Load environment variables from .env file
load_dotenv()

# Set default CSV file paths
CUSTOMER_SUMMARY_PATH = "customer_summary.csv"
PAYMENT_SUMMARY_PATH = "payment_summary.csv"

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

# Note: This function is currently unused but kept for potential future use
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

def load_datasets():
    """Load all required datasets"""
    try:
        # Load main loan data
        loan_df = pd.read_csv(CUSTOMER_SUMMARY_PATH)
        
        # Load payment data
        payment_df = pd.read_csv(PAYMENT_SUMMARY_PATH)
        
        return loan_df, payment_df
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        return None, None

def main():
    # Load datasets
    loan_df, payment_df = load_datasets()
    if loan_df is None or payment_df is None:
        print("Failed to load datasets. Exiting...")
        return
    
    # Show example queries
    print("\nExample queries you can try:")
    print("1. Analyze loan distribution by customer type and branch")
    print("2. Find the average loan amount by scheme")
    print("3. Compare loan status patterns by branch")
    print("4. Identify NPA patterns across different branches") 
    print("5. Analyze relationships between loan details and payment behavior")
    
    # Get user query
    user_query = input("\nEnter your analysis query: ")
    
    # Create loading instructions
    loading_instructions = f"""
    loan_df = pd.read_csv('{CUSTOMER_SUMMARY_PATH}')
    payment_df = pd.read_csv('{PAYMENT_SUMMARY_PATH}')
    """
    
    # Run the analysis using the crew orchestrator
    result = run_analysis(user_query, loading_instructions)
    print("\nAnalysis Results:")
    print(result)

if __name__ == "__main__":
    main()
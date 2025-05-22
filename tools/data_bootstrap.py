# Filename: tools/data_bootstrap.py

import os
import requests
import json
import pandas as pd
from typing import Optional
from tools.state_manager import AnalysisStateManager

API_BASE_URL = "http://localhost:8000"

def fetch_schema() -> dict:
    """Fetch database schema: tables, columns, and types"""
    query = """
    SELECT 
        TABLE_NAME, COLUMN_NAME, DATA_TYPE 
    FROM 
        INFORMATION_SCHEMA.COLUMNS
    ORDER BY 
        TABLE_NAME, ORDINAL_POSITION;
    """
    response = requests.post(f"{API_BASE_URL}/execute-query", json={"query": query})
    data = response.json()
    if data["status"] != "success":
        raise Exception(f"Schema fetch failed: {data}")
    
    schema = {}
    for row in data["results"]:
        table = row["TABLE_NAME"]
        if table not in schema:
            schema[table] = []
        schema[table].append({
            "column": row["COLUMN_NAME"],
            "type": row["DATA_TYPE"]
        })
    return schema

def fetch_table_data(table_name: str) -> pd.DataFrame:
    """Fetch all data from a single table"""
    query = f"SELECT * FROM [{table_name}];"
    response = requests.post(f"{API_BASE_URL}/execute-query", json={"query": query})
    data = response.json()
    if data["status"] != "success":
        raise Exception(f"Failed to fetch data from {table_name}: {data}")
    return pd.DataFrame(data["results"])

def bootstrap_database_data(state_manager: Optional[AnalysisStateManager] = None, output_dir: str = "./bootstrap_output"):
    """Fetch schema + data and save as schema.json and .pkl files"""
    os.makedirs(output_dir, exist_ok=True)
    state_manager = state_manager or AnalysisStateManager()

    print("[INFO] Fetching schema...")
    schema = fetch_schema()

    # Save schema.json
    schema_path = os.path.join(output_dir, "schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"[INFO] Schema saved to {schema_path}")
    state_manager.add_analysis_step("Schema bootstrap", {"saved_to": schema_path})

    # Fetch each table's data and save as .pkl
    for table_name in schema.keys():
        print(f"[INFO] Fetching data from table: {table_name}")
        df = fetch_table_data(table_name)
        pkl_path = os.path.join(output_dir, f"{table_name}.pkl")
        df.to_pickle(pkl_path)
        print(f"[INFO] Saved data for '{table_name}' to {pkl_path}")

        # Save to state manager
        state_manager.save_dataframe(table_name, df)
        state_manager.add_analysis_step(
            f"Data bootstrap: {table_name}",
            {
                "table": table_name,
                "rows": len(df),
                "columns": list(df.columns),
                "file": pkl_path
            }
        )

    print("[INFO] Bootstrap process complete.")

# Standalone run
if __name__ == "__main__":
    bootstrap_database_data()

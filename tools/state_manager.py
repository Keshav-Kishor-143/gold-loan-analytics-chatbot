import pandas as pd
from pathlib import Path
import json
from datetime import datetime

class AnalysisStateManager:
    def __init__(self):
        self.state = {
            'dataframes': {},
            'variables': {},
            'code_history': [],
            'analysis_steps': [],
            'current_step': 0
        }
        self.workspace_dir = Path('./analysis_workspace')
        self.workspace_dir.mkdir(exist_ok=True)
    def get_variable_dict(self):
        """Return a dictionary of all variables in state"""
        return self.state['variables']
    
    def get_dataframe_dict(self):
        """Return a dictionary of all dataframes in state"""
        return self.state['dataframes']
    
    def save_dataframe(self, name: str, df: pd.DataFrame):
        """Save a DataFrame to the state"""
        self.state['dataframes'][name] = df
        # Save to disk as backup
        df.to_pickle(self.workspace_dir / f"{name}.pkl")

    def get_dataframe(self, name: str) -> pd.DataFrame:
        """Retrieve a DataFrame from state"""
        return self.state['dataframes'].get(name)

    def add_variable(self, name: str, value):
        """Store a variable in state"""
        self.state['variables'][name] = value

    def get_variable(self, name: str):
        """Retrieve a variable from state"""
        return self.state['variables'].get(name)

    def add_code(self, code: str, description: str):
        """Add executed code to history"""
        self.state['code_history'].append({
            'code': code,
            'description': description,
            'timestamp': datetime.now().isoformat()
        })

    def add_analysis_step(self, step_description: str, results: dict):
        """Record an analysis step"""
        self.state['analysis_steps'].append({
            'step': self.state['current_step'],
            'description': step_description,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        self.state['current_step'] += 1

    def save_state(self):
        """Save entire state to disk"""
        state_to_save = self.state.copy()
        # Convert DataFrames to file references
        for df_name in state_to_save['dataframes']:
            state_to_save['dataframes'][df_name] = f"{df_name}.pkl"
        
        with open(self.workspace_dir / 'analysis_state.json', 'w') as f:
            json.dump(state_to_save, f, indent=2)

    def load_state(self):
        """Load state from disk"""
        try:
            with open(self.workspace_dir / 'analysis_state.json', 'r') as f:
                loaded_state = json.load(f)
            
            # Restore DataFrames from files
            for df_name, file_ref in loaded_state['dataframes'].items():
                df_path = self.workspace_dir / file_ref
                if df_path.exists():
                    self.state['dataframes'][df_name] = pd(df_path)
            
            # Restore other state components
            self.state['variables'] = loaded_state['variables']
            self.state['code_history'] = loaded_state['code_history']
            self.state['analysis_steps'] = loaded_state['analysis_steps']
            self.state['current_step'] = loaded_state['current_step']
        except FileNotFoundError:
            print("No previous state found")

    def get_current_state(self):
        """Get a summary of current state"""
        return {
            'available_dataframes': list(self.state['dataframes'].keys()),
            'variable_count': len(self.state['variables']),
            'code_history_length': len(self.state['code_history']),
            'analysis_steps_completed': self.state['current_step']
        }
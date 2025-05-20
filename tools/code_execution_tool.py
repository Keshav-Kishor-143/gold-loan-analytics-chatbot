from typing import Type, Optional, Dict, Any
import docker
import pandas as pd
import json
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from state_manager import AnalysisStateManager

class DockerSandbox:
    def __init__(self, container_name: str = "my-python-container", state_manager=None):
        self.client = docker.from_env()
        self.state_manager = state_manager
        try:
            self.container = self.client.containers.get(container_name)
            print(f"Attached to existing container: {container_name}")
        except docker.errors.NotFound:
            print(f"Container '{container_name}' not found.")
            self.container = None

    def prepare_code_with_state(self, code: str) -> str:
        """Wraps code with state management"""
        if not self.state_manager:
            return code

        # Get current state variables
        state_vars = self.state_manager.get_variable_dict()
        state_dfs = self.state_manager.get_dataframe_dict()

        # Create state initialization code
        init_code = []
        
        # Add DataFrame initializations
        for df_name, df in state_dfs.items():
            df_json = df.to_json()
            init_code.append(f"{df_name} = pd.read_json('{df_json}')")

        # Add variable initializations
        for var_name, value in state_vars.items():
            if isinstance(value, (int, float, bool, str)):
                init_code.append(f"{var_name} = {repr(value)}")
            else:
                init_code.append(f"{var_name} = {json.dumps(value)}")

        # Combine initialization and user code
        full_code = "\n".join([
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            *init_code,
            code
        ])
        return full_code

    def run_code(self, code: str) -> Dict[str, Any]:
        if not self.container:
            return {"error": "No container available to run code."}

        wrapped_code = self.prepare_code_with_state(code)
        
        # Execute code in container
        exec_result = self.container.exec_run(
            cmd=["python", "-c", wrapped_code],
            user="nobody",
            stdout=True,
            stderr=True
        )

        result = {
            "stdout": exec_result.output.decode() if exec_result.output else "",
            "success": exec_result.exit_code == 0,
            "variables": {},
            "dataframes": {}
        }

        # Extract variables and DataFrames if execution was successful
        if result["success"]:
            # Code to extract variables and DataFrames would go here
            pass

        return result

    def cleanup(self):
        pass

class CodeExecutionInput(BaseModel):
    """Input schema for CodeExecutionTool."""
    code: str = Field(..., description="Python code to execute in sandbox environment")
    save_state: bool = Field(True, description="Whether to save results to state manager")
    description: str = Field("", description="Description of the code execution step")

class CodeExecutionTool(BaseTool):
    name: str = "Code Execution Tool"
    description: str = """
    Executes Python code in a secure sandbox environment with state management.
    The tool maintains state between executions and can work with DataFrames.
    Results can be saved to the state manager for future reference.
    """
    args_schema: Type[BaseModel] = CodeExecutionInput

    def __init__(self, state_manager: Optional[AnalysisStateManager] = None):
        super().__init__()
        self.state_manager = state_manager or AnalysisStateManager()

    def _run(self, code: str, save_state: bool = True, description: str = "") -> Dict[str, Any]:
        sandbox = DockerSandbox(state_manager=self.state_manager)
        
        try:
            result = sandbox.run_code(code)
            
            if save_state and result["success"]:
                # Save code to history
                self.state_manager.add_code(code, description)
                
                # Save any new variables or DataFrames
                if "variables" in result:
                    for name, value in result["variables"].items():
                        self.state_manager.add_variable(name, value)
                
                if "dataframes" in result:
                    for name, df in result["dataframes"].items():
                        self.state_manager.save_dataframe(name, df)
                
                # Save execution step
                self.state_manager.add_analysis_step(
                    description or "Code execution",
                    {"output": result["stdout"]}
                )
            
            return result
        except Exception as e:
            return {"error": str(e), "success": False}
        finally:
            sandbox.cleanup()
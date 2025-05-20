import docker
import os
from typing import Optional

class DockerSandbox:
    def __init__(self, container_name: str = "my-python-container"):
        self.client = docker.from_env()
        try:
            self.container = self.client.containers.get(container_name)
            print(f"Attached to existing container: {container_name}")
        except docker.errors.NotFound:
            print(f"Container '{container_name}' not found.")
            self.container = None

    def run_code(self, code: str) -> Optional[str]:
        if not self.container:
            print("No container available to run code.")
            return None

        # Execute code in the container
        exec_result = self.container.exec_run(
            cmd=["python", "-c", code],
            user="nobody",
            stdout=True,
            stderr=True
        )

        # Collect all output
        output = exec_result.output.decode() if exec_result.output else None
        return output

    def cleanup(self):
        # No cleanup necessary for existing container
        pass


# Example usage:
sandbox = DockerSandbox()

try:
    # Define your agent code
    agent_code = """print(10+10+20*100)"""

    # Run the code in the sandbox
    output = sandbox.run_code(agent_code)
    print(output)

finally:
    sandbox.cleanup()

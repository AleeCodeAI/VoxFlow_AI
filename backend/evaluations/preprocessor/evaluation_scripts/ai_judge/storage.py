from pydantic_schemas import Result, EvaluationError
from datetime import datetime


class Storage:
    """Handles saving evaluation results to the execution log file"""

    def __init__(self, execution_path: str):
        self.execution_path = execution_path

    def save_execution(self, result: Result, mode: str = "a"):
        try:
            with open(self.execution_path, mode, encoding="utf-8") as f:
                f.write(result.model_dump_json() + "\n")
        except Exception as e:
            error = EvaluationError(
                error_type="ExecutionSaveError",
                message=str(e),
                timestamp=datetime.now().isoformat(),
                context={"result_id": result.id, "file_path": self.execution_path},
            )
            raise Exception(error.model_dump_json())

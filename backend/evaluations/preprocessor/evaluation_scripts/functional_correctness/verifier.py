from databases.preprocessor_eval_query_repository import PreprocessorEvalQueryRepository


class Verifier:
    """Verifies preprocessed output existence in the database"""

    def __init__(self):
        self.query = PreprocessorEvalQueryRepository()

    def verify_output_file(self, obj_id):
        """
        Check if preprocessed output exists in the database.
        """
        exists = self.query.exists(obj_id)
        return exists, None

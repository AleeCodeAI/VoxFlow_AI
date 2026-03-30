import json


class Storage:
    """Handles saving evaluation results to JSON file"""

    def __init__(self, results_file: str):
        self.results_file = results_file

    def save_results(self, results):
        """
        Save all evaluation results to JSON file.
        """
        results_data = [result.model_dump(mode='json') for result in results]

        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, default=str)
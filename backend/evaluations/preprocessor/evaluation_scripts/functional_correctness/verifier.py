import os
import json


class Verifier:
    """Verifies preprocessed output existence in the database"""

    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(
            BASE_DIR, "..", "..", "..", "..", "databases", "preprocessings.jsonl"
        )

    def verify_output_file(self, obj_id):
        """
        Check if preprocessed output exists in the database file.
        """
        if not os.path.exists(self.db_path):
            return False, None

        with open(self.db_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get("id") == obj_id:
                        return True, data.get("preprocessed_transcription", "")

        return False, None


if __name__ == "__main__":
    verifier = Verifier()
    print("==" * 15)
    print(
        verifier.verify_output_file("e6288c87-afeb-4260-9f08-f7eae9ed3e7d")
    )  # Replace with actual ID to test

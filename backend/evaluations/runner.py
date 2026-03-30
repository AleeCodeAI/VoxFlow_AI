#from evaluations.preprocessor.evaluation_scripts.ai_judge import AIJudge
from evaluations.preprocessor.evaluation_scripts.functional_correctness import EvaluationPipeline

if __name__ == "__main__":
    pipeline = EvaluationPipeline()
    pipeline.run()
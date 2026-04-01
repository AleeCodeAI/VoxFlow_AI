#from evaluations.preprocessor.evaluation_scripts.ai_judge import AIJudge
#from evaluations.preprocessor.evaluation_scripts.functional_correctness import EvaluationPipeline
from evaluations.transcriber import TranscriptionFunctionalEvaluator
   
if __name__ == "__main__":
    evaluator = TranscriptionFunctionalEvaluator()
    evaluator.evaluate()
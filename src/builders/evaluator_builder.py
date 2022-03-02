from src.core.evaluators import AccEvaluator

def build(eval_config, logger):
    evaluator = AccEvaluator(logger)

    logger.infov('Evaluator is build.')
    return evaluator

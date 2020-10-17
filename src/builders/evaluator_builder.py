from src.core.evaluators import Evaluator

def build(eval_config, logger):
    evaluator = Evaluator(logger)

    logger.infov('Evaluator is build.')
    return evaluator

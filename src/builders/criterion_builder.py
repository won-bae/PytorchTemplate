from src.core.criterions import CustomCriterion


def build(train_config, logger):
    criterion_params = train_config.get('criterion', {})
    criterion = CustomCriterion(**criterion_params)

    logger.infov('Criterion is built.')
    return criterion

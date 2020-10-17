from src.core.models import vgg

MODELS = {
    'vgg16': vgg.vgg16,
}

def build(model_config, logger):
    backbone = model_config['backbone']
    model_params = model_config.copy()
    model_params['logger'] = logger

    # Build a model
    if backbone in MODELS:
        model = MODELS[backbone](model_params)
    else:
        logger.error(
            'Specify valid backbone or model type among {}.'.format(MODELS.keys())
        ); exit()

    return model


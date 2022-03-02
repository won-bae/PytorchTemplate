from src.core.models import resnet

MODELS = {
    'resnet18': resnet.resnet18,
}

def build(model_config, logger):
    backbone = model_config['backbone']
    model_params = {
        'pretrained': model_config['pretrained']
    }

    # Build a model
    if backbone in MODELS:
        model = MODELS[backbone](**model_params)
    else:
        logger.error(
            'Specify valid backbone or model type among {}.'.format(MODELS.keys())
        ); exit()

    return model


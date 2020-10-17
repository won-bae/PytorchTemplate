from src.core.meters import MultiMeter

def build(num_batches, logger):
    meter = MultiMeter('loss meter', logger, ['loss'], num_batches, fmt=':f')

    logger.infov('Loss meter is built.')
    return meter

class MultiMeter(object):

    def __init__(self, name, logger, meters, num_batches, fmt=':f'):
        self.name = name
        self.meters = {}
        for meter_name in meters:
            self.meters[meter_name] = AverageEpochMeter(meter_name, logger, num_batches, fmt)
        self.reset()

    def reset(self):
        for meter in self.meters:
            meter.reset()

    def update(self, val_dict, batch_size):
        for meter in self.meters:
            meter.update(val_dict[meter], batch_size)

    def print_log(self, val_dict, epoch, batch_idx):
        log = ''
        for meter in self.meters:
            log += meter + ': {:.4f} '.format(val_dict[meter])

        self.logger.info(
            '[Epoch {}] Train batch {}/{}'.format(epoch, batch_idx+1))
        self.logger.info(log)


class AverageEpochMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

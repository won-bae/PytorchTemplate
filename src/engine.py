import os
import torch
import time
from torch.utils.tensorboard import SummaryWriter

from src.utils import util
from src.builders import model_builder, dataloader_builder, checkpointer_builder,\
                         optimizer_builder, criterion_builder, scheduler_builder,\
                         meter_builder, evaluator_builder


class BaseEngine(object):

    def __init__(self, config_path, logger, save_dir):
        # Assign a logger and save dir
        self.logger = logger
        self.save_dir = save_dir

        # Load configurations
        config = util.load_config(config_path)
        self.model_config = config['model']
        self.train_config = config['train']
        self.eval_config = config['eval']
        self.data_config = config['data']

        # Determine which device to use
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        if device == 'cpu':
            self.logger.warn('GPU is not available.')
        else:
            self.logger.warn('{} GPU(s) is/are available.'.format(
                torch.cuda.device_count()))

        # Determine the eval standard
        self.eval_standard = self.eval_config['standard']

        # Load a summary writer
        log_dir = os.path.join(self.save_dir, 'logs')
        self.writer = SummaryWriter(log_dir=log_dir)

    def run(self):
        pass

    def evaluate(self):
        pass


class Engine(BaseEngine):

    def __init__(self, config_path, logger, save_dir):
        super(Engine, self).__init__(config_path, logger, save_dir)

    def _build(self, mode='train'):
        # Build a dataloader
        self.dataloaders = dataloader_builder.build(
            self.data_config, self.logger)

        # Build a model
        self.model = model_builder.build(
            self.model_config, self.logger)

        # Use multi GPUs if available
        if torch.cuda.device_count() > 1:
            self.model = util.DataParallel(self.model)
        self.model.to(self.device)

        # Build an optimizer, scheduler and criterion
        self.optimizer = optimizer_builder.build(
            self.train_config['optimizer'], self.model.parameters(), self.logger)
        self.scheduler = scheduler_builder.build(
            self.train_config, self.optimizer, self.logger)
        self.criterion = criterion_builder.build(
            self.train_config, self.logger)
        self.loss_meter = meter_builder.build(self.logger)
        self.evaluator = evaluator_builder.build(
            self.eval_config, self.logger)

        # Build a checkpointer
        self.checkpointer = checkpointer_builder.build(
            self.save_dir, self.logger, self.model, self.optimizer,
            self.scheduler, self.eval_standard)
        checkpoint_path = self.model_config.get('checkpoint_path', '')
        self.misc = self.checkpointer.load(
            mode, checkpoint_path, use_latest=False)

    def run(self):
        start_epoch, num_steps = 0, 0
        num_epochs = self.train_config.get('num_epochs', 100)
        checkpoint_step = self.train_config.get('checkpoint_step', 10000)

        self._build(mode='train')

        self.logger.info(
            'Train for {} epochs starting from epoch {}'.format(num_epochs, start_epoch))

        # Start training
        for epoch in range(start_epoch, start_epoch + num_epochs):
            train_start = time.time()
            num_steps = self._train_one_epoch(epoch, num_steps, checkpoint_step)
            train_time = time.time() - train_start

            lr = self.scheduler.get_lr()[0]
            self.logger.infov(
                '[Epoch {}] with lr: {:5f} completed in {:3f} - train loss: {:4f}'\
                .format(epoch, lr, train_time, self.loss_meter.avg))
            self.writer.add_scalar('Train/learning_rate', lr, global_step=num_steps)

            self.scheduler.step()
            self.loss_meter.reset()

            # Evaluate
            if epoch - start_epoch > 0.0 * num_epochs:
                eval_metrics = self._evaluate_once(epoch, num_steps)
                self.checkpointer.save(epoch, num_steps, eval_metrics)
                self.logger.info(
                    '[Epoch {}] - {}: {:4f}'.format(
                        epoch, self.eval_standard, eval_metrics[self.eval_standard]))
                self.logger.info(
                    '[Epoch {}] - best {}: {:4f}'.format(
                        epoch, self.eval_standard, self.checkpointer.best_eval_metric))



    def _train_one_epoch(self, epoch, num_steps, checkpoint_step):
        dataloader = self.dataloaders['train']
        self.model.train()

        for i, input_dict in enumerate(dataloader):
            input_dict = util.to_device(input_dict, self.device)

            # Forward propagation
            output_dict = self.model(input_dict)
            output_dict['labels'] = input_dict['labels']

            # Compute losses
            losses = self.criterion(output_dict)
            loss = losses['loss']

            # Backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print losses
            batch_size = input_dict['inputs'].size(0)
            self.loss_meter.update(loss, batch_size)

            # Save a checkpoint
            num_steps += batch_size
            if num_steps % checkpoint_step == 0:
                self.checkpointer.save(epoch, num_steps)
                #self.logger.info(
                #    'A checkpoint at epoch={}, num_step={} has been saved.'.format(
                #        epoch, num_steps))

        self.writer.add_scalar('Train/loss', self.loss_meter.avg, global_step=num_steps)

        torch.cuda.empty_cache()
        return num_steps

    def evaluate(self, data_type='val'):
        self._build(mode='eval')
        eval_metrics = self._evaluate_once(0, 0, data_type=data_type)
        self.logger.info(
            '[Eval] - {}: {:4f}'.format(
                self.eval_standard, eval_metrics[self.eval_standard]))

    def _evaluate_once(self, epoch, num_steps, data_type='val'):
        dataloader = self.dataloaders[data_type]
        num_batches = len(dataloader)

        self.model.eval()
        for i, input_dict in enumerate(dataloader):
            with torch.no_grad():
                input_dict = util.to_device(input_dict, self.device)

                # Forward propagation
                output_dict = self.model(input_dict)
                output_dict['labels'] = input_dict['labels']

                # Print losses
                self.evaluator.update(output_dict)

                self.logger.info('[Epoch {}] Evaluation batch {}/{}'.format(
                    epoch, i+1, num_batches))

        self.evaluator.print_log(epoch, num_steps)
        torch.cuda.empty_cache()
        eval_metric = self.evaluator.compute()
        return {self.eval_standard: eval_metric}



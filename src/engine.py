import os
import torch
import time
from torch.utils.tensorboard import SummaryWriter

from src.utils import util
from src.builders import model_builder, dataloader_builder, checkpointer_builder,\
                         optimizer_builder, criterion_builder, scheduler_builder,\
                         meter_builder, evaluator_builder


class BaseEngine(object):

    def __init__(self, config_path, logger, train_dir, eval_dir=None):
        # Assign a logger
        self.logger = logger

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
            self.logger.warn('GPU is available.')

        # Determine the eval standard
        self.eval_standard = self.eval_config['standard']

        # Load a summary writer
        if eval_dir is not None:
            log_dir = os.path.join(eval_dir, 'logs')
        else:
            log_dir = os.path.join(train_dir, 'logs')
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self):
        pass

    def evaluate(self):
        pass


class Engine(BaseEngine):

    def __init__(self, mode, config_path, logger, train_dir, eval_dir=None):
        super(Engine, self).__init__(config_path, logger, train_dir, eval_dir)

        # Build a dataloader
        self.dataloader = dataloader_builder.build(
            mode, self.data_config, self.logger)

        # Build a model
        self.model = model_builder.build(
            self.model_config, self.logger)

        # Use multi GPUs if available
        if mode == 'train' and torch.cuda.device_count() > 1:
            self.model = util.DataParallel(self.model)
        self.model.to(self.device)

        # Build an optimizer, scheduler and criterion
        self.optimizer, self.scheduler = None, None
        if mode == 'train':
            self.optimizer = optimizer_builder.build(
                self.train_config['optimizer'], self.model.parameters(), self.logger)
            self.scheduler = scheduler_builder.build(
                self.train_config, self.optimizer, self.logger)
            self.criterion = criterion_builder.build(
                self.train_config, self.logger)
            self.loss_meter = meter_builder.build(
                self.train_config, len(self.dataloader), self.logger)
        else:
            self.evaluator = evaluator_builder.build(
                self.eval_config, self.logger)

        # Build a checkpointer
        self.checkpointer = checkpointer_builder.build(
            mode, train_dir, self.logger, self.model, self.optimizer,
            self.scheduler, self.eval_standard)
        checkpoint_path = self.model_config.get('checkpoint_path', '')
        self.misc = self.checkpointer.load(checkpoint_path, use_latest=False)


    def train(self):
        start_epoch, num_steps = 0, 0
        num_epochs = self.train_config.get('num_epochs', 100)
        checkpoint_step = self.train_config.get('checkpoint_step', 10000)

        self.logger.info(
            'Train for {} epochs starting from epoch {}'.format(num_epochs, start_epoch))

        # Start training
        for epoch in range(start_epoch, start_epoch + num_epochs):
            train_start = time.time()
            num_steps = self._train_one_epoch(epoch, num_steps, checkpoint_step)
            train_time = time.time() - train_start

            lr = self.scheduler.get_lr

            self.logger.infov(
                '[Epoch {}] with lr: {:5f} completed in {:3f} - train loss: {:4f}'\
                .format(epoch, lr, train_time, self.loss_meter.loss))
            self.writer.add_scalar('Train/learning_rate', lr, global_step=num_steps)

            self.scheduler.step()
            self.loss_meter.reset()


    def _train_one_epoch(self, epoch, num_steps, checkpoint_step):
        self.model.train()

        for i, input_dict in enumerate(self.dataloader):
            input_dict = util.to_device(input_dict, self.device)

            # Forward propagation
            output_dict = self.model(input_dict)

            # Compute losses
            losses = self.criterion(**output_dict)
            loss = losses['loss']

            # Backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print losses
            batch_size = input_dict['images'].size(0)
            self.loss_meter.update(losses, batch_size)
            self.loss_meter.print_log(loss.item(), epoch, i+1)

            # Save a checkpoint
            num_steps += batch_size
            if num_steps % checkpoint_step:
                self.checkpointer.save(epoch, num_steps)
                self.logger.info(
                    'A checkpoint at epoch={}, num_step={} has been saved.'.format(
                        epoch, num_steps))

        self.writer.add_scalar('Train/loss', self.loss_meter.loss, global_step=num_steps)

        torch.cuda.empty_cache()
        return num_steps


    def evaluate(self):
        def _get_misc_info(misc):
            infos = ['epoch', 'num_steps', 'cehckpoint_path']
            return (misc[info] for info in infos)

        epoch, num_steps, current_checkpoint_path = _get_misc_info(self.misc)
        last_evaluated_checkpoint_path = None
        while True:
            if last_evaluated_checkpoint_path == current_checkpoint_path:
                self.logger.warn('Found already evaluated checkpoint. Will try again in 60 seconds.')
                time.sleep(60)
            else:
                self._evaluate_once(epoch, num_steps)
                last_evaluated_checkpoint_path = current_checkpoint_path
                self.checkpointer.save(
                    epoch, num_steps, self.evaluator.metrics, self.eval_dir)

            # Reload a checkpoint. Break if file path was given as checkpoint path.
            checkpoint_path = self.model_config.get('checkpoint_path', '')
            if os.path.isfile(checkpoint_path): break
            misc = self.checkpointer.load(checkpoint_path, use_latest=True)
            epoch, num_step, current_checkpoint_path = _get_misc_info(misc)

            # Reset the evaluator
            self.evaluator.reset()


    def _evaluate_once(self, epoch, num_steps):
        num_batches = len(self.dataloader)

        self.model.eval()
        for i, input_dict in enumerate(self.dataloader):
            with torch.no_grad():
                input_dict = util.to_device(input_dict, self.device)

                # Forward propagation
                output_dict = self.model(input_dict)

                # Print losses
                self.evaluator.update(output_dict)

                self.log.info('[Epoch {}] Evaluation batch {}/{}'.format(
                    epoch, i+1, num_batches))

        self.evaluator.print_log(epoch, num_steps)
        torch.cuda.empty_cache()



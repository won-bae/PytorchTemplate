import os
import time
import torch


class CustomCheckpointer(object):
    def __init__(self, mode, checkpoint_dir, logger, model,
                 optimizer=None, scheduler=None, eval_standard='accuracy'):
        self.logger = logger
        self.mode = mode
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        if mode == 'train':
            self.logger.infov(
                'Directory {} to save checkpoints is ready.'.format(self.checkpoint_dir))
        else:
            self.eval_standard = eval_standard
            self.best_eval_metric = 0

        self.logger.infov('Checkpointer is built.')


    def save(self, epoch, num_steps, eval_metrics=None, eval_dir=None):
        # Save the given checkpoint in train time and best checkpoint in eval time.
        if self.mode == 'train':
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 'checkpoint' + '_' + str(epoch) + '_' + str(num_steps) + '.pth')
        else:
            eval_metric = eval_metrics[self.eval_standard]
            if eval_metric <= self.best_eval_metric:
                return
            self.best_eval_metric = eval_metric
            checkpoint_path = os.path.join(eval_dir, 'checkpoint_best.pth')

        model_params = {'epoch': epoch, 'num_step': num_steps}
        if torch.cuda.device_count() > 1:
            model_params['model_state_dict'] = self.model.module.state_dict()
        else:
            model_params['model_state_dict'] = self.model.state_dict()

        if self.mode == 'train':
            model_params['optimizer_state_dict'] = self.optimizer.state_dict()
            model_params['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(model_params, checkpoint_path)
        self.logger.info(
            'A checkpoint is saved for epoch={}, steps={}.'.format(epoch, num_steps))

        # Update the checkpoint record
        if self.mode != 'train':
            best_checkpoint_info = {'epoch': epoch, 'num_step': num_steps}
            best_checkpoint_info.update(eval_metrics)

            self._record_best_checkpoint(best_checkpoint_info, eval_dir)
        else:
            self._record_last_checkpoint(last_checkpoint_path=checkpoint_path)


    def load(self, checkpoint_path=None, use_latest=True):
        strict = True
        if self.mode == 'train':
            if self._has_checkpoint() and use_latest: # Override argument with existing checkpoint
                checkpoint_path = self._get_checkpoint_path()
            if not checkpoint_path:
                self.logger.info("No checkpoint found. Initializing model from scratch.")
                return {}
        else:
            if not checkpoint_path:
                while not self._has_checkpoint():
                    self.logger.warn('No checkpoint available. Wait for 60 seconds.')
                    time.sleep(60)
                checkpoint_path = self._get_checkpoint_path()

        self.logger.info("Loading checkpoint from {}".format(checkpoint_path))
        checkpoint = self._load_checkpoint(checkpoint_path)

        self.model.load_state_dict(
            checkpoint.pop('model_state_dict'), strict=strict)

        if strict:
            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(checkpoint_path))
                self.optimizer.load_state_dict(checkpoint.pop('optimizer_state_dict'))
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(checkpoint_path))
                self.scheduler.load_state_dict(checkpoint.pop('scheduler_state_dict'))

        return checkpoint

    def _freeze(self):
        for param in self.model.layers.parameters():
            param.requires_grad = False

    def _has_checkpoint(self):
        record_path = os.path.join(self.checkpoint_dir, "last_checkpoint")
        return os.path.exists(record_path)

    def _get_checkpoint_path(self):
        record_path = os.path.join(self.checkpoint_dir, "last_checkpoint")
        try:
            with open(record_path, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            self.logger.warn('If last_checkpoint file doesn not exist, maybe because \
                              it has just been deleted by a separate process.')
            last_saved = ''
        return last_saved

    def _record_last_checkpoint(self, last_checkpoint_path):
        record_path = os.path.join(self.checkpoint_dir, 'last_checkpoint')
        with open(record_path, 'w') as f:
            f.write(last_checkpoint_path)

    def _record_best_checkpoint(self, best_checkpoint_info, eval_dir):
        record_path = os.path.join(eval_dir, 'best_checkpoint')
        with open(record_path, 'w') as f:
            f.write(str(best_checkpoint_info))

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        checkpoint['checkpoint_path'] = checkpoint_path
        return checkpoint

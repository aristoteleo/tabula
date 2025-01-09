from typing import Dict

import torch


class FinetuneBase:
    '''
    Base class for fine-tuning tasks.
    '''
    def __init__(self,
                 model: torch.nn.Module,
                 task_params: Dict = None
                 ):
        self.task_params = task_params
        self.model = model

    def configure_optimizers(self, pl_object):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, pl_object):
        raise NotImplementedError

    def on_train_epoch_end(self, pl_object):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx, pl_object):
        raise NotImplementedError

    def on_validation_epoch_end(self, pl_object):
        raise NotImplementedError

    def define_metrics(self):
        raise NotImplementedError
    
    def on_after_backward(self):
        raise NotImplementedError

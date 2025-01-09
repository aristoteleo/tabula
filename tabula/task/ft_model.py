from typing import Dict, Optional

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning import seed_everything
from tabula.task.ft_factory import (
                                    # CellAnnotationFinetuneWrapper,
                                    # GenePerturbationPredictionFinetuneWrapper,
                                    IntegrationFinetuneWrapper,
                                    # ImputationFinetuneWrapper
                                    )
from tabula.embedder.corruption import CorruptionGenerator
from tabula.model.loss import ContrastiveLoss, DistillLoss, ReconstructionLoss, masked_mse_loss, criterion_neg_log_bernoulli
from tabula.model.transfomer.transformer import TabulaTransformer
from tabula.task.utils import FinetuneConfig


class FinetuneModel(pl.LightningModule):
    """
    A PyTorch-Lightning module for fine-tuning models.
    This module is responsible for training and evaluating the model on the downstream task.
    """
    def __init__(self,
                 finetune_type: str,
                 model: TabulaTransformer = None,
                 finetune_config: FinetuneConfig = None,
                 seed: Optional[int] = 42,
                 device: Optional[str] = 'cuda',
                 record_best_model: Optional[bool] = False,
                 task_params: Optional[Dict] = None,
                 enable_wandb: Optional[bool] = False
                 ):
        super().__init__()
        if finetune_type is None:
            raise ValueError('finetune_type must be provided')

        seed_everything(seed)

        self.model = model
        self.finetune_type = finetune_type
        self.finetune_config = finetune_config
        self.task_params = task_params
        self.enable_wandb = enable_wandb
        self.finetune_task = self._select_task()

        self.record_best_model = record_best_model
        self.criterion_masked_mse = masked_mse_loss
        self.criterion_neg_log_bernoulli = criterion_neg_log_bernoulli
        if self.finetune_config.get_finetune_param('do_dab'):
            self.criterion_dab = nn.CrossEntropyLoss()

        self.contrastive_loss = (
            DistillLoss(temperature=finetune_config.get_finetune_param('temperature')).to(self.device)
            if finetune_config.get_finetune_param('objective') in ["self_distill"]
            else ContrastiveLoss(temperature=finetune_config.get_finetune_param('temperature')).to(self.device)
        )
        self.reconstruction_loss = ReconstructionLoss()
        self.contrastive_fn = CorruptionGenerator(
            mode=finetune_config.get_finetune_param('augmentation_mode'),
            corruption_rate=finetune_config.get_finetune_param('corruption_rate')
        )

        self.to_device = device
        self.to(self.device)
        self.save_hyperparameters()

    def _select_task(self):
        if self.finetune_type == 'cell_type_annotation':
            # return CellAnnotationFinetuneWrapper(self.model, self.task_params)
            raise NotImplementedError('Will be released soon ...')
        elif self.finetune_type == 'integration':
            return IntegrationFinetuneWrapper(self.model, self.task_params)
        elif self.finetune_type == 'gene_perturbation':
            # return GenePerturbationPredictionFinetuneWrapper(self.model, self.task_params)
            raise NotImplementedError('Will be released soon ...')
        elif self.finetune_type == 'imputation':
            # return ImputationFinetuneWrapper(self.model, self.task_params)
            raise NotImplementedError('Will be released soon ...')
        else:
            raise ValueError(f'Unsupported finetune type {self.finetune_type}')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.finetune_task.training_step(batch, batch_idx, pl_object=self)

    def on_train_epoch_end(self) -> None:
        self.finetune_task.on_train_epoch_end(pl_object=self)

    def validation_step(self, batch, batch_idx):
        if self.trainer.global_step == 0:
            self._define_metrics()
        loss = self.finetune_task.validation_step(batch, batch_idx, pl_object=self)
        self.log('valid/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def on_validation_epoch_end(self) -> None:
        self.finetune_task.on_validation_epoch_end(pl_object=self)

    def configure_optimizers(self):
        return self.finetune_task.configure_optimizers(pl_object=self)

    def _define_metrics(self):
        self.finetune_task.define_metrics()

    def _shared_step(self, batch):
        pass

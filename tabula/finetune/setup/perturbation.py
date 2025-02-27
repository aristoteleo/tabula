from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as pl
from anndata import AnnData
from gears import PertData
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from tabula import logger
from tabula.finetune.model import perturbation
from tabula.finetune.utils import FinetuneConfig


class GenePerturbationPrediction:
    """
    The class is used to perform gene expression under perturbation condition
    """
    def __init__(self,
                 config: FinetuneConfig,
                 pert_data: PertData,
                 tabula_model: pl.LightningModule,
                 wandb_logger: WandbLogger,
                 device: str,
                 batch_size: int,
                 gene_ids: Optional[Union[List, np.ndarray]],
                 perts_to_plot: Optional[List] = None,
                 reverse_perturb: bool = False,
                 pert_data_eval: PertData = None,
                 ):
        self.config = config
        self.tabula_model = tabula_model
        self.wandb_logger = wandb_logger
        self.device = device
        self.save_path = self.config.get_finetune_param('save_folder')
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.pert_data = pert_data
        self.pert_data_eval = pert_data_eval
        self.gene_ids = gene_ids
        self.perts_to_plot = perts_to_plot
        self.reverse_perturb = reverse_perturb

    def finetune(self):
        """
        Finetune the tabula model for perturbation prediction task
        """
        seed_everything(self.config.seed)
        finetune_method = self.config.get_finetune_param('method')
        if finetune_method == 'light':
            max_epochs = self.config.get_finetune_param('light_epochs')
            logger.info(f"Finetune method: {finetune_method}. Max epochs: {max_epochs}")
        elif finetune_method == 'heavy':
            max_epochs = self.config.get_finetune_param('max_epochs')
            early_stopping_callback = EarlyStopping('valid/total_loss',
                                                    patience=self.config.get_finetune_param('patience'))
            logger.info(
                f"Finetune method: {finetune_method}. Max epochs: {max_epochs}. Patience: {early_stopping_callback.patience} ")
        else:
            raise ValueError(f"Finetune method {finetune_method} not supported.")

        self.pl_model = perturbation.FinetuneModel(
            model=self.tabula_model,
            save_path=self.save_path,
            pert_data=self.pert_data,
            pert_data_eval=self.pert_data_eval,
            gene_ids=self.gene_ids,
            perts_to_plot=self.perts_to_plot,
            config=self.config,
            reverse_perturb=self.reverse_perturb
        ).to(self.device)

        trainer_args = {
            'max_epochs': max_epochs,
            'default_root_dir': self.save_path,
            'callbacks': [early_stopping_callback] if finetune_method == 'heavy' else None,
            'precision': 16 if self.config.get_finetune_param('amp') else 32,
        }
        cuda_index = int(self.device.split(":")[-1])
        trainer = pl.Trainer(**trainer_args, logger=self.wandb_logger, gpus=[cuda_index])

        trainer.fit(model=self.pl_model,
                    train_dataloaders=self.pert_data.dataloader["train_loader"],
                    val_dataloaders=self.pert_data.dataloader["val_loader"]
                    )
        logger.info(f"Finetune finished.")

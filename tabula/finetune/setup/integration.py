from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from tabula import logger
from tabula.finetune.model import integration
from tabula.finetune.utils import FinetuneConfig
from anndata import AnnData


class MultiOmicsIntegration:
    """
    The class is used to perform multi-omics and -batch integration
    """
    def __init__(self,
                 config: FinetuneConfig,
                 tabula_model: pl.LightningModule,
                 wandb_logger: WandbLogger,
                 device: str,
                 batch_size: int,
                 gene_ids: Optional[Union[List, np.ndarray]],
                 eval_adata: AnnData,
                 dataloaders: dict,
                 ):
        self.config = config
        self.tabula_model = tabula_model
        self.wandb_logger = wandb_logger
        self.device = device
        self.save_path = self.config.get_finetune_param('save_folder')
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.gene_ids = gene_ids
        self.eval_adata = eval_adata
        self.dataloaders = dataloaders

    def finetune(self):
        seed_everything(self.config.seed)
        finetune_method = self.config.get_finetune_param('method')
        if finetune_method == 'light':
            max_epochs = self.config.get_finetune_param('light_epochs')
            logger.info(f"Finetune method: {finetune_method}. Max epochs: {max_epochs}")
        elif finetune_method == 'heavy':
            max_epochs = self.config.get_finetune_param('max_epochs')
            early_stopping_callback = EarlyStopping('valid/total_loss',
                                                    patience=self.config.get_finetune_param('patience'))
            logger.info(f"Finetune method: {finetune_method}. Max epochs: {max_epochs}. Patience: {early_stopping_callback.patience}.")
        else:
            raise ValueError(f"Finetune method {finetune_method} not supported.")
        
        self.pl_model = integration.FinetuneModel(
            model=self.tabula_model,
            config=self.config,
            save_path=self.save_path,
            gene_ids=self.gene_ids,
            eval_adata=self.eval_adata,
            test_loader=self.dataloaders["test_loader"],
        ).to(self.device)

        trainer_args = {
            'max_epochs': max_epochs,
            'default_root_dir': self.save_path,
            'callbacks': [early_stopping_callback] if finetune_method == 'heavy' else None,
            'gradient_clip_val': self.config.get_finetune_param('gradient_clip_val'),
        }
        cuda_index = int(self.device.split(":")[-1])
        trainer = pl.Trainer(**trainer_args, logger=self.wandb_logger, gpus=[cuda_index])

        trainer.fit(model=self.pl_model,
                    train_dataloaders=self.dataloaders["train_loader"],
                    val_dataloaders=self.dataloaders["val_loader"]
                    )
        logger.info(f"Finetune finished.")
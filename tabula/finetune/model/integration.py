from typing import List, Optional, Union
import json
import traceback

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pytorch_lightning as pl
import wandb
import torch
import torch.multiprocessing
from tabula import logger
from tabula.finetune.utils import eval_scib_metrics
from tabula.embedder.corruption import CorruptionGenerator
from tabula.finetune.utils import FinetuneConfig
from tabula.finetune.preprocessor import random_mask_value
from tabula.model.loss import ReconstructionLoss, masked_mse_loss, criterion_neg_log_bernoulli
from torch import nn
from anndata import AnnData


class FinetuneModel(pl.LightningModule):
    """
    The class is used to finetune the tabula model for multi-omics and -batch inetgration
    """
    def __init__(self,
                 model: nn.Module,
                 config: FinetuneConfig,
                 save_path: Optional[str],
                 gene_ids: Optional[Union[List, np.ndarray]],
                 eval_adata: Optional[AnnData] = None,
                 test_loader: Optional[torch.utils.data.DataLoader] = None,
                 ):
        """
        Parameters:
            model: nn.Module
                The tabula model to be finetuned
            config: FinetuneConfig
                finetune configure object
            save_path: str
                The path to save the finetuned model and finetune analysis results
            gene_ids: List or np.ndarray
                The gene ids
            eval_adata: AnnData
                The AnnData object for evaluation
            test_loader: torch.utils.data.DataLoader
                The dataloader for test set
        """
        super(FinetuneModel, self).__init__()
        self.model = model
        self.config = config
        self.save_path = save_path
        self.gene_ids = gene_ids
        self.eval_adata = eval_adata
        self.test_loader = test_loader

        self.contrastive_fn = CorruptionGenerator(
            mode=self.config.get_finetune_param('augmentation_mode'),
            corruption_rate=self.config.get_finetune_param('corruption_rate')
        )
        self.reconstruction_loss = ReconstructionLoss()
        self.criterion_masked_mse = masked_mse_loss
        if self.config.get_finetune_param('do_dab'):
            self.criterion_dab = nn.CrossEntropyLoss()

        self.val_loss_list = []
        self.best_val_loss = float('inf')

    def configure_optimizers(self):
        """
        Lightening method to configure the optimizer
        """
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.config.get_finetune_param('learning_rate'),
                                      weight_decay=self.config.get_finetune_param('weight_decay'))
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        Lightening method to perform training step
        Calculate the loss and log the loss
        """
        loss, _ = self._integration_forward(batch, 'train')
        return {'loss': loss}
    
    def on_train_epoch_end(self) -> None:
        """
        Lightening method to perform operations at the end of training epoch
        Record learning rate at the end of each epoch
        """
        for param_group in self.trainer.optimizers[0].param_groups:
            self.log('train/learning_rate', param_group['lr'])

    def validation_step(self, batch, batch_idx) -> None:
        """
        Lightening method to perform validation step
        Calculate the loss and record the loss
        """
        loss, _ = self._integration_forward(batch, 'validation')
        self.val_loss_list.append(loss)
        self.log('valid/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Lightening method to perform operations at the end of validation epoch
        1. Evaluate the model on test set
        2. Log the validation loss and save the best model
        """
        current_val_loss = torch.stack(self.val_loss_list).mean()
        self.val_loss_list.clear()
        if self.eval_adata is not None and current_val_loss < self.best_val_loss:
            self._scib_metrics2wandb()
            self.best_val_loss = current_val_loss
            torch.save(self.model.state_dict(),
                       f"{self.config.get_finetune_param('save_folder')}/best_model.pth")

    def _integration_forward(self, batch, loop_stage):
        """
        Forward pass for multi-omics and -batch integration

        Args:
            batch: data batch
            loop_stage: indicator of training or validation

        Returns:
            total_loss: total loss
            output: output of the model
        """
        genes, values, _, batch_info = batch
        total_loss = 0
        if self.config.get_finetune_param('do_rcs'):
            corrupted_value = self.contrastive_fn(values).to(self.device)
            reconstruction = self.model(genes=genes,
                                        values=corrupted_value,
                                        batch_info=batch_info,
                                        head='reconstruction')['reconstruction']
            reconstruction_loss = self.reconstruction_loss(values, reconstruction)
            total_loss = reconstruction_loss
            self.log(f'{loop_stage}/reconstruction_loss', reconstruction_loss,
                     on_step=True, on_epoch=True, prog_bar=True)

        masked_values = random_mask_value(values,
                                          mask_ratio=self.config.get_finetune_param('mask_rate'),
                                          mask_value=self.config.get_finetune_param('mask_value'))
        output = self.model(genes=genes,
                            values=masked_values.to(self.device),
                            batch_info=batch_info,
                            head=None,
                            do_dab=self.config.get_finetune_param('do_dab'),
                            do_mgm=self.config.get_finetune_param('do_mgm'),
                            do_cmgm=self.config.get_finetune_param('do_cmgm'),
                            do_sample=self.config.get_finetune_param('explicit_zero_prob'))
        masked_positions = (masked_values.to(self.device).eq(values))
        loss_mgm = self.criterion_masked_mse(output['mgm_pred'], values, masked_positions)
        loss_cmgm = self.criterion_masked_mse(output['cmgm_output'], values, masked_positions)
        loss_dab = self.criterion_dab(output['dab_output'], 
                                      batch_info.long()) * self.config.get_finetune_param('dab_weight')
        total_loss = total_loss + loss_dab + loss_cmgm + loss_mgm

        if self.config.get_finetune_param('explicit_zero_prob'):
            loss_zero_log_prob = criterion_neg_log_bernoulli(output["mgm_zero_probs"], values, masked_positions)
            loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(output["cmgm_zero_probs"], values, masked_positions)
            total_loss += loss_zero_log_prob + loss_gepc_zero_log_prob
            self.log(f'{loop_stage}/zero_log_prob_loss', loss_cmgm, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f'{loop_stage}/gepc_zero_log_prob_loss', loss_cmgm, on_step=True, on_epoch=True, prog_bar=True)

        self.log(f'{loop_stage}/mgm_loss', loss_mgm, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{loop_stage}/cmgm_loss', loss_cmgm, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{loop_stage}/dab_loss', loss_dab, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss, output

    def _scib_metrics2wandb(self):
        """
        Evaluate cls cell embeddings
        Called by validation step
        """
        logger.info(f"Start evaluating scib_metrics...")
        save_folder = self.config.get_finetune_param('save_folder')
        cell_embeddings = []
        # feature_tokenizer_embeddings = []

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                batch = [b.to(self.device) for b in batch]
                genes, values, _, batch = batch
                eval_output = self.model(genes=genes,
                                         values=values,
                                         batch_info=batch,
                                         head=None)
                cell_embeddings.append(eval_output['cell_embed'].to('cpu'))

        cell_embeddings = torch.cat(cell_embeddings).cpu().numpy()
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )
        self.eval_adata.obsm['tabula_embed'] = cell_embeddings
        
        results = {}
        try:
            results = eval_scib_metrics(self.eval_adata)
            if self.config.get_finetune_param('enable_wandb'):
                wandb.log({'test/metrics': results})
            else:
                logger.info(f"Test metrics: {results}")
        except Exception as e:
            traceback.print_exc()
            logger.error(e)

        # save eval_data and results to local
        self.eval_adata.write(f'{save_folder}/best_eval_data.h5ad')
        with open(f'{save_folder}/best_results.json', 'w') as f:
            json.dump(results, f)

        sc.pp.neighbors(self.eval_adata, use_rep="tabula_embed")
        sc.tl.umap(self.eval_adata, min_dist=0.3)
        with plt.rc_context({"figure.figsize": (6, 4), "figure.dpi": (600)}):
            fig = sc.pl.umap(
                self.eval_adata,
                color=["str_batch"],
                title=[f"batch, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
                frameon=False,
                return_fig=True,
                show=False,
            )
        fig.savefig(f'{save_folder}/batch_umap.png', bbox_inches='tight')

        sc.pp.neighbors(self.eval_adata, use_rep="tabula_embed")
        sc.tl.umap(self.eval_adata, min_dist=0.3)
        with plt.rc_context({"figure.figsize": (6, 4), "figure.dpi": (600)}):
            fig = sc.pl.umap(
                self.eval_adata,
                color=["celltype"],
                title=[f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",],
                frameon=False,
                return_fig=True,
                show=False,
            )
        fig.savefig(f'{save_folder}/celltype_umap.png', bbox_inches='tight')

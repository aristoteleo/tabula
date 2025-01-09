import json
import traceback
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import wandb
from tabula import logger
from tabula.task.base import FinetuneBase
from tabula.task.preprocessor import random_mask_value
from tabula.task.utils import eval_scib_metrics
from tabula.task.metrics import criterion_neg_log_bernoulli
from tqdm import tqdm


class IntegrationFinetuneWrapper(FinetuneBase):
    def __init__(self,
                 model: nn.Module,
                 task_params: Dict = None
                 ):
        super().__init__(model, task_params)
        self.task_params = task_params
        self.model = model
        self.test_loader = task_params['test_loader']
        self.eval_adata = task_params['eval_adata']
        self.val_loss_list = []
        self.best_val_loss = 1e9

    def configure_optimizers(self, pl_object):
        optimizer = torch.optim.AdamW(pl_object.model.parameters(),
                                      lr=pl_object.finetune_config.get_finetune_param('learning_rate'),
                                      weight_decay=pl_object.finetune_config.get_finetune_param('weight_decay'))
        return optimizer

    def training_step(self, batch, batch_idx, pl_object):
        batch = [b.to(pl_object.to_device) for b in batch]
        pl_object.to(pl_object.to_device)
        loss, _ = self._compute_loss(batch, pl_object, 'train')
        return {'loss': loss}

    def on_train_epoch_end(self, pl_object):
        pass

    def validation_step(self, batch, batch_idx, pl_object):
        batch = [b.to(pl_object.to_device) for b in batch]
        pl_object.to(pl_object.to_device)
        loss, _ = self._compute_loss(batch, pl_object, 'validation')
        self.val_loss_list.append(loss)
        return loss

    def on_validation_epoch_end(self, pl_object):
        current_val_loss = torch.stack(self.val_loss_list).mean().item()
        if self.eval_adata is not None and current_val_loss < self.best_val_loss:
            self._scib_metrics2wandb(pl_object)
            self.best_val_loss = current_val_loss
            torch.save(self.model.state_dict(),
                       f"{pl_object.finetune_config.get_finetune_param('save_folder')}/best_model.pth")
            logger.info(f"Best model saved at {pl_object.finetune_config.get_finetune_param('save_folder')}/best_model.pth") 
        self.val_loss_list = []

    def define_metrics(self):
        pass

    def _compute_loss(self, batch, pl_object, loop_stage):
        genes, values, _, batch_info = batch
        total_loss = 0
        if pl_object.finetune_config.get_finetune_param('do_rcs'):
            corrupted_value = pl_object.contrastive_fn(values).to(pl_object.to_device)
            reconstruction = pl_object.model(genes=genes,
                                             values=corrupted_value,
                                             batch_info=batch_info,
                                             head='reconstruction')['reconstruction']
            reconstruction_loss = pl_object.reconstruction_loss(values, reconstruction)
            total_loss = reconstruction_loss
            pl_object.log(f'{loop_stage}/reconstruction_loss', reconstruction_loss,
                          on_step=True, on_epoch=True, prog_bar=True)

        masked_values = random_mask_value(values,
                                          mask_ratio=pl_object.finetune_config.get_finetune_param('mask_rate'),
                                          mask_value=pl_object.finetune_config.get_finetune_param('mask_value'))
        output = pl_object.model(genes=genes,
                                 values=masked_values.to(pl_object.to_device),
                                 batch_info=batch_info,
                                 head=None,
                                 do_dab=pl_object.finetune_config.get_finetune_param('do_dab'),
                                 do_mgm=pl_object.finetune_config.get_finetune_param('do_mgm'),
                                 do_cmgm=pl_object.finetune_config.get_finetune_param('do_cmgm'),
                                 do_sample=pl_object.finetune_config.get_finetune_param('explicit_zero_prob'))
        masked_positions = (masked_values.to(pl_object.to_device).eq(values))
        loss_mgm = pl_object.criterion_masked_mse(output['mgm_pred'], values, masked_positions)
        loss_cmgm = pl_object.criterion_masked_mse(output['cmgm_output'], values, masked_positions)
        loss_dab = pl_object.criterion_dab(output['dab_output'], 
                                           batch_info.long()) * pl_object.finetune_config.get_finetune_param('dab_weight')
        total_loss = total_loss + loss_dab + loss_cmgm + loss_mgm

        if pl_object.finetune_config.get_finetune_param('explicit_zero_prob'):
            loss_zero_log_prob = criterion_neg_log_bernoulli(output["mgm_zero_probs"], values, masked_positions)
            loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(output["cmgm_zero_probs"], values, masked_positions)
            total_loss += loss_zero_log_prob + loss_gepc_zero_log_prob
            pl_object.log(f'{loop_stage}/zero_log_prob_loss', loss_cmgm, on_step=True, on_epoch=True, prog_bar=True)
            pl_object.log(f'{loop_stage}/gepc_zero_log_prob_loss', loss_cmgm, on_step=True, on_epoch=True, prog_bar=True)

        pl_object.log(f'{loop_stage}/mgm_loss', loss_mgm, on_step=True, on_epoch=True, prog_bar=True)
        pl_object.log(f'{loop_stage}/cmgm_loss', loss_cmgm, on_step=True, on_epoch=True, prog_bar=True)
        pl_object.log(f'{loop_stage}/dab_loss', loss_dab, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss, output

    def _scib_metrics2wandb(self, pl_object):
        """
        Evaluate cls cell embeddings
        Called by validation step
        """
        logger.info(f"Start evaluating scib_metrics...")
        pl_object.model.to(pl_object.to_device)
        save_folder = pl_object.finetune_config.get_finetune_param('save_folder')
        cell_embeddings = []
        feature_tokenizer_embeddings = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader,
                                           desc=f"Epoch {pl_object.current_epoch}: Evaluation Inference")):
                batch = [b.to(pl_object.to_device) for b in batch]
                genes, values, _, batch = batch
                eval_output = pl_object.model(genes=genes,
                                              values=values,
                                              batch_info=batch,
                                              head=None)
                cell_embeddings.append(eval_output['cell_embed'].to('cpu'))
                feature_tokenizers = eval_output['feature_tokenizer']
                aggregated_feature_tokenizer = torch.mean(feature_tokenizers, dim=1)
                feature_tokenizer_embeddings.append(aggregated_feature_tokenizer.to('cpu'))
                torch.cuda.empty_cache()

        cell_embeddings = torch.cat(cell_embeddings).cpu().numpy()
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )
        self.eval_adata.obsm['tabula_embed'] = cell_embeddings

        feature_tokenizer_embeddings = torch.cat(feature_tokenizer_embeddings).cpu().numpy()
        feature_tokenizer_embeddings = feature_tokenizer_embeddings / np.linalg.norm(
            feature_tokenizer_embeddings, axis=1, keepdims=True
        )
        self.eval_adata.obsm['feature_tokenizer_embed'] = feature_tokenizer_embeddings
        
        results = {}
        try:
            results = eval_scib_metrics(self.eval_adata)
            if pl_object.enable_wandb:
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

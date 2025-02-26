import io
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.multiprocessing
import wandb
from gears import PertData
from gears.inference import (compute_metrics, deeper_analysis,
                             non_dropout_analysis)
from gears.utils import create_cell_graph_dataset_for_prediction
from PIL import Image
from scgpt.utils import map_raw_id_to_vocab_id
from tabula import logger
from tabula.finetune.utils import FinetuneConfig
from tabula.model.loss import masked_mse_loss
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class FinetuneModel(pl.LightningModule):
    """
    The class is used to train model for predicting gene expression under perturbation condition
    """
    def __init__(self,
                 model: nn.Module,
                 config: FinetuneConfig,
                 save_path: Optional[str],
                 pert_data: Optional[PertData],
                 gene_ids: Optional[Union[List, np.ndarray]],
                 perts_to_plot: Optional[str],
                 ):
        """
        Parameters:
            model: nn.Module
                tabula model
            config: FinetuneConfig
                finetune configuration object
            save_path: str
                path to save the model and finetune analysis results
            pert_data: PertData
                perturbation data object
            gene_ids: List
                gene ids
            perts_to_plot: List
                perturbation conditions to plot for evaluation
        """
        super(FinetuneModel, self).__init__()
        self.model = model
        self.config = config

        self.save_path = save_path
        self.do_mgm = self.config.get_finetune_param('do_mgm')
        self.do_cmgm = self.config.get_finetune_param('do_cmgm')
        self.amp = self.config.get_finetune_param('amp')
        self.if_wandb = self.config.get_finetune_param('if_wandb')

        self.criterion_masked_mse = masked_mse_loss

        self.epoch_val_loss_list = []
        self.best_val_loss = float('inf')
        self.pert_data = pert_data
        self.gene_ids = gene_ids
        self.n_genes = len(gene_ids)
        self.perts_to_plot = perts_to_plot

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
        Calculate the mgm loss and log the loss
        """
        _, loss = self._perturbation_forward(batch)
        self.log('train/mgm_loss', loss['mgm_loss'], on_step=True, on_epoch=True, prog_bar=True)
        return loss['mgm_loss']

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
        Calculate the mgm loss and log the loss
        """
        _, loss = self._perturbation_forward(batch)
        self.epoch_val_loss_list.append(loss['mgm_loss'])
        loss = loss['mgm_loss']
        self.log('valid/mgm_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Lightening method to perform operations at the end of validation epoch
        1. Evaluate the model on test set and plot the perturbation conditions
        2. Log the validation loss and save the best model
        """
        val_loss = torch.stack(self.epoch_val_loss_list).mean()
        self.epoch_val_loss_list.clear()
        self.log('valid/total_loss', val_loss.item())
        if val_loss.item() < self.best_val_loss:
            try:
                self._eval_perturb(self.pert_data.dataloader["test_loader"])
            except Exception as e:
                logger.error(e)
                pass
            for perts_to_plot_item in self.perts_to_plot:
                self._plot_perturbation(query=perts_to_plot_item, pool_size=600)
        elif self.current_epoch == 0:
            self._eval_perturb(self.pert_data.dataloader["test_loader"])

        # save the best model
        if val_loss.item() < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            torch.save(self.model.state_dict(), f"{self.save_path}/best_model.pth")

    def _perturbation_forward(self, batch):
        """
        Forward pass for the model
        Args:
            batch: batch for every iteration
        Returns:
            output: model output
            loss: loss value
        """
        batch_size = len(batch.y)
        batch.to(self.device)
        x: torch.Tensor = batch.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, self.n_genes)
        pert_flags = x[:, 1].long().view(batch_size, self.n_genes)
        target_gene_values = batch.y  # (batch_size, n_genes)

        input_gene_ids = torch.arange(self.n_genes, device=self.device, dtype=torch.long)

        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]
        target_values = target_gene_values[:, input_gene_ids]
        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, self.gene_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

        with torch.cuda.amp.autocast(enabled=self.amp):
            output = self.model(genes=mapped_input_gene_ids.to(self.device),
                                values=input_values.to(self.device),
                                pert_flags=input_pert_flags.to(self.device),
                                head=None,
                                do_mgm=True,
                                )
            masked_positions = torch.ones_like(input_values, dtype=torch.bool, device=input_values.device)
            loss_mgm = self.criterion_masked_mse(output['mgm_pred'], target_values.to(self.device), masked_positions)
            output['input_values'] = input_values
        return output, {'mgm_loss': loss_mgm}

    def _perturbation_predict(self, pert_list, pool_size=300):
        """
        Predict the gene expression values for the given perturbations.
        Args:
            pert_list: list of perturbation conditions to be predicted
            pool_size: number of control samples to be used for prediction
        Returns:
            results_pred: predicted gene expression values after normalization
            results_pred_original: predicted gene expression values before normalization
        """
        logger.info(f"Start predicting perturbation...")
        adata = self.pert_data.adata
        ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
        if pool_size is None:
            pool_size = len(ctrl_adata.obs)
        gene_list = self.pert_data.gene_names.values.tolist()
        for pert in pert_list:
            for i in pert:
                if i not in gene_list:
                    raise ValueError("The gene is not in the perturbation graph. Please select from GEARS.gene_list!")
        self.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            results_pred = {}
            results_pred_original = {}
            for pert in pert_list:
                cell_graphs = create_cell_graph_dataset_for_prediction(pert, ctrl_adata, gene_list, device,
                                                                       num_samples=pool_size)
                loader = DataLoader(cell_graphs, batch_size=64, shuffle=False)
                preds = []

                for batch_data in tqdm(loader):
                    batch_data.to(device)
                    batch_size = len(batch_data.pert)
                    x: torch.Tensor = batch_data.x
                    ori_gene_values = x[:, 0].view(batch_size, self.n_genes)
                    pert_flags = x[:, 1].long().view(batch_size, self.n_genes)
                    input_gene_ids = torch.arange(self.n_genes, device=device, dtype=torch.long)

                    input_values = ori_gene_values[:, input_gene_ids]
                    input_pert_flags = pert_flags[:, input_gene_ids]
                    mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, self.gene_ids)
                    mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                    with torch.cuda.amp.autocast(enabled=self.amp):
                        output = self.model(genes=mapped_input_gene_ids.to(self.device),
                                            values=input_values.to(self.device),
                                            pert_flags=input_pert_flags.to(self.device),
                                            head=None,
                                            do_mgm=True,
                                            )
                        output_values = output['mgm_pred'].float().detach().cpu()
                        pred_gene_values = torch.zeros_like(ori_gene_values)
                        pred_gene_values[:, input_gene_ids] = output_values.to(self.device)
                    preds.append(pred_gene_values)

                preds = torch.cat(preds, dim=0)
                results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)
                results_pred_original["_".join(pert)] = preds.detach().cpu().numpy()
        return results_pred, results_pred_original

    def _plot_perturbation(self, query, pool_size: int = 600):
        """
        Plot the gene expression values for the given perturbation condition
        Args:
            query: perturbation condition to be plotted
            pool_size: number of control samples to be used for prediction
        """
        sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)
        adata = self.pert_data.adata
        gene2idx = self.pert_data.node_map
        cond2name = dict(adata.obs[["condition", "condition_name"]].values)
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
        de_idx = [gene2idx[gene_raw2id[i]]
                  for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]]
        genes = [gene_raw2id[i] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]]
        truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
        if query.split("+")[1] == "ctrl":
            pred, pred_original = self._perturbation_predict([[query.split("+")[0]]], pool_size=pool_size)
            pred = pred[query.split("+")[0]][de_idx]
            pred_original = pred_original[query.split("+")[0]][:, de_idx]
        else:
            pred, pred_original = self._perturbation_predict([query.split("+")], pool_size=pool_size)
            pred = pred["_".join(query.split("+"))][de_idx]
            pred_original = pred_original["_".join(query.split("+"))][:, de_idx]
        ctrl_means = adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values
        pred = pred - ctrl_means
        truth = truth - ctrl_means
        pred_original = pred_original - ctrl_means

        data, categories, gene_labels = [], [], []
        for i, gene in enumerate(genes):
            data.extend(truth.T[i])
            categories.extend(['Truth'] * len(truth.T[i]))
            gene_labels.extend([gene] * len(truth.T[i]))

            data.extend(pred_original.T[i])
            categories.extend(['Prediction'] * len(pred_original.T[i]))
            gene_labels.extend([gene] * len(pred_original.T[i]))
        df = pd.DataFrame({'Gene': gene_labels, 'Value': data, 'Type': categories})
        plt.figure(figsize=(16.5, 6.5))
        sns.violinplot(data=df, x='Gene', y='Value', hue='Type', dodge=True,
                       palette={'Truth': 'skyblue', 'Prediction': 'orange'}, cut=0, scale='width', split=False)
        plt.axhline(0, linestyle="dashed", color="green")
        fontsize = 20
        figure_fontsize = 18
        plt.title(query, fontsize=fontsize)
        plt.ylabel("Change in Gene \nExpression over Control", fontsize=fontsize)
        plt.xlabel("Gene", fontsize=fontsize)
        plt.xticks(rotation=90, fontsize=figure_fontsize)
        plt.yticks(fontsize=figure_fontsize)
        plt.legend(fontsize=figure_fontsize, title_fontsize=figure_fontsize, frameon=False, loc='lower left')
        sns.despine()
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/{query}_violin.png", format='png', bbox_inches='tight', dpi=300)
        if self.if_wandb:
            buf_grouped_violin = io.BytesIO()
            plt.savefig(buf_grouped_violin, format='png', bbox_inches='tight')
            buf_grouped_violin.seek(0)
            pil_img_grouped_violin = Image.open(buf_grouped_violin)
            wandb.log({f"test_{query}/perturbation_violin": [wandb.Image(pil_img_grouped_violin)]})
        plt.close()

        plt.figure(figsize=[16.5, 6.5])
        plt.title(query)
        plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))

        for i in range(pred.shape[0]):
            _ = plt.scatter(i + 1, pred[i], color="red")
        plt.axhline(0, linestyle="dashed", color="green")
        ax = plt.gca()
        ax.xaxis.set_ticklabels(genes, rotation=90)
        plt.ylabel("Change in Gene Expression over Control", labelpad=10)
        plt.tick_params(axis="x", which="major", pad=5)
        plt.tick_params(axis="y", which="major", pad=5)
        sns.despine()
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/{query}.png", format='png', bbox_inches='tight', dpi=300)
        if self.if_wandb:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            pil_img = Image.open(buf)
            wandb.log({f"test_{query}/perturbation": [wandb.Image(pil_img)]})
        plt.close()

    def _eval_perturb(self, loader: DataLoader):
        """
        Run model in inference mode using a given data loader and log the results
        Args:
            loader: data loader for evaluation
        """
        logger.info(f"Start evaluating perturbation...")
        pert_cat, pred, truth, pred_de, truth_de, results = [], [], [], [], [], {}
        for itr, batch in tqdm(enumerate(loader), total=len(loader)):
            batch.to(self.device)
            pert_cat.extend(batch.pert)
            with torch.no_grad():
                output, _ = self._perturbation_forward(batch)
                p = output['mgm_pred']
                t = batch.y
                pred.extend(p.cpu())
                truth.extend(t.cpu())
                for itr, de_idx in enumerate(batch.de_idx):
                    pred_de.append(p[itr, de_idx])
                    truth_de.append(t[itr, de_idx])

        results["pert_cat"] = np.array(pert_cat)
        pred = torch.stack(pred)
        truth = torch.stack(truth)
        results["pred"] = pred.detach().cpu().numpy().astype(float)
        results["truth"] = truth.detach().cpu().numpy().astype(float)
        pred_de = torch.stack(pred_de)
        truth_de = torch.stack(truth_de)
        results["pred_de"] = pred_de.detach().cpu().numpy().astype(float)
        results["truth_de"] = truth_de.detach().cpu().numpy().astype(float)

        test_metrics, test_pert_res = compute_metrics(results)
        if self.if_wandb:
            wandb.log({'test/metrics': test_metrics})
            wandb.log({'test_pert/metrics': test_pert_res})
        deeper_res = deeper_analysis(self.pert_data.adata, results)
        non_dropout_res = non_dropout_analysis(self.pert_data.adata, results)

        metrics = ["pearson_delta", "pearson_delta_de"]
        metrics_non_dropout = [
            "pearson_delta_top20_de_non_dropout",
            "pearson_top20_de_non_dropout",
        ]
        subgroup_analysis = {}
        for name in self.pert_data.subgroup["test_subgroup"].keys():
            subgroup_analysis[name] = {}
            for m in metrics:
                subgroup_analysis[name][m] = []
            for m in metrics_non_dropout:
                subgroup_analysis[name][m] = []

        for name, pert_list in self.pert_data.subgroup["test_subgroup"].items():
            for pert in pert_list:
                for m in metrics:
                    subgroup_analysis[name][m].append(deeper_res[pert][m])

                for m in metrics_non_dropout:
                    subgroup_analysis[name][m].append(non_dropout_res[pert][m])

        for name, result in subgroup_analysis.items():
            for m in result.keys():
                subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                if isinstance(subgroup_analysis[name][m], float) and not np.isnan(subgroup_analysis[name][m]):
                    logger.info("test_" + name + "_" + m + ": " + str(subgroup_analysis[name][m]))
                    if self.if_wandb:
                        wandb.log({"test/" + name + "_" + m: subgroup_analysis[name][m]})

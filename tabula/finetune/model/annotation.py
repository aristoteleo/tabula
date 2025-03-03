from typing import Optional
import json

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pytorch_lightning as pl
import wandb
import torch
import torch.multiprocessing
from tabula import logger
from tabula.finetune.utils import FinetuneConfig
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch import nn
from anndata import AnnData


class FinetuneModel(pl.LightningModule):
    """
    The class is used to finetune the tabula model for cell type annotation
    """
    def __init__(self,
                 model: nn.Module,
                 config: FinetuneConfig,
                 save_path: Optional[str],
                 id2celltype: dict,
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
            id2celltype: dict
                The dictionary mapping cell type id to cell type name
            test_loader: torch.utils.data.DataLoader
                The dataloader for test set
        """
        super(FinetuneModel, self).__init__()
        self.model = model
        self.config = config
        self.save_path = save_path
        self.id2type = id2celltype
        self.test_loader = test_loader
        self.val_loss_list = []
        self.best_val_loss = float('inf')

    def configure_optimizers(self):
        """
        Lightening method to configure the optimizer
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.get_finetune_param('learning_rate'),
            weight_decay=self.config.get_finetune_param('weight_decay'))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.get_finetune_param('max_epochs'))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def training_step(self, batch, batch_idx):
        """
        Lightening method to perform training step
        Calculate the loss and log the loss
        """
        batch = [b.to(self.device) for b in batch]
        loss, _ = self._annotation_forward(batch, 'train')
        return {'loss': loss}
    
    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch, batch_idx) -> None:
        """
        Lightening method to perform validation step
        Calculate the loss and record the loss
        """
        batch = [b.to(self.device) for b in batch]
        loss, _ = self._annotation_forward(batch, 'validation')
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
        if current_val_loss < self.best_val_loss:
            self._ann_predict()
            self.best_val_loss = current_val_loss
            torch.save(self.model.state_dict(),
                       f"{self.config.get_finetune_param('save_folder')}/best_model.pth")

    def _annotation_forward(self, batch, loop_stage):
        """
        Forward pass for cell type annotation

        Args:
            batch: data batch
            loop_stage: indicator of training or validation

        Returns:
            total_loss: total loss
            output: output of the model
        """
        genes, values, label, batch_info = batch
        output = self.model(genes=genes,
                            values=values,
                            batch_info=batch_info,
                            head='supervised')
        loss = torch.nn.functional.cross_entropy(output['supervised'], label.long())
        self.log(f'{loop_stage}/supervised_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss, output

    def _ann_predict(self):
        """
        Predict cell type labels for eval_anndata object
        """
        save_folder = self.config.get_finetune_param('save_folder')
        eval_predictions = []
        eval_labels = []
        eval_x_umap = []
        eval_results = {}
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                batch = [b.to(self.device) for b in batch]
                genes, values, label, batch_info, x_umap = batch
                eval_output = self.model(genes=genes,
                                         values=values,
                                         batch_info=batch_info,
                                         head='supervised')
                eval_pred = torch.argmax(eval_output['supervised'], dim=1)
                eval_predictions.append(eval_pred)
                eval_labels.append(label)
                eval_x_umap.append(x_umap)


        eval_predictions = torch.cat(eval_predictions)
        eval_labels = torch.cat(eval_labels)
        eval_x_umap = torch.cat(eval_x_umap)

        eval_results['accuracy'] = accuracy_score(eval_labels.cpu().numpy(), eval_predictions.cpu().numpy())
        eval_results['precision'] = precision_score(eval_labels.cpu().numpy(), eval_predictions.cpu().numpy(), average='macro')
        eval_results['recall'] = recall_score(eval_labels.cpu().numpy(), eval_predictions.cpu().numpy(), average='macro')
        eval_results['macro_f1'] = f1_score(eval_labels.cpu().numpy(), eval_predictions.cpu().numpy(), average='macro')
        eval_results['error_rate'] = torch.sum(eval_predictions.cpu() != eval_labels.cpu()).item() / len(eval_labels)

        # build a anndata object for the input of scanpy for UMAP visualization
        eval_anndata = AnnData(eval_x_umap.clone().detach().cpu().numpy())
        eval_anndata.obs['Annotated'] = [self.id2type[label] for label in eval_labels.cpu().numpy()]
        eval_anndata.obs['Predicted'] = [self.id2type[label] for label in eval_predictions.cpu().numpy()]
        eval_anndata.obsm['X_umap'] = eval_x_umap.cpu().numpy()

        if self.config.get_finetune_param('enable_wandb'):
            wandb.log({'test/metrics': eval_results})
        else:
            logger.info(eval_results)
        # save eval_data and results to local
        eval_anndata.write(f'{save_folder}/best_eval_data.h5ad')
        with open(f'{save_folder}/best_results.json', 'w') as f:
            json.dump(eval_results, f)
        # save UMAP to wandb
        celltypes = list(self.id2type.values())
        palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"] + plt.rcParams["axes.prop_cycle"].by_key()[
            "color"] + plt.rcParams["axes.prop_cycle"].by_key()["color"]
        palette_ = {c: palette_[i] for i, c in enumerate(celltypes)}
        with plt.rc_context({"figure.figsize": (6, 4), "figure.dpi": (600)}):
            sc.pl.umap(
                eval_anndata,
                color=["Predicted"],
                palette=palette_,
                show=False,
                frameon=False,
            )
        plt.savefig(f'{save_folder}/umap.png', bbox_inches='tight')
        plt.close()
        eval_labels = eval_labels.cpu()
        eval_predictions = eval_predictions.cpu()
        self._plotCM2wandb(list(self.id2type.values()), eval_labels, eval_predictions, save_folder)

    def _plotCM2wandb(self, celltypes, labels, predictions, save_folder):
        """
        Plot confusion matrix to wandb
        """
        unique_labels = np.sort(np.unique(labels))
        label_names = [self.id2type.get(label) for label in unique_labels]
        cm = confusion_matrix(labels, predictions, labels=unique_labels)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 12))
        plt.imshow(cm, interpolation='nearest', cmap="Blues")
        tick_marks = np.arange(len(label_names))
        plt.xticks(tick_marks, label_names, rotation=45, ha='right')
        plt.yticks(tick_marks, label_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], '.1f'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.colorbar(shrink=0.715)
        plt.tight_layout()
        plt.savefig(f'{save_folder}/confusion.png', bbox_inches='tight', dpi=600)
        plt.close()

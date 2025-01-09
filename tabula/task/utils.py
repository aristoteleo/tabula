from typing import Dict, Optional

import numpy as np
import os
import scib
from anndata import AnnData
from tabula import logger
import yaml


class FinetuneConfig:
    """
    Parameter loader and pre-setting for finetune task
    """
    def __init__(self, seed, config_path):
        self.seed = seed
        self.config_path = config_path

        with open(self.config_path, 'r') as f:
            self.framework = yaml.load(f, Loader=yaml.FullLoader)
        self.finetune_config = self.framework.get('Finetune', {})
        self.model_config = self.framework.get('Model', {})

    def get_finetune_param(self, key, default_value=None):
        """
        Get the value of a key in the finetune configuration. If the key is not found, return the default value.
        """
        if key not in self.finetune_config:
            if default_value is None:
                raise Exception(f"Key '{key}' not found in finetune configuration.")
            else:
                return default_value
        return self.finetune_config.get(key)

    def get_model_param(self, key, default_value=None):
        """
        Get the value of a key in the model configuration. If the key is not found, return the default value.
        """
        if key not in self.model_config:
            if default_value is None:
                raise Exception(f"Key '{key}' not found in model configuration.")
            else:
                return default_value
        return self.model_config.get(key)

    def set_finetune_param(self, key, value):
        self.finetune_config[key] = value
        if key == 'save_folder':
            if not os.path.exists(value):
                os.makedirs(value)

    def set_model_param(self, key, value):
        self.model_config[key] = value


def eval_scib_metrics(
        adata: AnnData,
        batch_key: str = "str_batch",
        label_key: str = "celltype",
        notes: Optional[str] = None,
) -> Dict:
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed="tabula_embed",
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )

    result_dict = results[0].to_dict()
    # logger.info(
    #     "Biological Conservation Metrics: \n"
    #     f"ASW (cell-type): {result_dict['ASW_label']:.4f}, graph cLISI: {result_dict['cLISI']:.4f}, "
    #     f"isolated label silhouette: {result_dict['isolated_label_silhouette']:.4f}, \n"
    #     "Batch Effect Removal Metrics: \n"
    #     f"PCR_batch: {result_dict['PCR_batch']:.4f}, ASW (batch): {result_dict['ASW_label/batch']:.4f}, "
    #     f"graph connectivity: {result_dict['graph_conn']:.4f}, graph iLISI: {result_dict['iLISI']:.4f}"
    # )

    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}

    return result_dict

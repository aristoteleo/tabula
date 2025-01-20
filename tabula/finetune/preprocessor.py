import json
from typing import Dict, Optional, Union

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from pytorch_lightning.callbacks import EarlyStopping
from scanpy.get import _get_obs_rep, _set_obs_rep
from scipy.sparse import issparse
from tabula import logger
from tabula.finetune.utils import FinetuneConfig
from tabula.model.transfomer.transformer import TabulaTransformer


def get_pretrained_model(finetune_config: FinetuneConfig = None,
                         model_path: str = None,
                         device: str = 'cpu'):
    """
    Get the pretrained model for downstrean tasks.
    Args:
        finetune_config: FinetuneConfig
        model_path: str
        device: str
        num_batches: int

    Returns:

    """
    backbone = finetune_config.get_finetune_param('pretrained_backbone').lower()
    additive_attention, flash_attention = False, False
    if backbone == 'fastformer':
        additive_attention = True
        logger.info('Loading FastFormer Tabula from path: {}'.format(model_path))
    elif backbone == 'flashattention':
        flash_attention = True
        logger.info('Loading FlashAttention Tabula from path: {}'.format(model_path))
    else:
        raise ValueError(f"Backbone {backbone} not supported.")

    model = TabulaTransformer(
        in_feature=finetune_config.get_model_param('in_feature'),
        embedding_in_feature=finetune_config.get_model_param('embedding_in_feature'),
        contrastive_out_feature=finetune_config.get_model_param('contrastive_out_feature'),
        supervised_out_feature=finetune_config.get_model_param('supervised_out_feature'),
        d_token=finetune_config.get_model_param('d_token'),
        n_blocks=finetune_config.get_model_param('n_blocks'),
        residual_dropout=finetune_config.get_model_param('residual_dropout'),
        additive_attention=additive_attention,
        flash_attention=flash_attention,
        attention_n_heads=finetune_config.get_model_param('attention_n_heads'),
        attention_dropout=finetune_config.get_model_param('attention_dropout'),
        ffn_d_hidden=finetune_config.get_model_param('ffn_d_hidden'),
        ffn_dropout=finetune_config.get_model_param('ffn_dropout'),
        cls=finetune_config.get_model_param('cls'),
        pre_normalization=finetune_config.get_model_param('pre_normalization'),
        global_token=finetune_config.get_model_param('global_token'),
        pretrain_objective=finetune_config.get_finetune_param('objective'),
        enable_batch=finetune_config.get_finetune_param('enable_batch'),
        n_batch=finetune_config.get_finetune_param('n_batch') if finetune_config.get_finetune_param('enable_batch') else 1,
        explicit_zero_prob=finetune_config.get_finetune_param('explicit_zero_prob'),
        do_mgm=finetune_config.get_finetune_param('do_mgm'),
        do_cmgm=finetune_config.get_finetune_param('do_cmgm'),
        cmgm_decoder_style=finetune_config.get_finetune_param('cmgm_decoder_style'),
        do_dab=finetune_config.get_finetune_param('do_dab'),
        embed_style=finetune_config.get_finetune_param('embed_style'),
    ).to(device)

    if model_path is None:
        logger.info('Initializing model from scratch.')
        return model

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        logger.info('Loading full pretrained weight from path: {}'.format(model_path))
    except Exception as e:
        logger.error(f"Error loading model from path: {model_path}, switch to load specific weights.")
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=torch.device(device))
        """
        add transformer to the front of every key in the pretrained_dict
        """
        pretrained_dict_copy = pretrained_dict.copy()
        for k, v in pretrained_dict_copy.items():
            if f'transformer.{k}' in list(model_dict.keys()):
                pretrained_dict[f'transformer.{k}'] = pretrained_dict.pop(k)

        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        not_loaded = [k for k in model_dict if k not in pretrained_dict]
        if len(not_loaded) > 0:
            logger.info(f"Params not loaded: {not_loaded}")

        logger.info('Randomly initializing the rest of the model, their shape is:')
        for k, v in model_dict.items():
            if k in not_loaded:
                logger.info(f"{k}: {v.shape}")

        logger.info(f'Print shape from the pretrained model that are not loaded:')
        for k, v in pretrained_dict_copy.items():
            if k not in pretrained_dict:
                logger.info(f"{k}: {v.shape}")

    return model


def check_vocab(adata, vocab_file):
    """
    To check if the gene is existed in the vocab

    Args:
        adata: AnnData
        vocab_file: vocab file path

    Returns:
        adata: AnnData
    """
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)

    # Get the genes in the vocab
    vocab_genes = set(vocab.keys())

    # Filter out the genes that are not in the vocab
    valid_genes_mask = adata.var["gene_name"].isin(vocab_genes)
    # Get the labels of genes that were removed
    removed_gene_labels = adata.var[~valid_genes_mask]["gene_name"].tolist()

    adata = adata[:, valid_genes_mask]
    return adata, removed_gene_labels


class Preprocessor:
    """
    The code is modified from scGPT: https://github.com/bowang-lab/scGPT/blob/main/scgpt/preprocess.py

    Prepare data into training, valid and test split. Normalize raw expression
    values, binning or using other transform into the preset model input format.
    """
    def __init__(
            self,
            use_key: Optional[str] = None,
            filter_gene_by_counts: Union[int, bool] = False,
            filter_cell_by_counts: Union[int, bool] = False,
            normalize_total: Union[float, bool] = 1e4,
            result_normed_key: Optional[str] = "X_normed",
            log1p: bool = False,
            result_log1p_key: str = "X_log1p",
            subset_hvg: Union[int, bool] = False,
            hvg_use_key: Optional[str] = None,
            hvg_flavor: str = "seurat_v3",
            binning: Optional[int] = None,
            result_binned_key: str = "X_binned",
    ):
        """
        Set up the preprocessor, use the args to config the workflow steps.

        Args:

        use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for preprocessing.
        filter_gene_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter genes by counts, if :class:`int`, filter genes with counts
        filter_cell_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter cells by counts, if :class:`int`, filter cells with counts
        normalize_total (:class:`float` or :class:`bool`, default: ``1e4``):
            Whether to normalize the total counts of each cell to a specific value.
        result_normed_key (:class:`str`, default: ``"X_normed"``):
            The key of :class:`~anndata.AnnData` to store the normalized data. If
            :class:`None`, will use normed data to replce the :attr:`use_key`.
        log1p (:class:`bool`, default: ``True``):
            Whether to apply log1p transform to the normalized data.
        result_log1p_key (:class:`str`, default: ``"X_log1p"``):
            The key of :class:`~anndata.AnnData` to store the log1p transformed data.
        subset_hvg (:class:`int` or :class:`bool`, default: ``False``):
            Whether to subset highly variable genes.
        hvg_use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for calculating highly variable
            genes. If :class:`None`, will use :attr:`adata.X`.
        hvg_flavor (:class:`str`, default: ``"seurat_v3"``):
            The flavor of highly variable genes selection. See
            :func:`scanpy.pp.highly_variable_genes` for more details.
        binning (:class:`int`, optional):
            Whether to bin the data into discrete values of number of bins provided.
        result_binned_key (:class:`str`, default: ``"X_binned"``):
            The key of :class:`~anndata.AnnData` to store the binned data.
        """
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor
        self.binning = binning
        self.result_binned_key = result_binned_key

    def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> Dict:
        """
        format controls the different input value wrapping, including categorical
        binned style, fixed-sum normalized counts, log1p fixed-sum normalized counts, etc.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        batch_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        key_to_process = self.use_key
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None  # the following scanpy apis use arg None to use X
        is_logged = self.check_logged(adata, obs_key=key_to_process)

        row_name = adata.var_names.tolist()
        row_name_upper = [i.upper() for i in row_name]
        adata.var.index = row_name_upper

        # step 1: filter genes
        if self.filter_gene_by_counts:
            logger.info("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_by_counts
                if isinstance(self.filter_gene_by_counts, int)
                else None,
            )

        # step 2: filter cells
        if isinstance(self.filter_cell_by_counts, int):
            logger.info("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_by_counts
                if isinstance(self.filter_cell_by_counts, int)
                else None,
            )

        # step 3: normalize total
        if self.normalize_total:
            logger.info("Normalizing total counts ...")
            normed_ = sc.pp.normalize_total(
                adata,
                target_sum=self.normalize_total
                if isinstance(self.normalize_total, float)
                else None,
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.result_normed_key or key_to_process
            _set_obs_rep(adata, normed_, layer=key_to_process)

        # step 4: log1p
        if self.log1p:
            logger.info("Log1p transforming ...")
            if is_logged:
                logger.warning(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log1p_key:
                _set_obs_rep(
                    adata,
                    _get_obs_rep(adata, layer=key_to_process),
                    layer=self.result_log1p_key,
                )
                key_to_process = self.result_log1p_key
            sc.pp.log1p(adata, layer=key_to_process)

        # step 5: subset hvg
        if self.subset_hvg:
            logger.info("Subsetting highly variable genes ...")
            if batch_key is None:
                logger.warning(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            sc.pp.highly_variable_genes(
                adata,
                layer=self.hvg_use_key,
                n_top_genes=self.subset_hvg
                if isinstance(self.subset_hvg, int)
                else None,
                batch_key=batch_key,
                flavor=self.hvg_flavor,
                subset=True,
            )

        # step 6: binning
        if self.binning:
            logger.info("Binning data ...")
            if not isinstance(self.binning, int):
                raise ValueError(
                    "Binning arg must be an integer, but got {}.".format(self.binning)
                )
            n_bins = self.binning  # NOTE: the first bin is always a spectial for zero
            binned_rows = []
            bin_edges = []
            layer_data = _get_obs_rep(adata, layer=key_to_process)
            layer_data = layer_data.A if issparse(layer_data) else layer_data
            for row in layer_data:
                non_zero_ids = row.nonzero()
                non_zero_row = row[non_zero_ids]
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
                # bins = np.sort(np.unique(bins))
                # NOTE: comment this line for now, since this will make the each category
                # has different relative meaning across datasets
                non_zero_digits = self._digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int64)
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
            adata.layers[self.result_binned_key] = np.stack(binned_rows)
            adata.obsm["bin_edges"] = np.stack(bin_edges)

    def _digitize(self, x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """
        Digitize the data into bins. This method spreads data uniformly when bins
        have same values.

        Args:

        x (:class:`np.ndarray`):
            The data to digitize.
        bins (:class:`np.ndarray`):
            The bins to use for digitization, in increasing order.

        Returns:

        :class:`np.ndarray`:
            The digitized data.
        """
        assert x.ndim == 1 and bins.ndim == 1

        left_digits = np.digitize(x, bins)
        right_difits = np.digitize(x, bins, right=True)

        rands = np.random.rand(len(x))  # uniform random numbers

        digits = rands * (right_difits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)
        return digits

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        """
        Check if the data is already log1p transformed.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True


def binning(
        row: Union[np.ndarray, torch.Tensor], n_bins: int
) -> Union[np.ndarray, torch.Tensor]:
    """Binning the row into n_bins."""
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row
    # TODO: use torch.quantile and torch.bucketize

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = np.digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = np.digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)


def random_mask_value(
        values: Union[torch.Tensor, np.ndarray],
        mask_ratio: float = 0.4,
        mask_value: int = -1
) -> torch.Tensor:
    """
    Randomly mask a batch of data, including the original padding values.

    Args:
        values (array-like):
            A batch of tokenized data, with shape (batch_size, n_features).
        mask_ratio (float): The ratio of genes to mask, default to 0.15.
        mask_value (int): The value to mask with, default to -1.

    Returns:
        torch.Tensor: A tensor of masked data.
    """
    if isinstance(values, torch.Tensor):
        values = values.clone().detach().cpu().numpy()
    else:
        values = values.copy()

    for i in range(len(values)):
        row = values[i]
        n_mask = int(len(row) * mask_ratio)
        mask_idx = np.random.choice(len(row), n_mask, replace=False)
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()


def get_ft_training_args(finetune_config: FinetuneConfig) -> Dict:
    """
    Get the training arguments for fine-tuning.

    Args:
        finetune_config (FinetuneConfig): The fine-tune configuration.

    Returns:
        Dict: The training arguments.
    """
    early_stopping_callback_func = None
    finetune_method = finetune_config.get_finetune_param('method')
    if finetune_method == 'light':
        max_epochs = finetune_config.get_finetune_param('light_epochs')
        logger.info(f"Finetune method: {finetune_method}. Max epochs: {max_epochs}")
    elif finetune_method == 'heavy':
        max_epochs = finetune_config.get_finetune_param('max_epochs')
        early_stopping_callback_func = EarlyStopping('valid/loss',
                                                     patience=finetune_config.get_finetune_param('patience'))
        logger.info(f"Finetune method: {finetune_method}. Max epochs: {max_epochs}. "
                    f"Patience: {early_stopping_callback_func.patience}")
    else:
        raise ValueError(f"Unsupported finetune method {finetune_method}")

    training_args = {
        'max_epochs': max_epochs,
        'callbacks': [early_stopping_callback_func] if finetune_method == 'heavy' else None,
        'default_root_dir': finetune_config.get_finetune_param('save_folder'),
        # 'gradient_clip_val': finetune_config.get_finetune_param('gradient_clip_val'),
        'precision': 16
    }

    return training_args

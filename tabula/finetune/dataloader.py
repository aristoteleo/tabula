import json
import random

random.seed(0)

from typing import Optional, Union

import numpy as np
import torch
from tabula import logger
from tabula.finetune.tokenizer import GeneVocab
from torch.utils.data import Dataset


def binning(row, return_index, n_bins=51):
    # Handle sparse matrix
    if hasattr(row, 'A'):
        row = row.A

    assert isinstance(row, np.ndarray), "Expected 'row' to be a numpy ndarray or a sparse matrix"

    # Check dimensionality
    if len(row.shape) == 1:
        non_zero_ids = row.nonzero()[0]
    elif len(row.shape) == 2:
        non_zero_ids = row.nonzero()
    else:
        raise ValueError(f"Expected 'row' to be 1D or 2D, but got shape {row.shape}")

    non_zero_row = row[non_zero_ids]
    bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
    non_zero_digits = np.digitize(non_zero_row, bins)
    row[non_zero_ids] = non_zero_digits

    # Return based on the dimensionality
    if len(row.shape) == 1:
        return torch.tensor(row[return_index]).float()
    elif len(row.shape) == 2:
        return torch.tensor(row[0, return_index]).float()


class CellAnnotationDataset(Dataset):
    def __init__(self,
                 expression_table,
                 masked_expression_table,
                 gene_ids,
                 labels,
                 batch_strings,
                 x_umap,
                 in_feature,
                 vocab_file):
        self.expression_table = expression_table
        self.masked_expression_table = masked_expression_table
        self.gene_ids = gene_ids
        self.labels = labels
        self.batch_strings = batch_strings
        self.if_multi_batch = len(set(batch_strings)) > 1
        self.x_umap = x_umap
        self.in_feature = in_feature
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        binned_x_values = self.expression_table[index]
        non_zero_indices = [i for i, value in enumerate(binned_x_values) if value != 0]
        # If not enough non-zero genes, supplement with zeros
        if len(non_zero_indices) < self.in_feature:
            zero_indices = [i for i in range(len(binned_x_values)) if i not in non_zero_indices]
            random.shuffle(zero_indices)
            non_zero_indices += zero_indices[:self.in_feature - len(non_zero_indices)]

        random.shuffle(non_zero_indices)
        genes = torch.tensor([self.vocab[self.gene_ids[i]] for i in non_zero_indices[:self.in_feature]],
                             dtype=torch.int)
        label = torch.tensor(self.labels[index], dtype=torch.int)
        batch = torch.tensor(int(self.batch_strings[index]), dtype=torch.int)
        limit_x_values = torch.tensor(binned_x_values[non_zero_indices[:self.in_feature]], dtype=torch.float)
        if self.x_umap is not None:
            x_umap = torch.tensor(self.x_umap[index], dtype=torch.float)
            return genes, limit_x_values, label, batch, x_umap

        return genes, limit_x_values, label, batch


class MultiOmicsDataset(Dataset):
    def __init__(self,
                 expression_table: np.ndarray,
                 gene_ids: list,
                 labels: list,
                 batch_id: list,
                 vocab_file: Optional[Union[str, GeneVocab]] = None,
                 in_feature: int = 1200,
                 ):
        self.multimodal_expression_table = expression_table
        self.gene_ids = gene_ids
        self.labels = labels
        self.batch_id = batch_id
        self.in_feature = in_feature
        if isinstance(vocab_file, str):
            with open(vocab_file, 'r') as f:
                self.vocab = json.load(f)
        else:
            self.vocab = vocab_file

    def upper_case_iterator(self, vocab):
        new_vocab = {}
        for k, v in vocab.items():
            if '<' in k:
                continue
            k = k.upper()
            new_vocab[k] = v
        return new_vocab

    def expand_vocab(self, modality_string_list):
        """
        Expand the vocab of scRNA-seq to include other omics

        Args:
            modality_string_list: list of modality strings
        """
        for multimodal_id in modality_string_list:
            self.vocab[multimodal_id] = len(self.vocab)
        # log vocab size
        logger.info(f'Vocab size: {len(self.vocab)}')
        self.vocab = self.upper_case_iterator(self.vocab)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        mixed_values = self.multimodal_expression_table[index]
        non_zero_indices = [i for i, value in enumerate(mixed_values) if value != 0]
        # If not enough non-zero genes, supplement with zeros
        if len(non_zero_indices) < self.in_feature:
            zero_indices = [i for i in range(len(mixed_values)) if i not in non_zero_indices]
            random.shuffle(zero_indices)
            non_zero_indices += zero_indices[:self.in_feature - len(non_zero_indices)]

        random.shuffle(non_zero_indices)
        genes = torch.tensor([self.vocab[self.gene_ids[i]] for i in non_zero_indices[:self.in_feature]],
                             dtype=torch.int)
        values = torch.tensor(mixed_values[non_zero_indices[:self.in_feature]], dtype=torch.float)
        label = torch.tensor(self.labels[index], dtype=torch.int)
        batch = torch.tensor(int(self.batch_id[index]), dtype=torch.int)

        return genes, values, label, batch


class GenePerturbationDataset(Dataset):
    def __init__(self,
                 gene_ids,
                 tokenized_input,
                 tokenized_target,
                 mask_expression,
                 in_feature,
                 ):
        self.gene_ids = gene_ids
        self.tokenized_input = tokenized_input
        self.tokenized_target = tokenized_target
        self.mask_expression = mask_expression
        self.in_feature = in_feature

    def __len__(self):
        return len(self.tokenized_input['values'])

    def __getitem__(self, index):
        """
        """
        genes = torch.tensor(self.tokenized_input['genes'][index], dtype=torch.int)
        perturbed_values = torch.tensor(self.mask_expression[index], dtype=torch.float)
        target_values = torch.tensor(self.tokenized_target['values'][index], dtype=torch.float)

        shuffled_indices = [i for i in range(len(genes))]
        random.shuffle(shuffled_indices)
        genes = genes[shuffled_indices[:self.in_feature]]
        perturbed_values = perturbed_values[shuffled_indices[:self.in_feature]]
        target_values = target_values[shuffled_indices[:self.in_feature]]
        return genes, perturbed_values, target_values


class ImputationDataset(Dataset):
    def __init__(self,
                 gene_ids: list,
                 original_values: np.ndarray,
                 masked_values: np.ndarray,
                 masked_positions: np.ndarray,
                 vocab_file: Optional[Union[str, GeneVocab]] = None,
                 ):
        self.gene_ids = gene_ids
        self.original_values = original_values
        self.masked_values = masked_values
        self.masked_positions = masked_positions

        if isinstance(vocab_file, str):
            with open(vocab_file, 'r') as f:
                self.vocab = json.load(f)
        else:
            self.vocab = vocab_file

    def __len__(self):
        return len(self.original_values)

    def __getitem__(self, index):

        genes = torch.tensor([self.vocab[self.gene_ids[i]] for i in range(len(self.gene_ids))], dtype=torch.int)
        original_values = torch.tensor(self.original_values[index], dtype=torch.float)
        masked_values = torch.tensor(self.masked_values[index], dtype=torch.float)
        masked_positions = torch.tensor(self.masked_positions[index], dtype=torch.bool)

        return genes, original_values, masked_values, masked_positions


class GrnDataset(Dataset):
    def __init__(self,
                 expression_table
                 ):
        self.expression_table = expression_table

    def __len__(self):
        return self.expression_table.shape[0]

    def __getitem__(self, idx):
        this_cell_expression = self.expression_table[idx].copy()
        return this_cell_expression

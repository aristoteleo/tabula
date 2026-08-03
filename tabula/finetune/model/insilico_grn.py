import torch
import tqdm
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from .plot import one_on_one_plot
from tabula import logger


N_BINS = 51


def binning_1d(row, return_index, n_bins=N_BINS):
    assert isinstance(row, np.ndarray), "Expected 'row' to be a numpy ndarray"
    assert row.ndim == 1, "Expected 'row' to be 1D"

    row[row < 0] = 0

    non_zero_ids = row.nonzero()[0]
    non_zero_values = row[non_zero_ids]

    bins = np.quantile(non_zero_values, np.linspace(0, 1, n_bins))
    non_zero_values_binned = np.digitize(non_zero_values, bins, right=True)

    row[non_zero_ids] = non_zero_values_binned

    return torch.tensor(row[return_index], dtype=torch.float), row


def binning_2d(matrix, return_index, n_bins=10):
    assert isinstance(matrix, np.ndarray), "Expected 'matrix' to be a numpy ndarray"
    assert matrix.ndim == 2, "Expected 'matrix' to be 2D"

    processed_rows = []

    for row in matrix:
        row[row < 0] = 0
        
        non_zero_ids = row.nonzero()[0]
        non_zero_values = row[non_zero_ids]

        if len(non_zero_values) > 0:
            bins = np.quantile(non_zero_values, np.linspace(0, 1, n_bins))
            non_zero_values_binned = np.digitize(non_zero_values, bins, right=True)
            row[non_zero_ids] = non_zero_values_binned
        
        processed_rows.append(row)
    
    processed_matrix = np.vstack(processed_rows)

    return torch.tensor(processed_matrix.T[return_index], dtype=torch.float), processed_matrix


def pairwise_inference(pair_list,
                       evolve_epoch,
                       model,
                       vocab,
                       gene_ids,
                       Grn_dataloader,
                       device,
                       save_path=None):
    for pair in pair_list:
        insilico_gene, mask_gene, regulatory_relation = pair
        this_save_path = os.path.join(save_path, f'{insilico_gene}_{regulatory_relation}_{mask_gene}')
        if not os.path.exists(this_save_path):
            os.makedirs(this_save_path)
        one_on_one_single_pair(insilico_gene=insilico_gene.upper(),
                               mask_gene=mask_gene.upper(),
                               evolve_epoch=evolve_epoch,
                               model=model,
                               vocab=vocab,
                               gene_ids=gene_ids,
                               Grn_dataloader=Grn_dataloader,
                               regulatory_relation=regulatory_relation,
                               device=device,
                               save_path=this_save_path)
    

def one_on_one_single_pair(insilico_gene,
                           mask_gene,
                           evolve_epoch,
                           model,
                           vocab,
                           gene_ids,
                           Grn_dataloader,
                           regulatory_relation,
                           device,
                           save_path):
    """
    Zero-shot gene regulatory network inference using a DataLoader for batch processing.
    :param insilico_gene: str, the gene to be manually activated or deactivated
    :param mask_gene: str, the gene to be masked and reconstructed
    :param activate: bool, indicator of activation or deactivation relationship between insilico_gene and mask_gene
                            activate=True: insilico_gene activates mask_gene
                            activate=False: insilico_gene deactivates mask_gene
    :param evolve_epoch: int, the number of epochs for in silico evolution
    :param model: sctab model, the pre-trained model for GRN inference
    :param Grn_dataloader: DataLoader, the DataLoader to iterate over cell expressions
    :param device: str, the device for computation

    :function: Run in silico evolution for cells in the DataLoader with the insilico_gene activated or deactivated,
    and observe the change of mask_gene in a continuous epoch manner. Save the result to a CSV file.
    """
    insilico_gene_id = vocab[insilico_gene]
    mask_gene_id = vocab[mask_gene]
    mask_gene_idx = gene_ids.index(mask_gene_id)
    insilico_gene_idx = gene_ids.index(insilico_gene_id)

    all_results = []  # Store all results across batches

    for _, batch in enumerate(tqdm.tqdm(Grn_dataloader)):
        batch = batch.to(device)
        original_expressions = batch.clone().detach()
        if regulatory_relation == 'deactivate':
            original_expressions = original_expressions[original_expressions[:, mask_gene_idx] > 0]
        elif regulatory_relation == 'activate':
            original_expressions = original_expressions[original_expressions[:, mask_gene_idx] < 50]

        # Prepare a 3D array to store results: [batch_size, evolve_epoch + 1, 2]
        batch_size = original_expressions.shape[0]
        results_array = np.zeros((batch_size, evolve_epoch + 1, 2))  # Columns: epoch, masked_gene_bin_value

        # Initial bin values (epoch -1)
        results_array[:, 0, 0] = -1  # Initial epoch is -1
        results_array[:, 0, 1] = original_expressions[:, mask_gene_idx].cpu().numpy()

        for epoch in range(evolve_epoch):
            original_expressions[:, insilico_gene_idx] = torch.minimum(
                original_expressions[:, insilico_gene_idx] + 1, torch.tensor(50).to(device)
            )
            with torch.no_grad():
                gene_ids_tensor = torch.tensor(gene_ids).int().to(device)
                reconstructed_expressions = model(genes=gene_ids_tensor,
                                                  values=original_expressions,
                                                  head='reconstruction')['reconstruction']

            reconstructed_expressions = reconstructed_expressions.cpu().numpy()
            this_bin_expression, binned_expressions = binning_2d(reconstructed_expressions, 
                                                                 return_index=[insilico_gene_idx, mask_gene_idx],
                                                                 n_bins=51)

            results_array[:, epoch + 1, 0] = epoch  # Current epoch
            results_array[:, epoch + 1, 1] = this_bin_expression[1, :]  # Masked gene values
            original_expressions = torch.tensor(binned_expressions).float().to(device)

        all_results.append(results_array)

    all_results = np.concatenate(all_results, axis=0)

    num_increase = np.sum(all_results[:, -1, 1] > all_results[:, 0, 1])
    num_decrease = np.sum(all_results[:, -1, 1] < all_results[:, 0, 1])
    num_nochange = np.sum(all_results[:, -1, 1] == all_results[:, 0, 1])
    avg_firt_epoch = np.mean(all_results[:, 0, 1])
    avg_last_epoch = np.mean(all_results[:, -1, 1])

    logger.info(f"Finished evolving {all_results.shape[0]} cells for exploring the relationship between {insilico_gene} and {mask_gene}.")
    logger.info(f"Result files are saved at {save_path}.")
    logger.info(f'Among {all_results.shape[0]} cells, {num_increase} masked genes increased, {num_decrease} decreased, and {num_nochange} did not change.')
    
    with open(f'{save_path}/{insilico_gene}_regulate_{mask_gene}_log.txt', 'w') as f:
        f.write(f'Exploring the relationship between {insilico_gene} and {mask_gene}.\n')
        f.write(f'{insilico_gene} regulates {mask_gene}.\n')
        f.write(f'Total evolved cells: {all_results.shape[0]}\n')
        f.write(f'Number of masked genes that increased: {num_increase}\n')
        f.write(f'Number of masked genes that decreased: {num_decrease}\n')
        f.write(f'Number of masked genes with no change: {num_nochange}\n')
        f.write(f'Average regulated gene expression at the first epoch: {avg_firt_epoch}\n')
        f.write(f'Average regulated gene expression at the last epoch: {avg_last_epoch}\n')
        if avg_firt_epoch > avg_last_epoch and regulatory_relation == 'deactivate':
            f.write(f'Activation relationship is correct, that is {insilico_gene} deactivate {mask_gene}.\n')
            logger.info(f'Activation relationship is correct, that is {insilico_gene} deactivate {mask_gene}.')
        elif avg_firt_epoch < avg_last_epoch and regulatory_relation == 'activate':
            f.write(f'Activation relationship is correct, that is {insilico_gene} activate {mask_gene}.\n')
            logger.info(f'Activation relationship is correct, that is {insilico_gene} activate {mask_gene}.')
        else:
            f.write(f'Activation relationship is incorrect.\n')
            logger.info(f'Activation relationship is incorrect.')

    one_on_one_plot(results_array=all_results[:, :, 1], masked_gene=mask_gene, in_silico_gene=insilico_gene, save_path=save_path, 
                    change_proportion={'increase': num_increase, 'decrease': num_decrease, 'nochange': num_nochange})


def combinatory_inference():
    raise NotImplementedError
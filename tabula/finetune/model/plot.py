import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def one_on_one_plot(results_array, masked_gene, in_silico_gene, save_path, change_proportion):
    """
    Plot gene expression changes over in silico evolution epochs.
    :param results_array: 2D array containing gene expression results [num_cells, evolve_epoch + 1]
    :param draw_epoch: int, number of epochs to draw
    :param masked_gene: str, the gene being monitored for changes
    :param in_silico_gene: str, the gene being activated or deactivated
    :param save_path: str, path to save the plot

    :return: matplotlib figure object
    """
    num_cells, num_epochs = results_array.shape

    joint_data = {
        'epoch': [],
        'gene_expression': [],
        'event': []
    }

    for epoch in range(num_epochs):
        joint_data['epoch'].extend([epoch] * num_cells)
        joint_data['gene_expression'].extend(results_array[:, epoch])
        joint_data['event'].extend([f'{in_silico_gene} -> {masked_gene}'] * num_cells)

    sns.set_theme(style="darkgrid")

    df = pd.DataFrame(joint_data)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(8, 6))
    sns.lineplot(x='epoch', y='gene_expression', hue='event', data=df, estimator='mean', ci='sd')
    plt.xlabel('In-silico Epoch')
    plt.ylabel('Gene Expression (averaged by cell)')
    plt.title(f'Gene Expression Changes Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{in_silico_gene}_regulates_{masked_gene}_line.png'))
    plt.close()

    # Plot violinplot for gene expression distribution at the first and last epoch
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='epoch', y='gene_expression', hue='event', data=df)
    plt.xlabel('In-silico Epoch')
    plt.ylabel('Gene Expression (by cell)')
    plt.title(f'Gene Expression Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{in_silico_gene}_regulates_{masked_gene}_violin.png'))
    plt.close()

    # Assuming you have calculated the proportions of each change type
    proportions = {
        'Activation': change_proportion['increase'] / sum(change_proportion.values()),
        'Repression': change_proportion['decrease'] / sum(change_proportion.values()),
        'Neutrality': change_proportion['nochange'] / sum(change_proportion.values())
    }

    plt.figure(figsize=(4, 6))
    plt.bar(
        [f'{in_silico_gene} regulates {masked_gene}'], 
        [1.0], 
        color='white', 
        edgecolor='black'
    )  
    bottoms = 0
    for change_type, prop in proportions.items():
        color = {
            'Activation': 'lightgreen',
            'Repression': 'plum',
            'Neutrality': 'peachpuff'
        }[change_type]

        plt.bar(
            [f'{in_silico_gene} regulates {masked_gene}'],
            [prop],
            bottom=bottoms,
            label=change_type,
            color=color
        )
        bottoms += prop

    plt.ylabel('Proportion')
    plt.title(f'Proportion of {in_silico_gene} regulating {masked_gene}')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{in_silico_gene}_regulates_{masked_gene}_bar.png'))
    plt.close()

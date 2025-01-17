import copy
import torch


class CorruptionGenerator:
    """
    Support within-batch tabular data corruption.
    Randomly permutate the values across rows with a probability.
    """
    def __init__(self, mode, corruption_rate):
        """
        Parameters
        ----------
        mode
            Which corruption to use. Support random permutation or no corruption.
        corruption_rate
            The probability of a tabular field gots corrupted, 0<=corruption_rate<=1.
        """
        self.mode = mode if mode is not None else "identical"
        self.corruption_rate = corruption_rate
        self.last_batch = None

    def __call__(self, batch):
        if self.mode == "identical":
            return self.identical(batch)
        elif self.mode == "corruption":
            return self.corruption(batch)
        else:
            raise ValueError(
                f"Current mode {self.mode} is not supported."
                "Consider choosing from the following options:"
                "identical, corruption."
            )

    def identical(self, batch):
        batch = copy.deepcopy(batch)
        return batch

    def corruption(self, batch):
        corruption_rate = self.corruption_rate

        batch_size = batch.size()[0]
        batch = copy.deepcopy(batch)

        num_features = batch.size()[1]

        corruption_mask = torch.zeros(
            batch_size, num_features,
            dtype=torch.bool,
            device=batch.device
        ).to(batch.device)
        corruption_len = int(num_features * corruption_rate)
        for i in range(batch_size):
            corruption_idx = torch.randperm(num_features)[:corruption_len]
            corruption_mask[i, corruption_idx] = True

        indices = torch.randint(
            high=batch_size, size=(batch_size, num_features))

        """
        1.	Broadcasting:
        	•	torch.arange(num_features).unsqueeze(0) has shape (1, num_features).
        	•	It is broadcasted to shape (batch_size, num_features) to match the shape of indices.
    	2.	Resulting Tensor:
        	•	For each pair (i, j):
        	•	indices[i, j] specifies the row index.
        	•	The broadcasted torch.arange(num_features).unsqueeze(0)[i, j] specifies the column index (always j).
        	•	PyTorch fetches batch[indices[i, j], j].
        
        In the end, we corrupt the data for each gene following the marginal distribution of the gene across cells. 
        """
        random_sample = batch[indices, torch.arange(
            num_features).unsqueeze(0)].clone().to(batch.device)
        batch = torch.where(corruption_mask, random_sample, batch)

        return batch

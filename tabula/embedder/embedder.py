import math
import torch.nn as nn
from torch import Tensor


class GeneEmbedder(nn.Module):
    """
    Embedder gene id
    
    Args:
        embedding_in_feature: int
            the len of vocab
        d_token: int
            the dimension of token
    """
    def __init__(self, embedding_in_feature, d_token):
        super().__init__()
        self.embedding = nn.Embedding(embedding_in_feature, d_token)
        self.enc_norm = nn.LayerNorm(d_token)

    def forward(self, x):
        x = self.embedding(x)
        x = self.enc_norm(x)

        return x


class ValueEmbedder(nn.Module):
    """
    Embedder expression value
    
    Args:
        in_feature: int
            the number of column(gene)
        d_token: int
            the dimension of token
    """

    def __init__(self, in_feature, d_token=192, bias=True):
        super().__init__()
        self.weight = nn.Parameter(Tensor(in_feature, d_token))
        nn.init.normal_(self.weight, 1 / math.sqrt(d_token))
        if bias:
            self.bias = nn.Parameter(Tensor(in_feature, d_token))
            nn.init.normal_(self.bias, 1 / math.sqrt(d_token))
        else:
            self.bias = None
        self.enc_norm = nn.LayerNorm(d_token)

    def forward(self, x):
        """
        	1.	Input Shape:
        	    •	x has shape (batch_size, in_feature) where in_feature represents the number of genes.
        	2.	Broadcasting self.weight:
            	•	self.weight has shape (in_feature, d_token).
            	•	Adding None (or unsqueeze) at the beginning expands its shape to (1, in_feature, d_token).
            	•	The result is broadcasted across the batch dimension.
        	3.	Element-Wise Multiplication:
            	•	The input x is reshaped to (batch_size, in_feature, 1) by adding an extra dimension at the end ([..., None]).
            	•	Element-wise multiplication is performed:
            	•	For each batch, x[i, j] (a scalar gene expression value) scales the corresponding row in the weight matrix: self.weight[j].
        	4.	Result:
        	    •	x now has shape (batch_size, in_feature, d_token), where each row is the weighted embedding of the gene expression values.
     """
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x += self.bias
        x = self.enc_norm(x)

        return x


class FeatureEmbedder(nn.Module):
    """
    Gene id embedder with Expression value embedder
    """
    def __init__(self, in_feature, embedding_in_feature, d_token=192):
        super().__init__()
        self.gene_embedder = GeneEmbedder(embedding_in_feature, d_token)
        self.value_embedder = ValueEmbedder(in_feature, d_token)

    def forward(self, gene, value):
        gene = self.gene_embedder(gene)
        value = self.value_embedder(value)

        x = gene + value

        return x

import math
import torch.nn as nn
from torch import Tensor


class GeneEncoder(nn.Module):
    """
    Encoder gene id
    Parameters:
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


class ValueEncoder(nn.Module):
    """
    Encoder expression value
    Parameters:
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
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x += self.bias
        x = self.enc_norm(x)

        return x


class FeatureEncoder(nn.Module):
    """
    Gene id embedder with Expression value embedder
    """
    def __init__(self, in_feature, embedding_in_feature, d_token=192):
        super().__init__()
        self.gene_encoder = GeneEncoder(embedding_in_feature, d_token)
        self.value_encoder = ValueEncoder(in_feature, d_token)

    def forward(self, gene, value):
        gene = self.gene_encoder(gene)
        value = self.value_encoder(value)

        x = gene + value

        return x

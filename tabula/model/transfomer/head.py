import torch
import torch.nn as nn


class SupervisedHead(nn.Module):
    """
    Parameters:
        d_in: int
            the output of transformer
        d_out: int
            the number classes
    """
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.normalization = nn.LayerNorm(d_in)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(d_in, d_in, bias)
        self.linear2 = nn.Linear(d_in, d_out, bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.normalization(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class ContrastiveHead(nn.Module):
    """
    Parameters:
        d_in: int
            the output of transformer
        d_out: int
            the dimension of feature
    """
    def __init__(self, d_in, d_out, bias=True, cls=True):
        super().__init__()
        self.normalization = nn.LayerNorm(d_in)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(d_in, d_in, bias)
        self.linear2 = nn.Linear(d_in, d_out, bias)
        self.cls = cls

    def forward(self, x):
        if self.cls:
            x = x[:, :-1]
        x = self.linear1(x)
        x = self.normalization(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


class ReconstructionHead(nn.Module):
    """
    Parameters:
        d_in: int
            the output of transformer
        d_out: int
            the number of token(column)
    """
    def __init__(self, d_in, d_out, bias=True, cls=True):
        super().__init__()
        self.normalization = nn.LayerNorm(d_in)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(d_in, d_in, bias)
        self.out = nn.ModuleList([nn.Linear(d_in, 1) for _ in range(d_out)])
        self.cls = cls

    def forward(self, x):
        if self.cls:
            x = x[:, :-1]
        x = self.linear1(x)
        x = self.normalization(x)
        x = self.relu(x)
        out = [f(x[:, i]) for i, f in enumerate(self.out)]
        out = torch.cat(out, dim=1)

        return out

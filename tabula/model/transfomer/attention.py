import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

try:
    from flash_attn.modules.mha import MHA
except ImportError:
    import warnings

    warnings.warn("flash_attn is not installed")


class MutilHeadSelfAttention(nn.Module):
    def __init__(self, d_token, n_heads, dropout, bias=True):
        """
        Parameters:
            d_token: int
                the token size, the multiple of n_heads
            n_heads: int
                the number of heads
            bias: bool
                whether bias
            dropout: float
                dropout rate
        """
        super().__init__()
        self.d_token = d_token
        self.W_q = nn.Linear(d_token, d_token, bias)
        self.W_k = nn.Linear(d_token, d_token, bias)
        self.W_v = nn.Linear(d_token, d_token, bias)
        self.n_heads = n_heads
        if n_heads > 1:
            self.W_out = nn.Linear(d_token, d_token, bias)
        else:
            self.W_out = None
        self.dropout = nn.Dropout(dropout)

        # initial weight
        for m in [self.W_q, self.W_k, self.W_v]:
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def forward(self, x_q, x_kv):
        """
        Parameters:
            x_q: tensor
                query tokens
            x_kv: tensor
                key-value tokens
        """
        q = self.W_q(x_q)
        k = self.W_k(x_kv)
        v = self.W_v(x_kv)

        batch_size = q.shape[0]
        n_token = q.shape[1]
        n_heads = self.n_heads
        d_heads_key = k.shape[-1] // n_heads
        d_heads_value = v.shape[-1] // n_heads

        # calculate attention
        q = self.reshape(q)
        k = self.reshape(k)
        v = self.reshape(v)

        attention_logit = q @ k.transpose(1, 2) / math.sqrt(d_heads_key)
        attention_prob = F.softmax(attention_logit, dim=-1)
        attention_prob = self.dropout(attention_prob)
        x = attention_prob @ v
        x = x.reshape(batch_size, n_heads, n_token, d_heads_value).transpose(1, 2).reshape(batch_size, n_token,
                                                                                           n_heads * d_heads_key)
        if self.W_out is not None:
            x = self.W_out(x)

        return x

    # reshape for multi-head calculate
    def reshape(self, x):
        b, n, d = x.shape
        n_heads = self.n_heads
        d_heads = d // n_heads
        x = x.reshape(b, n, n_heads, d_heads).transpose(1, 2).reshape(b * n_heads, n, d_heads)

        return x


class AdditiveAttention(nn.Module):
    def __init__(self, d_token, n_heads, dropout, bias=True):
        """
        Parameters:
            d_token: int
                the token size, the multiple of n_heads
            n_heads: int
                the number of heads
            bias: bool
                whether bias
            dropout: float
                dropout rate
        """
        super().__init__()
        self.d_token = d_token
        self.n_heads = n_heads
        self.d_heads = d_token // n_heads
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_token, d_token, bias)
        self.v_proj = nn.Linear(d_token, d_token, bias)
        self.k_proj = nn.Linear(d_token, d_token, bias)
        self.W_q = nn.Linear(d_token, n_heads, bias)
        self.W_k = nn.Linear(d_token, n_heads, bias)
        self.out = nn.Linear(d_token, d_token, bias)

        # initial weight
        for m in [self.q_proj, self.v_proj, self.k_proj, self.W_q, self.W_k, self.out]:
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_q, x_kv):
        """
        Parameters:
            x_q: tensor
                query tokens
            x_kv: tensor
                key-value tokens
        """
        batch_size, n_q_tokens, d_token = x_q.shape
        batch_size, n_k_tokens, d_token = x_kv.shape

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        alpha = self.W_q(q) / math.sqrt(self.d_heads)
        alphas = F.softmax(alpha, dim=1)
        q_reshape = q.reshape(batch_size, n_q_tokens, self.n_heads, self.d_heads)
        q_global = torch.einsum("b t h,b t h d -> b h d", alphas, q_reshape)
        q_global = q_global.reshape(batch_size, self.n_heads * self.d_heads).unsqueeze(1)

        p = k * q_global

        beta = self.W_k(p) / math.sqrt(self.d_heads)
        betas = F.softmax(beta, dim=1)
        p_reshape = p.reshape(batch_size, n_k_tokens, self.n_heads, self.d_heads)
        k_global = torch.einsum("b t h,b t h d -> b h d", betas, p_reshape)
        k_global = k_global.reshape(batch_size, self.n_heads * self.d_heads).unsqueeze(1)

        u = v * k_global
        x = q + self.dropout(self.out(u))

        return x


class FlashAttention(nn.Module):
    def __init__(self, d_token, n_heads, dropout, bias=True):
        """
        Parameters:
            d_token: int
                the token size, the multiple of n_heads
            n_heads: int
                the number of heads
            bias: bool
                whether bias
            dropout: float
                dropout rate
        """
        super().__init__()
        self.attention = MHA(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=dropout,
            qkv_proj_bias=bias,
            use_flash_attn=True,
        )

    def forward(self, x_q, x_kv):
        """
        Parameters:
            x_q: tensor
                query tokens
            x_kv: tensor
                key-value tokens
        """
        with autocast():
            x_q = x_q.half()
            x = self.attention(x_q)
            x = x.float()

        return x

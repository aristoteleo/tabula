import math
import torch
import torch.nn as nn

from typing import cast
from torch import Tensor
from typing import Optional, Dict, Union
from torch.distributions import Bernoulli

from tabula.embedder.embedder import FeatureEmbedder
from tabula.model.transfomer.activation import ReGLU
from tabula.model.transfomer.attention import AdditiveAttention, MutilHeadSelfAttention, FlashAttention
from tabula.model.transfomer.head import ContrastiveHead, ReconstructionHead, SupervisedHead
from tabula.model.grad_reverse import grad_reverse
from tabula import logger


LABEL = "label"


class AppendCLS(nn.Module):
    """
    Append <cls> token to the end of sequenece.
    """
    def __init__(self, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, d_token))
        nn.init.normal_(self.weight, std=1 / math.sqrt(d_token))

    def forward(self, x):
        cls_vector = self.weight.expand(x.size(0), 1, -1)
        return torch.cat([x, cls_vector], dim=1)


class FeedForward(nn.Module):
    """
    The feed forward module of transformer after attention.
    """
    def __init__(self, d_token, d_hidden, dropout, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(d_token, d_hidden * 2, bias)
        self.ReGLU = ReGLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_token, bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.ReGLU(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """
    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class BatchLabelEncoder(nn.Module):
    """
    Encode the batch label to the embedding
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class MGMDecoder(nn.Module):
    """
    Decoder for the Masked Gene Modeling task, there are two returns:
    1. pred: the predicted value for the expression
    2. zero_probs: the probability of the expression being zero, approaching 0 or 1
    """
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)


class RCSCDecoder(nn.Module):
    """
    Decoder for Tabula cell context-aware reconstruction prediction.
    """
    def __init__(self,
                 d_model: int,
                 arch_style: str = "inner product",
                 query_activation: nn.Module = nn.Sigmoid,
                 explicit_zero_prob: bool = False,
                 use_batch_labels: bool = False,
                 ):
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:
                self.W_zero_logit = nn.Linear(d_model, d_in)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        else:
            raise ValueError(f"Unknown arch_style: {self.arch_style}")


class CMGMDecoder(nn.Module):
    """
    Decoder for the cell context-aware masked gene modeling.
    """
    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)


class Transformer(nn.Module):
    def __init__(self,
                 d_token,
                 n_blocks,
                 residual_dropout,
                 additive_attention,
                 flash_attention,
                 attention_n_heads,
                 attention_dropout,
                 ffn_d_hidden,
                 ffn_dropout,
                 pre_normalization,
                 global_token,
                 ):
        """
        Parameters:
            d_token: int
                the token size
            n_blocks: int
                the number of attention via ffn blocks
            residual_dropout: float
                the dropout rate in residual
            additive_attention: bool
                whether fastFormer
            flash_attention: bool
                whether flash attention
            attention_n_heads: int
                the number of head
            attention_dropout: float
                the dropout rate in attention
            ffn_d_hidden: int
                the hidden dimension in ffn
            ffn_dropout: float
                the dropout rate in ffn
            pre_normalization: bool
                whether pre normalization
            global_token: bool
                whether global_token in blocks
        """
        super().__init__()
        self.pre_normalization = pre_normalization
        self.global_token = global_token
        self.blocks = nn.ModuleList([])
        self.head = nn.Identity()

        for i in range(n_blocks):
            layer = nn.ModuleDict(
                {
                    "attention": AdditiveAttention(
                        d_token=d_token,
                        n_heads=attention_n_heads,
                        bias=True,
                        dropout=attention_dropout,
                    )
                    if additive_attention
                    else (
                        FlashAttention(
                            d_token=d_token,
                            n_heads=attention_n_heads,
                            bias=True,
                            dropout=attention_dropout,
                        )
                        if flash_attention
                        else MutilHeadSelfAttention(
                            d_token=d_token,
                            n_heads=attention_n_heads,
                            bias=True,
                            dropout=attention_dropout,
                        )
                    ),
                    "ffn": FeedForward(
                        d_token=d_token,
                        d_hidden=ffn_d_hidden,
                        dropout=ffn_dropout,
                        bias=True,
                    ),
                    "attention_residual_dropout": nn.Dropout(residual_dropout),
                    "ffn_residual_dropout": nn.Dropout(residual_dropout),
                    "output": nn.Identity(),
                }
            )
            self.blocks.append(layer)

    def start_residual(self, layer, stage, x):
        x_residual = x
        if self.pre_normalization:
            norm_key = f"{stage}_normalization"
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)

        return x_residual

    def end_residual(self, layer, stage, x, x_residual):
        x_residual = layer[f"{stage}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.pre_normalization:
            x = layer[f"{stage}_normalization"](x)

        return x

    def start_global_token(self, x):
        if self.global_token:
            x = torch.cat(
                [torch.mean(x, dim=1).unsqueeze(1), x],
                dim=1,
            )
        return x

    def end_global_token(self, x):
        if self.global_token:
            x = x[:, 1:]
        return x

    def forward(self, x):
        for layer_idx, layer in enumerate(self.blocks):
            layer = cast(nn.ModuleDict, layer)
            x = self.start_global_token(x)
            x_residual = self.start_residual(layer, "attention", x)
            x_residual = layer["attention"](
                x_residual,
                x_residual,
            )
            x = self.end_residual(layer, "attention", x, x_residual)
            x_residual = self.start_residual(layer, "ffn", x)
            x_residual = layer["ffn"](x_residual)
            x = self.end_residual(layer, "ffn", x, x_residual)
            x = layer["output"](x)
            x = self.end_global_token(x)

        x = self.head(x)

        return x


class TabulaTransformer(nn.Module):
    def __init__(self,
                 in_feature,
                 embedding_in_feature,
                 contrastive_out_feature,
                 supervised_out_feature,
                 d_token,
                 n_blocks,
                 residual_dropout,
                 additive_attention,
                 flash_attention,
                 attention_n_heads,
                 attention_dropout,
                 ffn_d_hidden,
                 ffn_dropout,
                 cls,
                 pre_normalization,
                 global_token,
                 pretrain_objective,
                 enable_batch=False,
                 explicit_zero_prob=False,
                 do_mgm=False,
                 do_cmgm=False,
                 cmgm_decoder_style: str = "inner product",
                 do_dab=False,
                 n_batch: Optional[int] = None,
                 embed_style: str = "cls",
                 ):
        """
        Parameters:
            in_feature: int
                the number of column
            embedding_in_feature: int
                the number of vocab
            contrastive_out_feature: int
                the dimension when contrastive
            supervised_out_feature: int
                number of class of cell types, head = num_classes, for supervised cell type annotation
            d_token: int
                the token size
            n_blocks: int
                the number of attention via ffn blocks
            residual_dropout: float
                the dropout rate in residual
            additive_attention: bool
                whether fastFormer
            attention_n_heads: int
                the number of head
            attention_dropout: float
                the dropout rate in attention
            ffn_d_hidden: int
                the hidden dimension in ffn
            ffn_dropout: float
                the dropout rate in ffn
            cls: bool
                whether append cls in embedding
            pre_normalization: bool
                whether pre normalization
            global_token: bool
                whether global_token in blocks
            pretrain_objective:
                "supervised", "reconstruction" ,"contrastive" ,"both" : both "reconstruction" and "contrastive"
            enable_batch: bool
                whether integrate batch information into the model
            explicit_zero_prob: bool
                if True, the output of the decoder from mgm and cmgm will include the zero_probs, approaching 0 or 1
            do_mgm: bool
                Specific for mask gene modeling
            do_cmgm: bool
                Specific for cell context-aware gene masked modeling for cell modelling
            cmgm_decoder_style: str
                style of cmgm decoder, choice from "inner product", "concat query", "sum query"
            do_dab: bool
                employ a specific MLP classifier for batch correction
            n_batch: int
                the number of batch label in the dataset
            embed_style: str
                the style of get cell embedding, choice from "cls", "avg-pool"
        """
        super().__init__()
        self.prefix = "TabulaTransformer"
        self.feature_tokenizer = FeatureEmbedder(in_feature=in_feature,
                                                 embedding_in_feature=embedding_in_feature,
                                                 d_token=d_token)
        self.bn = nn.BatchNorm1d(d_token, eps=6.1e-5)
        self.cls = AppendCLS(d_token=d_token) if cls else nn.Identity()
        self.transformer = Transformer(d_token=d_token,
                                       n_blocks=n_blocks,
                                       residual_dropout=residual_dropout,
                                       additive_attention=additive_attention,
                                       flash_attention=flash_attention,
                                       attention_n_heads=attention_n_heads,
                                       attention_dropout=attention_dropout,
                                       ffn_d_hidden=ffn_d_hidden,
                                       ffn_dropout=ffn_dropout,
                                       pre_normalization=pre_normalization,
                                       global_token=global_token)

        self.enable_batch = enable_batch
        if self.enable_batch:
            self.batch_encoder = BatchLabelEncoder(num_embeddings=n_batch, embedding_dim=d_token)

        self.n_batch = n_batch
        self.bn = nn.BatchNorm1d(d_token, eps=6.1e-5)

        self.explicit_zero_prob = explicit_zero_prob
        self.embed_style = embed_style

        self.do_mgm = do_mgm
        if self.do_mgm:
            self.mgm_decoder = MGMDecoder(d_model=d_token,
                                          explicit_zero_prob=self.explicit_zero_prob,
                                          use_batch_labels=enable_batch)

        self.do_cmgm = do_cmgm
        if self.do_cmgm:
            self.cmgm_decoder = CMGMDecoder(d_model=d_token,
                                            arch_style=cmgm_decoder_style,
                                            explicit_zero_prob=self.explicit_zero_prob,
                                            use_batch_labels=enable_batch)

        self.heads = nn.ModuleDict(
            {
                "supervised": SupervisedHead(d_in=d_token, d_out=supervised_out_feature),
                "contrastive": ContrastiveHead(d_in=d_token, d_out=contrastive_out_feature, cls=cls)
                if pretrain_objective in ["contrastive", "both"]
                else None,
                "reconstruction": ReconstructionHead(d_in=d_token, d_out=in_feature, cls=cls)
                if pretrain_objective in ["reconstruction", "both"]
                else None,
            }
        )
        self.pert_encoder = nn.Embedding(3, d_token, padding_idx=2)

        self.do_dab = do_dab
        if self.do_dab:
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                d_token,
                n_cls=n_batch,
                reverse_grad=True,
            )

    def forward(self,
                genes: Tensor,
                values: Tensor,
                batch_info=None,
                pert_flags=None,
                head=None,
                do_sample: bool = False,
                do_dab: bool = False,
                do_mgm: bool = False,
                do_cmgm: bool = False,
                ):
        """
        Parameters
        ----------
        genes: Tensor
        values: Tensor
        batch_info: Tensor
        head: str, choice from "supervised", "contrastive", "reconstruction"
        do_sample: bool, whether to sample from the bernoulli distribution with the zero_probs
        do_dab: bool, whether to compute the domain adversarial batch correction objective output
        do_mgm: bool, whether to compute the masked gene modeling objective output
        do_cmgm: bool, whether to compute the cell context-aware masked gene modeling objective output
        """
        output = {}
        x = self.feature_tokenizer(genes, values)
        output['feature_tokenizer'] = x

        if pert_flags is not None:
            pert_emb = self.pert_encoder(pert_flags)
            x = x + pert_emb

        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.cls(x)
        output['transformer'] = self.transformer(x)

        # batch integration for downstream task if mutil batch are provided
        if self.enable_batch and batch_info is not None:
            output['batch_emb'] = self.batch_encoder(batch_info)
            # concat batch_embed to output of transformer
            transformed_batch_embed = output['batch_emb'].unsqueeze(1).repeat(1, output['transformer'].shape[1], 1)
            batch_concat = torch.cat([output['transformer'], transformed_batch_embed], dim=2)

        cell_embed = self._get_cell_embed(output['transformer'], self.embed_style)
        output['cell_embed'] = cell_embed

        if head == 'supervised':
            output['supervised'] = self.heads[head](cell_embed)
        elif head == 'contrastive':
            output['contrastive'] = self.heads[head](output['transformer'])
        elif head == 'reconstruction':
            output['reconstruction'] = self.heads[head](output['transformer'])
        else:
            if head is not None:
                logger.warning(f"Unknown head: {head}")

        if self.do_mgm and do_mgm:
            mgm_output = self.mgm_decoder(batch_concat[:, :-1, :]
                                          if self.enable_batch and batch_info is not None
                                          else output['transformer'][:, :-1, :],
                                          )
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mgm_output["zero_probs"])
                output["mgm_pred"] = bernoulli.sample() * mgm_output["pred"]
            else:
                output['mgm_pred'] = mgm_output['pred']
            if self.explicit_zero_prob:
                output['mgm_zero_probs'] = mgm_output['zero_probs']

        if self.do_cmgm and do_cmgm:
            cmgm_output = self.cmgm_decoder(
                cell_emb=torch.cat([output['transformer'][:, -1, :], output['batch_emb']], dim=1)
                if self.enable_batch
                else output['transformer'][:, -1, :],
                gene_embs=self.feature_tokenizer.gene_encoder(genes),  #
            )
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=cmgm_output["zero_probs"])
                output["cmgm_output"] = bernoulli.sample() * cmgm_output["pred"]
            else:
                output['cmgm_output'] = cmgm_output['pred']
            if self.explicit_zero_prob:
                output['cmgm_zero_probs'] = cmgm_output['zero_probs']

        if self.do_dab and do_dab:
            output['dab_output'] = self.grad_reverse_discriminator(cell_embed)

        return output

    def _get_cell_embed(self, x: Tensor, cell_embed_style: str) -> Tensor:
        """
        Get the cell embedding from the output of the transformer.
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
            cell_embed_style: str, choice from "cls", "avg-pool", "w-pool"
        """
        output = None
        if cell_embed_style == "cls":
            output = x[:, -1, :]
        elif cell_embed_style == "avg-pool":
            output = torch.mean(x, dim=1)
        elif cell_embed_style == "w-pool":
            raise NotImplementedError
        return output

    def _get_gene_embed(self, x: Tensor) -> Tensor:
        """
        Get the cell embedding from the output of the transformer.
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        output = torch.mean(x, dim=0)

        return output

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

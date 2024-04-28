from random import sample
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn

from .common import MultiHeadAttention, PositionwiseFeedForward


class BERT4RecBlock(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_prob, hidden_act="gelu"):
        super(BERT4RecBlock, self).__init__()
        self.attention = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_size, dropout_prob, hidden_act)

    def forward(self, input_enc, mask):
        output_enc, attn_dist = self.attention(input_enc, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist


class BERT4Rec(nn.Module):
    def __init__(
        self,
        num_item: int,
        num_cat: int,
        idx_groups: Optional[dict] = None,
        linear_in_size: Optional[
            int
        ] = None,  # TODO: raise warning if this parameter is set and use_linear is False, and vice versa.
        hidden_size: int = 256,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        hidden_act: Literal["gelu", "mish", "silu"] = "gelu",
        max_len: int = 30,
        dropout_prob: float = 0.2,
        pos_emb: bool = True,
        cat_emb: bool = False,
        use_linear: bool = True,  # True if using linear layer at last
        device: str = "cpu",
        **kwargs
    ):
        super(BERT4Rec, self).__init__()

        self.idx_groups = idx_groups
        self.hidden_size = hidden_size
        self.num_item = num_item
        self.pos_emb = pos_emb
        self.cat_emb = cat_emb
        self.use_linear = use_linear
        self.device = device

        self.in_size = hidden_size if linear_in_size is None else linear_in_size

        self.item_emb = nn.Embedding(num_item + 2, hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout_prob)
        self.emb_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)

        if self.pos_emb:
            self.positional_emb = nn.Embedding(max_len, hidden_size)
        if self.cat_emb:
            self.category_emb = nn.Embedding(num_cat, hidden_size)

        self.bert = nn.ModuleList(
            [
                BERT4RecBlock(num_attention_heads, hidden_size, dropout_prob, hidden_act)
                for _ in range(num_hidden_layers)
            ]
        )

        if self.use_linear:
            self.out = nn.Linear(self.in_size, self.num_item + 1)

    def forward(self, log_seqs, labels, **kwargs):
        # TODO: try to remove this two block(206~214)
        if self.cat_emb is not None:
            item_ids = log_seqs.clone().detach()
            mask_index = torch.where(item_ids == self.num_item + 1)  # mask 찾기
            item_ids[mask_index] = labels[mask_index]  # mask의 본래 아이템 번호 찾기
            item_ids -= 1
        if self.idx_groups is not None:
            item_ids = np.vectorize(lambda x: sample(self.idx_groups[x], k=1)[0] if x != -1 else -1)(
                item_ids.detach().cpu()
            )

        seqs = self.item_emb(log_seqs).to(self.device)
        attn_mask = (log_seqs > 0).unsqueeze(1).repeat(1, log_seqs.shape[1], 1).unsqueeze(1).to(self.device)

        if self.pos_emb:
            positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
            seqs += self.positional_emb(torch.tensor(positions).to(self.device))
        if self.cat_emb:
            seqs += self.category_emb(self.item_prod_type[item_ids])

        seqs = self.emb_layernorm(self.dropout(seqs))

        for block in self.bert:
            seqs, _ = block(seqs, attn_mask)
        bert_out = seqs

        out = self.out(bert_out) if self.use_linear else bert_out
        return out

from typing import Literal

import torch
import torch.nn as nn

from .bert import BERT4Rec
from .mlp import MLPRec


class MLPBERT4Rec(nn.Module):
    def __init__(
        self,
        num_item: int,
        modal_emb_size: int,
        hidden_size: int = 256,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        hidden_act: Literal["gelu", "mish", "silu"] = "gelu",
        num_gen_img: int = 1,
        max_len: int = 30,
        dropout_prob: float = 0.2,
        pos_emb: bool = True,
        num_mlp_layers: int = 2,
        device: str = "cpu",
        merge: str = "concat",
        **kwargs
    ):
        super(MLPBERT4Rec, self).__init__()

        self.num_item = num_item
        self.modal_emb_size = modal_emb_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.device = device
        self.pos_emb = pos_emb
        self.num_mlp_layers = num_mlp_layers
        self.num_gen_img = num_gen_img
        self.merge = merge

        if self.merge == "concat":
            in_size = self.hidden_size + self.modal_emb_size
        elif self.merge == "mul":
            self.mul_linear = nn.Linear(hidden_size, modal_emb_size)

        self.bert4rec_module = BERT4Rec(
            num_item=num_item,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_act=hidden_act,
            max_len=max_len,
            dropout_prob=dropout_prob,
            pos_emb=pos_emb,
            use_linear=False,
            device=device,
            **kwargs
        )

        self.mlp_module = MLPRec(
            num_item=num_item,
            num_gen_img=num_gen_img,
            linear_in_size=in_size,
            num_mlp_layers=num_mlp_layers,
            use_linear=False,
            hidden_act=hidden_act,
            device=device,
            **kwargs
        )

        self.out = nn.Linear(in_size // (2**num_mlp_layers), self.num_item + 1)

    def forward(self, log_seqs, modal_emb, labels):
        bert_out = self.bert4rec_module(log_seqs=log_seqs, modal_emb=modal_emb)
        mlp_merge = modal_emb * (labels != 0).unsqueeze(-1)  # loss 계산에 포함되지 않는 것 0으로 변경
        mlp_mask = (log_seqs > 0).unsqueeze(-1).repeat(1, 1, self.modal_emb_size).to(self.device)

        if self.merge == "concat":
            mlp_in = torch.concat([bert_out, mlp_merge * mlp_mask], dim=-1)
        elif self.merge == "mul":
            mlp_in = self.mul_linear(bert_out) * mlp_merge

        mlp_out = self.mlp_module(mlp_in)
        out = self.out(mlp_out)

        return out

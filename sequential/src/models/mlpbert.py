from typing import Literal, Optional

import torch
import torch.nn as nn

from .bert import BERT4Rec
from .mlp import MLPRec


class MLPBERT4Rec(nn.Module):
    def __init__(
        self,
        num_item: int,
        num_cat: int,
        gen_img_emb: torch.Tensor,  # TODO: try to remove
        item_prod_type: torch.Tensor,
        idx_groups: Optional[dict] = None,
        hidden_size: int = 256,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        hidden_act: Literal["gelu", "mish", "silu"] = "gelu",
        num_gen_img: int = 1,
        max_len: int = 30,
        dropout_prob: float = 0.2,
        pos_emb: bool = True,
        cat_emb: bool = False,
        mlp_cat: bool = False,
        num_mlp_layers: int = 2,
        device: str = "cpu",
        text_emb: Optional[torch.Tensor] = None,
        merge: str = "concat",
        **kwargs
    ):
        super(MLPBERT4Rec, self).__init__()

        self.num_item = num_item
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.device = device
        self.pos_emb = pos_emb
        self.cat_emb = cat_emb
        self.mlp_cat = mlp_cat
        self.num_mlp_layers = num_mlp_layers
        self.num_gen_img = num_gen_img
        self.merge = merge
        self.gen_img_emb = gen_img_emb.to(self.device) if self.num_gen_img else None  # (num_item) X (3*512)
        self.text_emb = text_emb.to(self.device) if text_emb is not None else text_emb  # (num_item) X (3*512)

        self.item_prod_type = item_prod_type.to(self.device)  # [item_id : category]
        self.idx_groups = idx_groups

        if self.merge == "concat":
            in_size = self.hidden_size + self.hidden_size * self.mlp_cat
            if self.text_emb is not None:
                in_size += self.text_emb.shape[-1]
            if self.num_gen_img:
                in_size += self.gen_img_emb.shape[-1] * self.num_gen_img
        elif self.merge == "mul":
            in_size = self.gen_img_emb.shape[-1] * self.num_gen_img
            self.mul_linear = nn.Linear(hidden_size, in_size)

        self.bert4rec_module = BERT4Rec(
            num_item=num_item,
            num_cat=num_cat,
            idx_groups=idx_groups,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_act=hidden_act,
            max_len=max_len,
            dropout_prob=dropout_prob,
            pos_emb=pos_emb,
            cat_emb=cat_emb,
            use_linear=False,
            device=device,
            **kwargs
        )

        self.mlp_module = MLPRec(
            num_item=num_item,
            num_cat=num_cat,
            item_prod_type=item_prod_type,
            gen_img_emb=gen_img_emb,
            num_gen_img=num_gen_img,
            idx_groups=idx_groups,
            linear_in_size=in_size,
            hidden_size=hidden_size,
            num_mlp_layers=num_mlp_layers,
            mlp_cat=mlp_cat,
            text_emb=text_emb,
            use_linear=False,
            hidden_act=hidden_act,
            device=device,
            **kwargs
        )

        self.out = nn.Linear(in_size // (2**num_mlp_layers), self.num_item + 1)

    def forward(self, log_seqs, gen_img, labels):
        bert_out = self.bert4rec_module(log_seqs=log_seqs, gen_img=gen_img, labels=labels)

        mlp_merge = gen_img * (labels != 0).unsqueeze(-1)  # loss 계산에 포함되지 않는 것 0으로 변경
        mlp_mask = (log_seqs > 0).unsqueeze(-1).repeat(1, 1, gen_img.shape[-1]).to(self.device)

        if self.merge == "concat":
            mlp_in = torch.concat([bert_out, mlp_merge * mlp_mask], dim=-1)
        elif self.merge == "mul":
            mlp_in = self.mul_linear(bert_out) * mlp_merge
        mlp_out = self.mlp_module(mlp_in)
        out = self.out(mlp_out)
        return out

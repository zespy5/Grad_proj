from typing import Literal, Optional

import torch
import torch.nn as nn


class MLPRec(nn.Module):
    def __init__(
        self,
        num_item: int,
        gen_img_emb: torch.Tensor,
        num_gen_img: int = 1,
        linear_in_size: Optional[int] = None,
        num_mlp_layers: int = 2,
        text_emb: Optional[torch.Tensor] = None,
        use_linear: bool = True,
        hidden_act: Literal["gelu", "mish", "selu"] = "gelu",
        device: str = "cpu",
        **kwargs
    ):
        super(MLPRec, self).__init__()

        self.num_item = num_item
        self.device = device
        self.num_gen_img = num_gen_img
        self.gen_img_emb = gen_img_emb.to(self.device) if self.num_gen_img else None
        self.in_size = self.gen_img_emb.shape[-1] * self.num_gen_img if linear_in_size is None else linear_in_size
        self.text_emb = text_emb
        self.use_linear = use_linear
        self.hidden_act = hidden_act
        self.num_mlp_layers = num_mlp_layers

        self.MLP_modules = []

        if self.hidden_act == "gelu":
            self.activate = nn.GELU()
        if self.hidden_act == "mish":
            self.activate = nn.Mish()
        if self.hidden_act == "silu":
            self.activate = nn.SiLU()

        for _ in range(self.num_mlp_layers):
            self.MLP_modules.append(nn.Linear(self.in_size, self.in_size // 2))
            self.MLP_modules.append(self.activate)
            self.in_size = self.in_size // 2

        self.MLP = nn.Sequential(*self.MLP_modules)
        if self.use_linear:
            self.out = nn.Linear(self.in_size, self.num_item + 1)

    def forward(self, x, **kwargs):
        # gen_img *= (labels != 0).unsqueeze(-1)  # loss 계산에 포함되지 않는 것 0으로 변경
        # mlp_mask = (log_seqs > 0).unsqueeze(-1).repeat(1, 1, gen_img.shape[-1]).to(self.device)

        out = self.MLP(x)  # TODO: or mlp_merge * mlp_mask?
        if self.use_linear:
            out = self.out(out)

        return out

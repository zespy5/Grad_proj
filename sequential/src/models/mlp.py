from typing import Literal, Optional

import torch.nn as nn


class MLPRec(nn.Module):
    def __init__(
        self,
        num_item: int,
        linear_in_size: Optional[int] = None,
        num_mlp_layers: int = 2,
        use_linear: bool = True,
        hidden_act: Literal["gelu", "mish", "selu"] = "gelu",
        device: str = "cpu",
        **kwargs
    ):
        super(MLPRec, self).__init__()
        activates = {"gelu": nn.GELU(), "mish": nn.Mish(), "silu": nn.SiLU()}

        self.num_item = num_item
        self.device = device
        self.in_size = linear_in_size
        self.use_linear = use_linear
        self.hidden_act = hidden_act
        self.num_mlp_layers = num_mlp_layers

        self.MLP_modules = []
        self.activate = activates[hidden_act]

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

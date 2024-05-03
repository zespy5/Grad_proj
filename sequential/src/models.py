import math
from random import sample
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertForMaskedLM


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_units, dropout_prob):
        super(ScaledDotProductAttention, self).__init__()
        self.head_units = head_units
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, Q, K, V, mask):
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_units)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_dist = self.dropout(F.softmax(attn_score, dim=-1))
        # dim of output : batchSize x num_head x seqLen x head_units
        output = torch.matmul(attn_dist, V)
        return output, attn_dist


class MultiHeadAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_prob):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.head_units = self.hidden_size // self.num_attention_heads

        # query, key, value, output
        self.W_Q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_K = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_O = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.attention = ScaledDotProductAttention(self.head_units, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.layerNorm = nn.LayerNorm(self.hidden_size, 1e-6)

    def forward(self, enc, mask):
        residual = enc  # residual connection
        batch_size, seqlen = enc.size(0), enc.size(1)

        Q = self.W_Q(enc).view(batch_size, seqlen, self.num_attention_heads, self.head_units)
        K = self.W_K(enc).view(batch_size, seqlen, self.num_attention_heads, self.head_units)
        V = self.W_V(enc).view(batch_size, seqlen, self.num_attention_heads, self.head_units)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        output, attn_dist = self.attention(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seqlen, -1)

        output = self.layerNorm(self.dropout(self.W_O(output)) + residual)
        return output, attn_dist


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_prob, hidden_act="gelu"):
        super(PositionwiseFeedForward, self).__init__()

        self.W_1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.W_2 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layerNorm = nn.LayerNorm(hidden_size, 1e-6)

        if hidden_act == "gelu":
            self.activate = F.gelu
        if hidden_act == "mish":
            self.activate = F.mish
        if hidden_act == "silu":
            self.activate = F.silu

    def forward(self, x):
        residual = x
        output = self.W_2(self.activate(self.dropout(self.W_1(x))))
        output = self.layerNorm(self.dropout(output) + residual)
        return output


class BERT4RecBlock(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_prob, hidden_act="gelu"):
        super(BERT4RecBlock, self).__init__()
        self.attention = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_size, dropout_prob, hidden_act)

    def forward(self, input_enc, mask):
        output_enc, attn_dist = self.attention(input_enc, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist


class BERT4RecWithHF(nn.Module):
    def __init__(
        self,
        num_items,
        hidden_size=256,
        num_attention_heads=4,
        num_hidden_layers=3,
        hidden_act="gelu",
        max_len=30,
        dropout_prob=0.2,
        pos_emb=False,
        device="cpu",
        **kwargs,
    ):
        super(BERT4RecWithHF, self).__init__()

        self.num_items = num_items
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_len = max_len
        self.dropout_prob = dropout_prob
        self.pos_emb = pos_emb
        self.device = device

        # init BERT
        bert_config = BertConfig(
            vocab_size=self.num_items + 2,  # mask, padding
            hidden_size=self.hidden_size,
            intermediate_size=4 * self.hidden_size,
            max_position_embeddings=self.max_len,
            attention_probs_dropout_prob=self.dropout_prob,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.dropout_prob,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
        )
        self.bert = BertForMaskedLM(bert_config)

        if not pos_emb:
            # remove pos_emb
            pos_emb_shape = self.bert.bert.embeddings.position_embeddings.weight.shape
            self.bert.bert.embeddings.position_embeddings.weight.data = torch.zeros(pos_emb_shape)
            self.bert.bert.embeddings.position_embeddings.weight.requires_grad = False

    def forward(self, tokens):
        token_type_ids = torch.zeros_like(tokens).to(self.device)
        mask = tokens > 0
        output = self.bert(
            tokens,
            attention_mask=mask,
            token_type_ids=token_type_ids,
        )

        return output.logits


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
        **kwargs,
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


class MLPRec(nn.Module):
    def __init__(
        self,
        num_item: int,
        num_cat: int,
        item_prod_type: torch.Tensor,
        gen_img_emb: torch.Tensor,
        num_gen_img: int = 1,
        idx_groups: Optional[dict] = None,
        linear_in_size: Optional[int] = None,
        hidden_size: int = 256,
        num_mlp_layers: int = 2,
        mlp_cat: bool = False,
        text_emb: Optional[torch.Tensor] = None,
        use_linear: bool = True,
        hidden_act: Literal["gelu", "mish", "selu"] = "gelu",
        device: str = "cpu",
        **kwargs,
    ):
        super(MLPRec, self).__init__()

        self.num_item = num_item
        self.item_prod_type = item_prod_type
        self.device = device
        self.num_gen_img = num_gen_img
        self.gen_img_emb = gen_img_emb.to(self.device) if self.num_gen_img else None
        self.idx_groups = idx_groups
        self.in_size = self.gen_img_emb.shape[-1] * self.num_gen_img if linear_in_size is None else linear_in_size
        self.mlp_cat = mlp_cat
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

        if self.mlp_cat:
            self.category_emb = nn.Embedding(num_cat, hidden_size)

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
        **kwargs,
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
            **kwargs,
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
            **kwargs,
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


class MLPwithBERTFreeze(MLPBERT4Rec):
    def __init__(
        self,
        num_item: int,
        num_cat: int,
        model_ckpt_path: str,
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
        **kwargs,
    ):
        super(MLPwithBERTFreeze, self).__init__(
            num_item=num_item,
            num_cat=num_cat,
            gen_img_emb=gen_img_emb,
            item_prod_type=item_prod_type,
            idx_groups=idx_groups,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_act=hidden_act,
            num_gen_img=num_gen_img,
            max_len=max_len,
            dropout_prob=dropout_prob,
            pos_emb=pos_emb,
            cat_emb=cat_emb,
            mlp_cat=mlp_cat,
            num_mlp_layers=num_mlp_layers,
            device=device,
            text_emb=text_emb,
            merge=merge,
            **kwargs,
        )

        bert4rec_weight = torch.load(model_ckpt_path)

        try:
            incompat_keys = self.bert4rec_module.load_state_dict(bert4rec_weight, strict=False)
            assert incompat_keys.unexpected_keys == ["out.weight", "out.bias"]

            for param in self.bert4rec_module.parameters():
                param.requires_grad = False
        except AssertionError as e:
            print(f"Unexpected keys: {incompat_keys}")
            raise e


class RegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super().__init__()
        self.reg_loss = RegLoss()
        self.gamma = gamma

    def forward(self, pos_score, neg_scores, parameters):
        diff = pos_score - neg_scores
        is_same = diff != 0
        sig_diff = torch.sigmoid(diff)

        num = torch.sum(is_same)

        loss = -torch.log(self.gamma + sig_diff)
        loss = is_same * loss
        loss = torch.sum(loss) / num

        reg_loss = self.reg_loss(parameters)
        return loss + reg_loss

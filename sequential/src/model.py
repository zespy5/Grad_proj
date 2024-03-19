import math
from random import sample

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

        if self.hidden_act == "gelu":
            self.activate = F.gelu
        if self.hidden_act == "mish":
            self.activate = F.mish
        if self.hidden_act == "silu":
            self.activate = F.silu

    def forward(self, x):
        residual = x
        output = self.W_2(self.hidden_act(self.dropout(self.W_1(x))))
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


class BERT4Rec(nn.Module):
    def __init__(
        self,
        num_item,
        hidden_size=256,
        num_attention_heads=4,
        num_hidden_layers=3,
        hidden_act="gelu",
        max_len=30,
        dropout_prob=0.2,
        pos_emb=True,
        device="cpu",
    ):
        super(BERT4Rec, self).__init__()

        self.num_item = num_item
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.pos_emb = pos_emb
        self.device = device

        self.item_emb = nn.Embedding(num_item + 2, hidden_size, padding_idx=0)
        self.positional_emb = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.emb_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.bert = nn.ModuleList(
            [
                BERT4RecBlock(num_attention_heads, hidden_size, dropout_prob, hidden_act)
                for _ in range(num_hidden_layers)
            ]
        )

        self.out = nn.Linear(hidden_size, self.num_item + 1)

    def forward(self, tokens):
        seqs = self.item_emb(tokens).to(self.device)
        if self.pos_emb:
            positions = np.tile(np.array(range(tokens.shape[1])), [tokens.shape[0], 1])
            seqs += self.positional_emb(torch.tensor(positions).to(self.device))
        seqs = self.emb_layernorm(self.dropout(seqs))

        mask = (tokens > 0).unsqueeze(1).repeat(1, tokens.shape[1], 1).unsqueeze(1).to(self.device)

        for block in self.bert:
            seqs, _ = block(seqs, mask)

        out = self.out(seqs)
        return out


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


class MLPBERT4Rec(nn.Module):
    def __init__(
        self,
        num_item,
        gen_img_emb,
        num_cat,
        item_prod_type,
        hidden_size=256,
        num_attention_heads=4,
        num_hidden_layers=3,
        hidden_act="gelu",
        num_gen_img=1,
        max_len=30,
        dropout_prob=0.2,
        pos_emb=False,
        cat_emb=False,
        num_mlp_layers=2,
        device="cpu",
    ):
        super(MLPBERT4Rec, self).__init__()

        self.num_item = num_item
        self.num_cat = num_cat
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.device = device
        self.pos_emb = pos_emb
        self.cat_emb = cat_emb
        self.num_mlp_layers = num_mlp_layers
        self.num_gen_img = num_gen_img
        self.gen_img_emb = gen_img_emb.to(self.device)  # (num_item) X (3*512)
        self.item_prod_type = item_prod_type.to(self.device)  # [item_id : category]

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

        # init MLP
        self.MLP_modules = []
        in_size = self.hidden_size + self.gen_img_emb.shape[-1] * self.num_gen_img

        if self.hidden_act == "gelu":
            self.activate = nn.GELU()
        if self.hidden_act == "mish":
            self.activate = nn.Mish()
        if self.hidden_act == "silu":
            self.activate = nn.SiLU()

        for _ in range(self.num_mlp_layers):
            self.MLP_modules.append(nn.Linear(in_size, in_size // 2))
            self.MLP_modules.append(self.activate)
            in_size = in_size // 2

        self.MLP = nn.Sequential(*self.MLP_modules)
        self.out = nn.Linear(in_size, self.num_item + 1)

    def forward(self, log_seqs, labels):
        seqs = self.item_emb(log_seqs).to(self.device)

        if self.pos_emb:
            positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
            seqs += self.positional_emb(torch.tensor(positions).to(self.device))
        if self.cat_emb:
            seqs += self.category_emb(self.item_prod_type[log_seqs])

        seqs = self.emb_layernorm(self.dropout(seqs))
        mask = (log_seqs > 0).unsqueeze(1).repeat(1, log_seqs.shape[1], 1).unsqueeze(1).to(self.device)

        for block in self.bert:
            seqs, _ = block(seqs, mask)

        item_ids = log_seqs.clone().detach()
        mask_index = torch.where(item_ids == self.num_item + 1)  # mask 찾기
        item_ids[mask_index] = labels[mask_index]  # mask의 본래 아이템 번호 찾기
        # log_seq는 원래 아이템 id + 1 되어 있으므로 -1해서 인덱싱
        img_idx = sample([0, 1, 2], k=self.num_gen_img)  # 생성형 이미지 추출
        gen_imgs = torch.flatten(self.gen_img_emb[item_ids - 1][:, :, img_idx, :], start_dim=-2, end_dim=-1)
        mlp_in = torch.concat([seqs, gen_imgs], dim=-1)
        out = self.out(self.MLP(mlp_in))
        return out


"""
###아직 완성되지 않음###

class MLPBERT4RecWithHF(nn.Module):
    def __init__(
        self,
        num_items,
        gen_img_emb,
        hidden_size=256,
        hidden_act="gelu",
        num_hidden_layers=3,
        num_attention_heads=4,
        max_len=30,
        dropout_prob=0.2,
        pos_emb=False,
        num_mlp_layers=2,
    ):

        self.num_items = num_items
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_len = max_len
        self.dropout_prob = dropout_prob
        self.pos_emb = pos_emb
        self.num_mlp_layer = num_mlp_layers
        self.gen_img_emb = gen_img_emb  # num_item X (3*512)

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
            pos_emb_shape = self.bert.embeddings.position_embeddings.weight.shape
            self.bert.embeddings.position_embeddings.weight.data = torch.zeros(pos_emb_shape)
            self.bert.embeddings.position_embeddings.requires_grad_ = False

        # init MLP
        self.MLP_modules = []
        in_size = self.hidden_size + self.gen_img_emb.shape[1]

        for _ in range(num_mlp_layers):
            self.MLP_modules.append(nn.Linear(in_size, in_size // 2))
            self.MLP_modules.append(nn.ReLU())
            in_size = in_size // 2

        self.MLP = nn.Sequential(*self.MLP_modules)
        self.out = nn.Linear(in_size, self.num_items + 1)

    def forward(self, tokens, labels):
        token_type_ids = torch.zeros_like(tokens)
        positions = torch.zeros_like(tokens)
        mask = (
            torch.BoolTensor(tokens > 0)
            .unsqueeze(1)
            .repeat(1, tokens.shape[1], 1)
            .unsqueeze(1)
            .to(self.device)
        )

        logits = self.bert(
            tokens,
            labels=labels,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            position_ids=positions,
        ).logits

        res = self.out(self.MLP())

        return res

"""

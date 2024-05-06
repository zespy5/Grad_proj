import random
from random import sample
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    def __init__(
        self,
        user_seq,
        sim_matrix,
        num_user,
        num_item,
        gen_img_emb: Optional[torch.Tensor] = None,
        idx_groups: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        neg_sampling: bool = False,
        neg_size: int = 50,
        neg_sample_size: int = 3,
        max_len: int = 30,
        mask_prob: float = 0.15,
        num_gen_img: int = 1,
        img_noise: bool = False,
        std: float = 1,
        mean: float = 0,
    ) -> None:
        self.user_seq = user_seq
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.sim_matrix = sim_matrix  # text emb sim for neg sampling
        self.num_gen_img = num_gen_img
        self.gen_img_emb = gen_img_emb
        self.idx_groups = idx_groups
        self.text_emb = text_emb
        self.neg_sampling = neg_sampling
        self.neg_size = neg_size
        self.neg_sample_size = neg_sample_size
        self.img_noise = img_noise
        self.std = std
        self.mean = mean

    def get_modal_emb(self, tokens, labels):
        item_ids = tokens.clone().detach()
        mask_index = torch.where(item_ids == self.num_item + 1)  # find mask
        item_ids[mask_index] = labels[mask_index]  # recover mask's original id
        item_ids -= 1
        modal_emb = torch.tensor([])

        if self.idx_groups is not None:
            item_id_group_sampler = np.vectorize(lambda x: sample(self.idx_groups[x], k=1)[0] if x != -1 else -1)
            item_ids = item_id_group_sampler(item_ids)

        if self.gen_img_emb is not None:
            img_idx = sample([0, 1, 2], k=self.num_gen_img)
            modal_emb = torch.flatten(self.gen_img_emb[item_ids][:, img_idx, :], start_dim=-2, end_dim=-1)
            if self.img_noise:
                modal_emb += torch.randn_like(modal_emb) * self.std + self.mean  # add noise to gen image

        if self.text_emb is not None:
            modal_emb = self.text_emb[item_ids]  # detail text embedding

        return modal_emb

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = torch.tensor(self.user_seq[index], dtype=torch.long) + 1
        tokens = []
        labels = []

        for s in seq:
            prob = random.random()
            if prob < self.mask_prob:  # train
                prob /= self.mask_prob
                # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                if prob < 0.8:
                    tokens.append(self.num_item + 1)
                elif prob < 0.9:
                    tokens.append(random.choice(range(1, self.num_item + 1)))
                else:
                    tokens.append(s)
                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # padding
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        tokens = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        seq = seq[-self.max_len :]
        seq = nn.ZeroPad1d((mask_len, 0))(seq)

        modal_emb = self.get_modal_emb(tokens, labels)

        return (tokens, modal_emb, labels, torch.tensor([], dtype=torch.long))


class BERTTestDataset(BERTDataset):
    def __init__(
        self,
        user_seq,
        sim_matrix,
        num_user,
        num_item,
        gen_img_emb: Optional[torch.Tensor] = None,
        idx_groups: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        neg_sampling: bool = False,
        neg_size: int = 50,
        neg_sample_size: int = 3,
        max_len: int = 30,
        num_gen_img: int = 1,
        img_noise: bool = False,
        std: float = 1,
        mean: float = 0,
    ) -> None:
        super().__init__(
            user_seq=user_seq,
            sim_matrix=sim_matrix,
            num_user=num_user,
            num_item=num_item,
            gen_img_emb=gen_img_emb,
            idx_groups=idx_groups,
            text_emb=text_emb,
            neg_sampling=neg_sampling,
            neg_size=neg_size,
            neg_sample_size=neg_sample_size,
            max_len=max_len,
            num_gen_img=num_gen_img,
            img_noise=img_noise,
            std=std,
            mean=mean,
        )

    def __getitem__(self, index):
        tokens = torch.tensor(self.user_seq[index], dtype=torch.long) + 1
        labels = [0 for _ in range(self.max_len)]

        labels[-1] = tokens[-1].item()  # target
        tokens[-1] = self.num_item + 1  # masking

        tokens = tokens[-self.max_len :]
        mask_len = self.max_len - len(tokens)
        tokens = torch.concat((torch.zeros(mask_len, dtype=torch.long), tokens), dim=0)

        labels = torch.tensor(labels, dtype=torch.long)

        modal_emb = self.get_modal_emb(tokens, labels)

        return index, tokens, modal_emb, labels, torch.tensor([], dtype=torch.long)


class BERTDatasetWithSampling(BERTDataset):
    def __init__(
        self,
        user_seq,
        sim_matrix,
        num_user,
        num_item,
        gen_img_emb: Optional[torch.Tensor] = None,
        idx_groups: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        neg_sampling: bool = False,
        neg_size: int = 50,
        neg_sample_size: int = 3,
        max_len: int = 30,
        mask_prob: float = 0.15,
        num_gen_img: int = 1,
        img_noise: bool = False,
        std: float = 1,
        mean: float = 0,
    ) -> None:
        super().__init__(
            user_seq=user_seq,
            sim_matrix=sim_matrix,
            num_user=num_user,
            num_item=num_item,
            gen_img_emb=gen_img_emb,
            idx_groups=idx_groups,
            text_emb=text_emb,
            neg_sampling=neg_sampling,
            neg_size=neg_size,
            neg_sample_size=neg_sample_size,
            max_len=max_len,
            mask_prob=mask_prob,
            num_gen_img=num_gen_img,
            img_noise=img_noise,
            std=std,
            mean=mean,
        )

    def neg_sampler(self, item, user_seq):
        candidate = np.setdiff1d(self.sim_matrix[item][: self.neg_size], user_seq, assume_unique=True)
        np.random.shuffle(candidate)
        return candidate[: self.neg_sample_size]  # negative sampling

    def __getitem__(self, index):
        seq = torch.tensor(self.user_seq[index], dtype=torch.long) + 1
        tokens = []
        labels = []
        negs = []

        for s in seq:
            prob = random.random()
            if prob < self.mask_prob:  # train
                prob /= self.mask_prob
                # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                if prob < 0.8:
                    tokens.append(self.num_item + 1)
                elif prob < 0.9:
                    tokens.append(random.choice(range(1, self.num_item + 1)))
                else:
                    tokens.append(s)
                labels.append(s)
                negs.append(self.neg_sampler(s - 1, seq - 1) + 1)
            else:
                tokens.append(s)
                labels.append(0)
                negs.append(np.zeros(self.neg_sample_size))

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        negs = negs[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # padding
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        negs = np.concatenate([np.zeros((mask_len, self.neg_sample_size)), negs], axis=0)

        tokens = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        negs = torch.tensor(negs, dtype=torch.long)

        seq = seq[-self.max_len :]
        seq = nn.ZeroPad1d((mask_len, 0))(seq)

        modal_emb = self.get_modal_emb(tokens, labels)

        return (
            tokens,
            modal_emb,
            labels,
            negs,
        )


class BERTTestDatasetWithSampling(BERTDatasetWithSampling):
    def __init__(
        self,
        user_seq,
        sim_matrix,
        num_user,
        num_item,
        gen_img_emb: Optional[torch.Tensor] = None,
        idx_groups: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        neg_sampling: bool = False,
        neg_size: int = 50,
        neg_sample_size: int = 3,
        max_len: int = 30,
        num_gen_img: int = 1,
        img_noise: bool = False,
        std: float = 1,
        mean: float = 0,
    ) -> None:
        super().__init__(
            user_seq=user_seq,
            sim_matrix=sim_matrix,
            num_user=num_user,
            num_item=num_item,
            gen_img_emb=gen_img_emb,
            idx_groups=idx_groups,
            text_emb=text_emb,
            neg_sampling=neg_sampling,
            neg_size=neg_size,
            neg_sample_size=neg_sample_size,
            max_len=max_len,
            num_gen_img=num_gen_img,
            img_noise=img_noise,
            std=std,
            mean=mean,
        )

    def __getitem__(self, index):
        tokens = torch.tensor(self.user_seq[index], dtype=torch.long) + 1
        labels = [0 for _ in range(self.max_len)]
        negs = np.zeros((self.max_len, self.neg_sample_size))

        labels[-1] = tokens[-1].item()  # target
        tokens[-1] = self.num_item + 1  # masking
        negs[-1] = self.neg_sampler(labels[-1] - 1, tokens - 1) + 1

        tokens = tokens[-self.max_len :]
        mask_len = self.max_len - len(tokens)
        tokens = torch.concat((torch.zeros(mask_len, dtype=torch.long), tokens), dim=0)

        labels = torch.tensor(labels, dtype=torch.long)
        negs = torch.tensor(negs, dtype=torch.long)

        modal_emb = self.get_modal_emb(tokens, labels)

        return index, tokens, modal_emb, labels, negs

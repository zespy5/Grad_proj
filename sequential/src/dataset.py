import random

import numpy as np
import torch
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    def __init__(self, user_seq, sim_matrix, num_user, num_item,
                 neg_size: int = 50, neg_sample_size : int = 3, #negative sampling
                 max_len: int = 30, mask_prob: float = 0.15) -> None:
        self.user_seq = user_seq
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.sim_matrix = sim_matrix
        self.neg_size = neg_size                 #negative sampling
        self.neg_sample_size = neg_sample_size   #negative sampling
        
    def sampler(self, item, user_seq):
        candidate = np.setdiff1d(self.sim_matrix[item][:3000], user_seq, assume_unique=True)
        return candidate[:self.neg_size+1] #negative sampling

    def __len__(self):
        return self.num_user

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
                negs.append(
                    self.sampler(
                        s - 1,
                        seq - 1,
                    )[np.random.randint(0,51, self.neg_sample_size)] #3개 뽑기.
                    + 1
                )
            else:  # not train
                tokens.append(s)
                labels.append(0)
                negs.append(np.zeros(self.neg_sample_size))#3개 뽑기.

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        negs = negs[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # padding
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        negs = np.concatenate([np.zeros((mask_len, self.neg_sample_size)), negs], axis=0)
        #3개 뽑기.

        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(negs, dtype=torch.long),
        )


class BERTTestDataset(Dataset):
    def __init__(self, user_seq, sim_matrix, num_user, num_item,
                 neg_size: int = 50, neg_sample_size : int = 3, #negative sampling
                 max_len: int = 30) -> None:
        self.user_seq = user_seq
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        self.sim_matrix = sim_matrix
        self.neg_size = neg_size                     #negative sampling
        self.neg_sample_size = neg_sample_size       #negative sampling

    def sampler(self, item, user_seq):
        candidate = np.setdiff1d(self.sim_matrix[item][:3000], user_seq, assume_unique=True)
        return candidate[:self.neg_size+1] #negative sampling

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        tokens = torch.tensor(self.user_seq[index], dtype=torch.long) + 1

        labels = [0 for _ in range(self.max_len)]
        negs = np.zeros((self.max_len, self.neg_sample_size)) #3개 뽑기
        labels[-1] = tokens[-1].item()  # target
        negs[-1] = self.sampler(labels[-1] - 1, tokens - 1)[np.random.randint(0,51, self.neg_sample_size)] + 1 #3개 뽑기
        tokens[-1] = self.num_item + 1  # masking

        tokens = tokens[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # padding
        tokens = torch.concat((torch.zeros(mask_len, dtype=torch.long), tokens), dim=0)

        return index, tokens, torch.tensor(labels, dtype=torch.long), torch.tensor(negs, dtype=torch.long)

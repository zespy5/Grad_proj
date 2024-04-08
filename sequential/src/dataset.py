import random

import torch
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    def __init__(self, user_seq, num_user, num_item, max_len: int = 30, mask_prob: float = 0.15) -> None:
        self.user_seq = user_seq
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = torch.tensor(self.user_seq[index], dtype=torch.long) + 1
        tokens = []
        labels = []
        negative = []

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
                # negative sampling data를 이용하여 negative에 update
            else:  # not train
                tokens.append(s)
                labels.append(0)
                negative.append(0)

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        negative = negative[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # padding
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        negative = [0] * mask_len + negative
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long), torch.tensor(negative, dtype=torch.long)


class BERTTestDataset(Dataset):
    def __init__(self, user_seq, num_user, num_item, max_len: int = 30) -> None:
        self.user_seq = user_seq
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        tokens = torch.tensor(self.user_seq[index], dtype=torch.long)
        tokens = tokens + 1  # cos padding : 0

        labels = [0 for _ in range(self.max_len)]
        labels[-1] = tokens[-1].item()  # target
        tokens[-1] = self.num_item + 1  # masking

        tokens = tokens[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # padding
        tokens = torch.concat((torch.zeros(mask_len, dtype=torch.long), tokens), dim=0)

        return index, tokens, torch.tensor(labels, dtype=torch.long)

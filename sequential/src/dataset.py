import random

import torch
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    def __init__(self, user_train, num_user, num_item, max_len: int = 30, mask_prob: float = 0.15) -> None:
        self.user_train = user_train
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = self.user_train[index]
        seq = seq + 1  # cos padding : 0,
        tokens = []
        labels = []

        for s in seq:
            prob = random.random()
            if prob < self.mask_prob:  # train
                prob /= self.mask_prob

                if prob < 0.8:
                    # masking
                    tokens.append(
                        self.num_item + 1
                    )  # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                elif prob < 0.9:
                    tokens.append(random.choice(range(1, self.num_item + 1)))
                else:
                    tokens.append(s)
                labels.append(s)
            else:  # not train
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # padding
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.tensor(tokens), torch.tensor(labels)


class BERTTestDataset(Dataset):
    def __init__(self, user_seq, test_seq, num_user, num_item, max_len: int = 30) -> None:
        self.user_seq = user_seq
        self.test_seq = test_seq
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = self.user_seq[index]
        target = self.test_seq[index].item() + 1
        seq = seq + 1  # cos padding : 0

        tokens = list(seq)
        tokens.append(self.num_item + 1)
        labels = [0 for _ in range(self.max_len)]
        labels[-1] = target
        
        tokens = tokens[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # padding
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return index, torch.tensor(tokens), torch.tensor(labels)

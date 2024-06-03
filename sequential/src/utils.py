import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_timestamp(date_format: str = "%m%d%H%M%S") -> str:
    timestamp = datetime.now()
    return timestamp.strftime(date_format)


def get_config(path):
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def load_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def recall_at_k(k, true, pred):
    true = true.data.cpu().numpy()
    pred = (pred[:k]).data.cpu().numpy()
    return len(np.intersect1d(true, pred)) / len(true)


def ndcg_at_k(k, true, pred):
    true = true.data.cpu().numpy()
    pred = (pred[:k]).data.cpu().numpy()

    log2i = np.log2(np.arange(2, k + 2))
    dcg = np.sum(np.isin(pred, true) / log2i)  # rel_i = 1
    idcg = np.sum((1 / log2i)[: min(len(true), k)])

    return dcg / idcg


def simple_recall_at_k(k, rank):
    if rank < k:
        return 1
    return 0


def simple_recall_at_k_batch(k, rank):
    return (rank < k).int()


def simple_ndcg_at_k(k, rank):
    if rank < k:
        return 1 / torch.log2(rank + 2)
    return 0


def simple_ndcg_at_k_batch(k, rank):
    return torch.where(rank < k, 1 / torch.log2(rank + 2), torch.tensor(0.0))


def mk_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

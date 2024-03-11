import numpy as np
import torch
from src.model import BERT4Rec, BERT4RecWithHF, MLPBERT4Rec
from src.utils import ndcg_at_k, recall_at_k, simple_ndcg_at_k, simple_recall_at_k
from tqdm import tqdm


def train(model, optimizer, scheduler, dataloader, criterion, device):
    model.train()
    total_loss = 0

    for tokens, labels in tqdm(dataloader):
        tokens = tokens.to(device)
        labels = labels.to(device)

        if isinstance(model, (MLPBERT4Rec)):
            logits = model(tokens, labels)
        if isinstance(model, (BERT4Rec, BERT4RecWithHF)):
            logits = model(tokens)

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = criterion(logits, labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # scheduler.step()
    return total_loss / len(dataloader)


def eval(
    model,
    mode,
    dataloader,
    criterion,
    num_items,
    used_items_each_user,
    device,
):
    model.eval()
    metrics = {"R10": [], "R20": [], "R40": [], "N10": [], "N20": [], "N40": []}
    total_loss = 0
    all_items = np.arange(1, num_items + 1)

    with torch.no_grad():
        for tokens, labels in tqdm(dataloader):
            tokens = tokens.to(device)
            labels = labels.to(device)

            if isinstance(model, (MLPBERT4Rec)):
                logits = model(tokens, labels)
            if isinstance(model, (BERT4Rec, BERT4RecWithHF)):
                logits = model(tokens)

            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            for i, label in enumerate(labels):
                user_res = -logits[i, -1, 1:] # without zero(padding)
                target = label[-1]
                item_rank = user_res.argsort().argsort() # get rank(start with 0~) of all items(id:idx = 1:0)
                
                for k in [10, 20, 40]:
                    metrics["R" + str(k)].append(simple_recall_at_k(k, item_rank[target-1]+1))
                    metrics["N" + str(k)].append(simple_ndcg_at_k(k, item_rank[target-1]+1))

        for k in [10, 20, 40]:
            metrics["R" + str(k)] = sum(metrics["R" + str(k)]) / len(metrics["R" + str(k)])
            metrics["N" + str(k)] = sum(metrics["N" + str(k)]) / len(metrics["N" + str(k)])

        if mode == "valid":
            return total_loss / len(dataloader), metrics
    return metrics

import numpy as np
import torch
from src.model import BERT4Rec, BERT4RecWithHF, MLPBERT4Rec
from src.utils import ndcg_at_k, recall_at_k
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
        for i, (users, tokens, labels) in enumerate(tqdm(dataloader)):
            tokens = tokens.to(device)
            labels = labels.to(device)
            users = users.to(device)

            if isinstance(model, (MLPBERT4Rec)):
                logits = model(tokens, labels)
            if isinstance(model, (BERT4Rec, BERT4RecWithHF)):
                logits = model(tokens)

            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            for u in users:
                u = u.item()
                used_item = np.array(used_items_each_user[u]) + 1
                candidate_items = torch.tensor(np.setdiff1d(all_items, used_item))

                u -= i * dataloader.batch_size
                user_res = -logits[u, -1, candidate_items]
                sorted_item = user_res.argsort()

                for k in [10, 20, 40]:
                    metrics["R" + str(k)].append(recall_at_k(k, labels[-1], sorted_item))
                    metrics["N" + str(k)].append(ndcg_at_k(k, labels[-1], sorted_item))

        for k in [10, 20, 40]:
            metrics["R" + str(k)] = round(np.asarray(metrics["R" + str(k)]).mean(), 10)
            metrics["N" + str(k)] = round(np.asarray(metrics["N" + str(k)]).mean(), 10)

        if mode == "valid":
            return total_loss / len(dataloader), metrics
    return metrics

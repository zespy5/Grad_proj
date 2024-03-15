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
    model, mode, categoty_clue, dataloader, criterion, train_data, item_prod_type, items_by_prod_type, device
):
    model.eval()
    metrics = {"R10": [], "R20": [], "R40": [], "N10": [], "N20": [], "N40": []}
    total_loss = 0
    pred_list = []
    all_items = np.arange(0, model.num_item)

    with torch.no_grad():
        for idx, (users, tokens, labels) in enumerate(tqdm(dataloader)):
            tokens = tokens.to(device)
            labels = labels.to(device)
            users = users.to(device)
            pred_list = []

            if isinstance(model, (MLPBERT4Rec)):
                logits = model(tokens, labels)
            if isinstance(model, (BERT4Rec, BERT4RecWithHF)):
                logits = model(tokens)

            if mode == "valid":
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()

            for user in users:
                user = user.item()
                used_items = torch.tensor(train_data[user]).unique()
                u = user - idx * dataloader.batch_size

                target = labels[u, -1]
                user_res = -logits[u, -1, 1:]  # without zero(padding), itme start with 0
                user_res[used_items] = (user_res.max()) + 1  # for remove used items

                if categoty_clue and (not model.num_gen_img):
                    # for remove other category items
                    other_prod_type = torch.tensor(
                        np.setdiff1d(all_items, items_by_prod_type[item_prod_type[target - 1]])
                    ).to(device)
                    user_res[other_prod_type] = user_res.max()

                # sorted item id e.g. [3452(1st), 7729(2nd), ... ]
                item_rank = user_res.argsort()

                if mode == "test":
                    pred_list.append(item_rank[:40].tolist() + [target])

                # rank of items e.g. index: item_id(0~), item_rank[0] : rank of item_id 0
                item_rank = item_rank.argsort()

                for k in [10, 20, 40]:
                    metrics["R" + str(k)].append(simple_recall_at_k(k, item_rank[target - 1]))
                    metrics["N" + str(k)].append(simple_ndcg_at_k(k, item_rank[target - 1]))

        for k in [10, 20, 40]:
            metrics["R" + str(k)] = sum(metrics["R" + str(k)]) / len(metrics["R" + str(k)])
            metrics["N" + str(k)] = sum(metrics["N" + str(k)]) / len(metrics["N" + str(k)])

        if mode == "valid":
            return total_loss / len(dataloader), metrics
    return pred_list, metrics

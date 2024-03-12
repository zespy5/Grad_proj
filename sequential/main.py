import os

import dotenv
import torch
import torch.nn as nn
import wandb
from huggingface_hub import snapshot_download
from src.dataset import BERTDataset, BERTTestDataset
from src.model import BERT4Rec, BERT4RecWithHF, MLPBERT4Rec
from src.train import eval, train
from src.utils import get_timestamp, load_pickle, mk_dir, seed_everything
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader


def main():
    ############# SETTING #############
    seed_everything()
    mk_dir("./model")
    mk_dir("./data")
    timestamp = get_timestamp()
    name = f"work-{timestamp}"

    ############ SET HYPER PARAMS #############
    ## MODEL ##
    model_name = "BERT4Rec"
    hidden_size = 512
    num_attention_heads = 4
    num_hidden_layers = 3
    max_len = 50
    dropout_prob = 0.25
    num_mlp_layers = 2
    pos_emb = False
    hidden_act = "gelu"
    num_gen_img = 1
    ## TRAIN ##
    lr = 0.0001
    epoch = 80
    batch_size = 256
    weight_decay = 0.01
    ## DATA ##
    data_local = False
    dataset = "bert"
    data_version = "846eac5c742885ce41684da01af5382d54eb1b19"
    ## ETC ##
    n_cuda = "0"

    ############ WANDB INIT #############
    print("--------------- Wandb SETTING ---------------")
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project="sequential",
        name=name,
    )
    wandb.log(
        {
            "model_name": model_name,
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "num_gen_img": num_gen_img,
            "max_len": max_len,
            "dropout_prob": dropout_prob,
            "num_mlp_layers": num_mlp_layers,
            "pos_emb": pos_emb,
            "hidden_act": hidden_act,
            "lr": lr,
            "epoch": epoch,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "data_version": data_version,
        },
    )

    ############# LOAD DATASET #############
    # when calling data from huggingface Hub
    if not data_local:
        path = (
            snapshot_download(
                repo_id="SLKpnu/HandM_Dataset",
                repo_type="dataset",
                cache_dir="./data",
                revision=data_version,
            )
            + "/"
            + dataset
        )
    else:
        path = f"./data/{dataset}"

    print("-------------LOAD DATA-------------")
    used_items_each_user = load_pickle(f"{path}/pos_items_each_user_small.pkl")
    train_dataset = torch.load(f"{path}/train_dataset_small.pt")
    valid_dataset = torch.load(f"{path}/valid_dataset_small.pt")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4)

    ############# SETTING FOR TRAIN #############
    num_user = train_dataset.num_user
    num_item = train_dataset.num_item
    device = f"cuda:{n_cuda}" if torch.cuda.is_available() else "cpu"

    ## MODEL INIT ##
    if model_name == "MLPBERT4Rec":
        gen_img_emb = torch.load(f"{path}/gen_img_emb_sep.pth")  # dim : ((num_item) * (512*3))
        model = MLPBERT4Rec(
            num_item,
            gen_img_emb,
            hidden_size,
            num_attention_heads,
            num_hidden_layers,
            hidden_act,
            num_gen_img,
            max_len,
            dropout_prob,
            pos_emb,
            num_mlp_layers,
            device,
        ).to(device)

    if model_name == "BERT4Rec":
        model = BERT4Rec(
            num_item,
            hidden_size,
            num_attention_heads,
            num_hidden_layers,
            hidden_act,
            max_len,
            dropout_prob,
            pos_emb,
            device,
        ).to(device)

    if model_name == "BERT4RecWithHF":
        model = BERT4RecWithHF(
            num_item,
            hidden_size,
            num_attention_heads,
            num_hidden_layers,
            hidden_act,
            max_len,
            dropout_prob,
            pos_emb,
            device,
        ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.85**epoch)

    ############# TRAIN AND EVAL #############
    for i in range(epoch):
        print("-------------TRAIN-------------")
        train_loss = train(model, optimizer, scheduler, train_dataloader, criterion, device)
        print(f'EPOCH : {i+1} | TRAIN LOSS : {train_loss} | LR : {optimizer.param_groups[0]["lr"]}')
        wandb.log({"loss": train_loss, "epoch": i + 1, "lr": optimizer.param_groups[0]["lr"]})

        if i % 5 == 0:
            print("-------------VALID-------------")
            valid_loss, valid_metrics = eval(
                model,
                "valid",
                valid_dataloader,
                criterion,
                num_item,
                None,
                device,
            )
            print(f"EPOCH : {i+1} | VALID LOSS : {valid_loss}")
            print(
                f'R10 : {valid_metrics["R10"]} | R20 : {valid_metrics["R20"]} | R40 : {valid_metrics["R40"]} | N10 : {valid_metrics["N10"]} | N20 : {valid_metrics["N20"]} | N40 : {valid_metrics["N40"]}'
            )
            wandb.log(
                {
                    "epoch": i + 1,
                    "valid_loss": valid_loss,
                    "valid_R10": valid_metrics["R10"],
                    "valid_R20": valid_metrics["R20"],
                    "valid_R40": valid_metrics["R40"],
                    "valid_N10": valid_metrics["N10"],
                    "valid_N20": valid_metrics["N20"],
                    "valid_N40": valid_metrics["N40"],
                }
            )

    ############ WANDB FINISH #############
    wandb.finish()


if __name__ == "__main__":
    main()

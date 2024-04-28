import os
import shutil

import dotenv
import torch
import torch.nn as nn
import wandb
from huggingface_hub import snapshot_download

# from src.model import BERT4Rec, BERT4RecWithHF, BPRLoss, MLPBERT4Rec, MLPRec
from src import models
from src.dataset import BERTDataset, BERTTestDataset
from src.models import BPRLoss
from src.train import eval, train
from src.utils import get_config, get_timestamp, load_json, mk_dir, seed_everything
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader


def main():
    ############# SETTING #############
    setting_yaml_path = "./settings/base.yaml"
    timestamp = get_timestamp()

    seed_everything()
    mk_dir("./model")
    mk_dir("./data")
    mk_dir(f"./model/{timestamp}")

    settings = get_config(setting_yaml_path)

    model_name: str = settings["model_name"]
    model_args: dict = settings["model_arguments"]
    name = f"work-{timestamp}_" + settings["experiment_name"]

    shutil.copy(setting_yaml_path, f"./model/{timestamp}/setting.yaml")

    ############ SET HYPER PARAMS #############
    ## TRAIN ##
    lr = settings["lr"]
    epoch = settings["epoch"]
    batch_size = settings["batch_size"]
    weight_decay = settings["weight_decay"]
    num_workers = settings["num_workers"]
    loss = settings["loss"]

    ## DATA ##
    data_local = settings["data_local"]
    data_repo = settings["data_repo"]
    dataset = settings["dataset"]
    data_version = settings["data_version"]

    ## ETC ##
    n_cuda = settings["n_cuda"]

    ############ WANDB INIT #############
    print("--------------- Wandb SETTING ---------------")
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT"),
        name=name,
        mode=os.environ.get("WANDB_MODE"),
    )
    wandb.log(settings)

    ############# LOAD DATASET #############
    # when calling data from huggingface Hub
    if not data_local:
        path = (
            snapshot_download(
                repo_id=f"SLKpnu/{data_repo}",
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
    metadata = load_json(f"{path}/metadata.json")
    item_prod_type = torch.load(f"{path}/item_with_prod_type_idx.pt")  # tensor(prod_type_idx), index : item_idx
    items_by_prod_type = torch.load(f"{path}/items_by_prod_type_idx.pt")  # {prod_type_idx : tensor(item_ids)}
    train_data = torch.load(f"{path}/train_data.pt")
    valid_data = torch.load(f"{path}/valid_data.pt")
    test_data = torch.load(f"{path}/test_data.pt")
    id_group_dict = torch.load(f"{path}/id_group_dict.pt") if settings["model_arguments"]["description_group"] else None
    sim_matrix = torch.load(f"{path}/sim_matrix_sorted.pt")

    num_user = metadata["num of user"]
    num_item = metadata["num of item"]
    num_cat = len(items_by_prod_type)

    gen_img_emb = torch.load(f"{path}/gen_img_emb.pt")  # dim : ((num_item)*512*3)

    text_emb = None
    if model_args["cat_text"]:
        text_emb = torch.load(f"{path}/cat_text_embeddings.pt")  # dim : 82*512
    elif model_args["detail_text"]:
        text_emb = torch.load(f"{path}/detail_text_embeddings.pt")  # dim : 60233*512

    train_dataset = BERTDataset(
        user_seq=train_data,
        sim_matrix=sim_matrix,
        num_user=num_user,
        num_item=num_item,
        num_cat=num_cat,
        gen_img_emb=gen_img_emb,
        item_prod_type=item_prod_type,
        idx_groups=id_group_dict,
        text_emb=text_emb,
        neg_size=settings["neg_size"],
        neg_sample_size=settings["neg_sample_size"],
        max_len=model_args["max_len"],
        mask_prob=model_args["mask_prob"],
        num_gen_img=model_args["num_gen_img"],
        img_noise=model_args["img_noise"],
        std=model_args["std"],
        mean=model_args["mean"],
        mlp_cat=model_args["mlp_cat"],
    )
    valid_dataset = BERTTestDataset(
        valid_data,
        sim_matrix,
        num_user,
        num_item,
        num_cat,
        gen_img_emb,
        item_prod_type,
        id_group_dict,
        text_emb,
        settings["neg_size"],
        settings["neg_sample_size"],
        model_args["max_len"],
        model_args["num_gen_img"],
        model_args["img_noise"],
        model_args["std"],
        model_args["mean"],
        model_args["mlp_cat"],
    )
    test_dataset = BERTTestDataset(
        test_data,
        sim_matrix,
        num_user,
        num_item,
        num_cat,
        gen_img_emb,
        item_prod_type,
        id_group_dict,
        text_emb,
        settings["neg_size"],
        settings["neg_sample_size"],
        model_args["max_len"],
        model_args["num_gen_img"],
        model_args["img_noise"],
        model_args["std"],
        model_args["mean"],
        model_args["mlp_cat"],
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    ############# SETTING FOR TRAIN #############
    device = f"cuda:{n_cuda}" if torch.cuda.is_available() else "cpu"

    ## MODEL INIT ##
    model_class_ = getattr(models, model_name)

    if model_name in ("MLPBERT4Rec", "MLPRec"):
        model_args["gen_img_emb"] = gen_img_emb
        model_args["text_emb"] = text_emb

    model = model_class_(
        **model_args,
        num_cat=num_cat,
        num_item=num_item,
        item_prod_type=item_prod_type,
        idx_groups=id_group_dict,
        device=device,
    ).to(device)

    if loss == "BPR":
        criterion = BPRLoss()
    if loss == "CE":
        criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.85**epoch)

    ############# TRAIN AND EVAL #############
    for i in range(epoch):
        print("-------------TRAIN-------------")
        train_loss = train(model, optimizer, scheduler, train_dataloader, criterion, device)
        print(f'EPOCH : {i+1} | TRAIN LOSS : {train_loss} | LR : {optimizer.param_groups[0]["lr"]}')
        wandb.log({"loss": train_loss, "epoch": i + 1, "lr": optimizer.param_groups[0]["lr"]})

        if i % settings["valid_step"] == 0:
            print("-------------VALID-------------")
            (
                valid_loss,
                valid_metrics,
            ) = eval(
                model=model,
                mode="valid",
                dataloader=valid_dataloader,
                criterion=criterion,
                train_data=train_data,
                device=device,
            )
            print(f"EPOCH : {i+1} | VALID LOSS : {valid_loss}")
            print(
                (
                    f'R10 : {valid_metrics["R10"]} | R20 : {valid_metrics["R20"]} | R40 : {valid_metrics["R40"]} | '
                    f'N10 : {valid_metrics["N10"]} | N20 : {valid_metrics["N20"]} | N40 : {valid_metrics["N40"]}'
                )
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
            torch.save(model.state_dict(), f"./model/{timestamp}/model_val_{valid_loss}.pt")

    print("-------------EVAL-------------")
    pred_list, test_metrics = eval(
        model=model,
        mode="test",
        dataloader=test_dataloader,
        criterion=criterion,
        train_data=train_data,
        device=device,
    )
    print(
        (
            f'R10 : {test_metrics["R10"]} | R20 : {test_metrics["R20"]} | R40 : {test_metrics["R40"]} | '
            f'N10 : {test_metrics["N10"]} | N20 : {test_metrics["N20"]} | N40 : {test_metrics["N40"]}'
        )
    )
    wandb.log(test_metrics)
    torch.save(pred_list, f"./model/{timestamp}/prediction.pt")
    wandb.save(f"./model/{timestamp}/prediction.pt")

    ############ WANDB FINISH #############
    wandb.finish()


if __name__ == "__main__":
    main()

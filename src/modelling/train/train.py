import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from modelling.load.load import MyDataset
from modelling.models.VAE import VAE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser(description="Tune cVAE")


# python src/modelling/train/VAE/VAE_train.py --data_path "data/processed_data" --feature_type "rsfmri" --project_title "VAE_rsfmri_final_train" --batch_size 256 --learning_rate 0.005 --latent_dim 13 --hidden_dim "30"


def int_parse_list(arg_value):
    return [int(x) for x in arg_value.split("-")]


parser.add_argument(
    "--data_path",
    type=str,
    default="",
    help="Path to processed data",
)
parser.add_argument(
    "--feature_type",
    type=str,
    default="",
    help="Brain features of interest, options: 'cortical_thickness, cortical_volume, rsfmri'",
)
parser.add_argument(
    "--project_title",
    type=str,
    default="Tune cVAE",
    help="Title of the project for weights and biases",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size for training, e.g., '64-128-256', which will be parsed into [64, 128, 256]",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.01,
    help="Learning rate for the optimizer, e.g., '0.005-0.001-0.0005-0.0001', which will be parsed into [0.005, 0.001, 0.0005, 0.0001]",
)
parser.add_argument(
    "--latent_dim",
    type=int,
    default=5,
    help="Dimensions of the latent space, e.g., '10-11-12', which will be parsed into [10, 11, 12]",
)
parser.add_argument(
    "--hidden_dim",
    type=int_parse_list,
    default=[30],
    help="Pass dimensions of multiple hidden layers, use ';' to separate layers and '-' to separate dimensions within layers, e.g., '30-30;40-40-40'",
)

args = parser.parse_args()
processed_data_path = Path(args.data_path)
feature_type = args.feature_type

feature_sets_map = {
    "cortical_thickness": "t1w_cortical_thickness_rois",
    "cortical_volume": "t1w_cortical_volume_rois",
    "rsfmri": "gordon_net_subcor_limbic_no_dup",
}


brain_features_of_interest_path = Path(
    processed_data_path,
    "brain_features_of_interest.json",
)

with open(brain_features_of_interest_path, "r") as f:
    brain_features_of_interest = json.load(f)

FEATURES = brain_features_of_interest[feature_sets_map[feature_type]]


TRAIN_DATA_PATH = Path(
    processed_data_path,
    "all_brain_features_resid.csv",
)

CHECKPOINT_PATH = Path(
    "checkpoints",
    feature_type,
)

if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.mkdir(parents=True)

data_splits_path = Path(
    processed_data_path,
    "data_splits_with_clinical_val.json",
)

with open(data_splits_path, "r") as f:
    data_splits = json.load(f)

if "cortical" in feature_type:
    modality_data_split = data_splits["structural"]

else:
    modality_data_split = data_splits["functional"]


TRAIN_SUBS = modality_data_split["train"]
VAL_SUBS = modality_data_split["val"]


def build_model(config, input_dim):
    # Set random seed for CPU
    torch.manual_seed(123)

    # Set random seed for CUDA (GPU) if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    # Initialize the model
    model = VAE(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        learning_rate=config["learning_rate"],
        non_linear=True,
    ).to(DEVICE)

    # if isinstance(model, nn.Module):
    #     first_layer_params = list(model.parameters())[0]
    # print("Parameter values of the first layer:", first_layer_params)

    return model


def one_hot_encode_covariate(
    data,
    covariate,
    subjects,
):
    """Return one hot encoded covariate for the given subjects as required by the cVAE model."""
    covariate_data = data.loc[
        subjects,
        [covariate],
    ]

    covariate_data[covariate] = pd.Categorical(covariate_data[covariate])

    category_codes = covariate_data[covariate].cat.codes

    num_categories = len(covariate_data[covariate].cat.categories)

    one_hot_encoded_covariate = np.eye(num_categories)[category_codes]

    return one_hot_encoded_covariate


def validate(model, val_loader, device):
    model.eval()
    total_val_loss = 0.0
    total_recon = 0.0
    total_kl_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            fwd_rtn = model.forward(batch)
            val_loss = model.loss_function(batch, fwd_rtn)
            batch_val_loss = val_loss["total"].item()
            total_val_loss += batch_val_loss
            total_recon += val_loss["ll"].item()
            total_kl_loss += val_loss["kl"].item()

    mean_val_loss = total_val_loss / len(val_loader)
    mean_recon = total_recon / len(val_loader)
    mean_kl_loss = total_kl_loss / len(val_loader)

    wandb.log({"val_total_loss": mean_val_loss})
    wandb.log({"val_recon": mean_recon})
    wandb.log({"val_kl_loss": mean_kl_loss})

    return mean_val_loss


def train(
    config,
    model,
    train_loader,
    val_loader,
    tolerance=50,
):

    model.to(DEVICE)

    best_val_loss = float("inf")

    model_artifact = wandb.Artifact(wandb.run.name, type="model")

    epochs_no_improve = 0

    for epoch in range(1, config.epochs + 1):

        total_loss = 0.0
        total_recon = 0.0
        kl_loss = 0.0

        model.train()

        for batch_idx, batch in enumerate(train_loader):
            data_curr = batch.to(DEVICE)
            fwd_rtn = model.forward(data_curr)
            loss = model.loss_function(data_curr, fwd_rtn)
            model.optimizer.zero_grad()
            loss["total"].backward()
            model.optimizer.step()

            total_loss += loss["total"].item()
            total_recon += loss["ll"].item()
            kl_loss += loss["kl"].item()

        val_loss = validate(model, val_loader, DEVICE)

        if val_loss < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = val_loss

            print("New best val loss:", val_loss)
            print("at epoch:", epoch)

            best_model = model.state_dict()

            print("Improved at epoch:", epoch)

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= tolerance:
            print("Early stopping at epoch:", epoch)

            torch.save(
                best_model,
                Path(CHECKPOINT_PATH, "VAE_model_weights.pt"),
            )

            model_artifact.add_file(
                Path(CHECKPOINT_PATH, "VAE_model_weights.pt"),
            )

            wandb.run.log_artifact(model_artifact)

            break

        wandb.log(
            {
                "Epoch_total_loss": total_loss / len(train_loader),
                "Epoch_recon_loss": total_recon / len(train_loader),
                "Epoch_kl_loss": kl_loss / len(train_loader),
            }
        )


# Remove subjects with high entropy
# high_entropy_subs_path = Path(
#     "data",
#     "LCA",
#     "subjects_with_high_entropy.csv",
# )

# high_entropy_subs = pd.read_csv(
#     high_entropy_subs_path,
#     low_memory=False,
# )["subject"].tolist()

# TRAIN_SUBS = [sub for sub in TRAIN_SUBS if sub not in high_entropy_subs]

# VAL_SUBS = [sub for sub in VAL_SUBS if sub not in high_entropy_subs]


def main(config):

    data = pd.read_csv(
        Path(TRAIN_DATA_PATH),
        index_col=0,
        low_memory=False,
    )

    train_dataset = data.loc[
        TRAIN_SUBS,
        FEATURES,
    ].to_numpy()

    val_dataset = data.loc[
        VAL_SUBS,
        FEATURES,
    ].to_numpy()

    scaler = StandardScaler()

    train_data = scaler.fit_transform(train_dataset)

    val_data = scaler.transform(val_dataset)

    train_loader = DataLoader(
        MyDataset(train_data),
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        MyDataset(val_data),
        batch_size=config.batch_size,
        shuffle=False,
    )

    input_dim = train_data.shape[1]

    model = build_model(
        config,
        input_dim,
    )

    train(
        config,
        model,
        train_loader,
        val_loader,
    )


if __name__ == "__main__":

    project_title = args.project_title
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    latent_dim = args.latent_dim
    hidden_dim = args.hidden_dim

    with wandb.init(
        project=f"VAE_{feature_type}_final_train",
        config={
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "learning_rate": learning_rate,
            "epochs": 5000,
        },
    ):
        main(wandb.config)

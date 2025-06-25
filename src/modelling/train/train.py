import json
import logging
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ..load.load import MyDataset
from ..models.VAE import VAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_train_data(version_name="test"):
    """Setup data and paths for training."""
    logging.info("-----------------------")
    logging.info("Training VAE model")

    data_store_path = Path(
        "/",
        "Volumes",
        "GenScotDepression",
    )

    if data_store_path.exists():
        logging.info("Mounted data store path: %s", data_store_path)

    analysis_root_path = Path(
        data_store_path,
        "users",
        "Eric",
        "nm",
    )

    processed_data_path = Path(
        analysis_root_path,
        version_name,
        "processed_data",
    )

    checkpoint_path = Path(
        analysis_root_path,
        version_name,
        "checkpoints",
    )

    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    imaging_features_path = Path(
        processed_data_path,
        "mri_all_features_post_deconfound.csv",
    )

    data = pd.read_csv(
        imaging_features_path,
        index_col=0,
        low_memory=False,
    )

    brain_features_of_interest_path = Path(
        processed_data_path,
        "features_of_interest.json",
    )

    with open(brain_features_of_interest_path, "r") as f:
        features_of_interest = json.load(f)

    data_splits_path = Path(
        processed_data_path,
        "imaging_data_splits.json",
    )

    with open(data_splits_path, "r") as f:
        data_splits = json.load(f)

    train_subs = data_splits["train"]
    val_subs = data_splits["val"]
    
    return data, features_of_interest, train_subs, val_subs, checkpoint_path


def build_model(
    config,
    input_dim,
):
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

    return model


def validate(
    model,
    val_loader,
    device,
):
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

    logging.info(
        "Validation loss: %.4f, Recon loss: %.4f, KL loss: %.4f",
        mean_val_loss,
        mean_recon,
        mean_kl_loss,
    )

    return mean_val_loss


def train_model(
    config,
    model,
    train_loader,
    val_loader,
    checkpoint_path,
    tolerance=50,
):
    model.to(DEVICE)

    best_val_loss = float("inf")

    epochs_no_improve = 0

    for epoch in range(1, config["epochs"] + 1):
        total_loss = 0.0
        total_recon = 0.0
        kl_loss = 0.0

        model.train()

        for _, batch in enumerate(train_loader):
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

            logging.info(
                "Epoch %d: New best val score: %.4f",
                epoch,
                best_val_loss,
            )

            best_model = model.state_dict()

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= tolerance:
            logging.info(
                "Early stopping triggered after %d epochs without improvement.",
                epochs_no_improve,
            )

            modality = config["modality"]

            torch.save(
                best_model,
                Path(checkpoint_path, f"VAE_model_weights_{modality}.pt"),
            )

            break

        logging.info(
            "Epoch %d: Train total loss: %.4f, Recon loss: %.4f, KL loss: %.4f",
            epoch,
            total_loss / len(train_loader),
            total_recon / len(train_loader),
            kl_loss / len(train_loader),
        )


def train(config, version_name="test"):
    data, features_of_interest, train_subs, val_subs, checkpoint_path = setup_train_data(version_name)
    
    modality = config["modality"]

    logging.info("Training VAE model for modality: %s", modality)
    logging.info("Using configuration: %s", config)

    features = features_of_interest[modality]

    train_dataset = data.loc[
        train_subs,
        features,
    ].to_numpy()

    val_dataset = data.loc[
        val_subs,
        features,
    ].to_numpy()

    scaler = StandardScaler()

    train_data = scaler.fit_transform(train_dataset)
    val_data = scaler.transform(val_dataset)

    train_loader = DataLoader(
        MyDataset(train_data),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        MyDataset(val_data),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    input_dim = train_data.shape[1]

    model = build_model(
        config,
        input_dim,
    )

    train_model(
        config,
        model,
        train_loader,
        val_loader,
        checkpoint_path,
    )


if __name__ == "__main__":
    config = {
        "modality": "cortical_thickness",
        "hidden_dim": [10],
        "latent_dim": 5,
        "learning_rate": 0.01,
        "batch_size": 64,
        "epochs": 10,
    }

    train(config)

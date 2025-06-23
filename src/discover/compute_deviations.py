import json
import logging
from pathlib import Path

import pandas as pd
import torch

from src.discover.utils import prepare_discovery
from src.modelling.train.train import build_model

version_name = "test"

logging.info("-----------------------")
logging.info("Deconfounding image features")

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

imaging_features_path = Path(
    processed_data_path,
    "mri_all_features_post_deconfound.csv",
)


imaging_features = pd.read_csv(
    imaging_features_path,
    index_col=0,
    low_memory=False,
)

lca_class_membership_path = Path(
    processed_data_path,
    "lca_class_member_entropy.csv",
)

lca_class_membership = pd.read_csv(
    lca_class_membership_path,
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

checkpoint_path = Path(
    analysis_root_path,
    version_name,
    "checkpoints",
)

model_config = {
    "cortical_thickness": {
        "hidden_dim": [10],
        "latent_dim": 5,
        "batch_size": 64,
        "learning_rate": 0.01,
        "epochs": 10,
    },
    "cortical_surface_area": {
        "hidden_dim": [10],
        "latent_dim": 5,
        "batch_size": 64,
        "learning_rate": 0.01,
        "epochs": 10,
    },
    "cortical_volume": {
        "hidden_dim": [10],
        "latent_dim": 5,
        "batch_size": 64,
        "learning_rate": 0.01,
        "epochs": 10,
    },
    "subcortical_volume": {
        "hidden_dim": [10],
        "latent_dim": 5,
        "batch_size": 64,
        "learning_rate": 0.01,
        "epochs": 10,
    },
}

for modality in model_config.keys():
    logging.info("Discovering modality: %s", modality)

    model_checkpoint_path = Path(
        checkpoint_path,
        f"VAE_model_weights_{modality}.pt",
    )

    features = features_of_interest[modality]

    train_dataset, test_dataset, input_dim, discovery_data = prepare_discovery(
        imaging_features=imaging_features,
        lca_class_membership=lca_class_membership,
        features=features,
        data_splits=data_splits,
        if_low_entropy=True,
        entropy_threshold=0.2,
    )

    hyperparameters = model_config["modality"]

    model = build_model(
        config=hyperparameters,
        input_dim=input_dim,
    )

    model.load_state_dict(torch.load(model_checkpoint_path))

    model.eval()

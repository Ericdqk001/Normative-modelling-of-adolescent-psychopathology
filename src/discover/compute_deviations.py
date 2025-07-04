import json
import logging
import os
from pathlib import Path

import pandas as pd
import torch

from .utils import compute_deviations, prepare_discovery
from ..modelling.train.train import build_model


def get_discovery_data(
    model_config: dict,
    version_name: str = "test",
    if_low_entropy: bool = True,
    entropy_threshold: float = 0.2,
):
    logging.info("-----------------------")
    logging.info("Discovering deviations in brain features")

    # Get paths from environment variables
    data_store_path = Path(os.getenv("ABCD_DATA_ROOT", "./abcd_data"))
    analysis_root_path = Path(os.getenv("ANALYSIS_ROOT", "./analysis_output"))

    if data_store_path.exists():
        logging.info("Data store path: %s", data_store_path)

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

    for modality in model_config.keys():
        logging.info("Discovering modality: %s", modality)

        model_checkpoint_path = Path(
            checkpoint_path,
            f"VAE_model_weights_{modality}.pt",
        )

        features = features_of_interest[modality]

        _, test_dataset, input_dim, discovery_data = prepare_discovery(
            imaging_features=imaging_features,
            lca_class_membership=lca_class_membership,
            features=features,
            data_splits=data_splits,
            if_low_entropy=if_low_entropy,
            entropy_threshold=entropy_threshold,
        )

        hyperparameters = model_config[modality]

        model = build_model(
            config=hyperparameters,
            input_dim=input_dim,
        )

        model.load_state_dict(torch.load(model_checkpoint_path))

        model.eval()

        discovery_data = compute_deviations(
            model=model,
            test_dataset=test_dataset,
            discovery_data=discovery_data,
        )

        results_path = Path(
            analysis_root_path,
            version_name,
            "results",
        )

        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        discovery_data_save_path = Path(
            results_path,
            f"discovery_data_{modality}.csv",
        )

        discovery_data.to_csv(discovery_data_save_path)

        logging.info(
            "Saved discovery data for %s to %s",
            modality,
            discovery_data_save_path,
        )


if __name__ == "__main__":
    model_config = {
        "cortical_thickness": {
            "hidden_dim": [10],
            "latent_dim": 5,
            "batch_size": 64,
            "learning_rate": 0.01,
            "epochs": 1000,
        },
        "cortical_surface_area": {
            "hidden_dim": [10],
            "latent_dim": 5,
            "batch_size": 64,
            "learning_rate": 0.01,
            "epochs": 1000,
        },
        "cortical_volume": {
            "hidden_dim": [10],
            "latent_dim": 5,
            "batch_size": 64,
            "learning_rate": 0.01,
            "epochs": 1000,
        },
        "subcortical_volume": {
            "hidden_dim": [10],
            "latent_dim": 5,
            "batch_size": 64,
            "learning_rate": 0.01,
            "epochs": 1000,
        },
    }

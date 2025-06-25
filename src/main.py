import logging
import os
from pathlib import Path

from .discover.compute_deviations import get_discovery_data
from .discover.stat_tests import perform_stat_tests
from .LCA.lca_cbcl import perform_lca
from .modelling.train.train import train
from .preprocess.deconfound import deconfound
from .preprocess.prepare_data import prepare_data
from .preprocess.split import split


def main(
    version_name: str = "test",
    num_blrt_repetitions: int = 1,
    config: dict = None,
    if_low_entropy: bool = True,
    entropy_threshold: float = 0.2,
):
    # Prepare the data for the specified wave
    prepare_data(version_name=version_name)

    # Perform LCA on the prepared data
    perform_lca(
        version_name=version_name,
        num_blrt_repetitions=num_blrt_repetitions,
    )

    # Split the data into sets
    split(
        version_name=version_name,
    )

    # Deconfound the imaging features
    deconfound(
        version_name=version_name,
    )

    modality_list = config.keys()

    for modality in modality_list:
        modality_train_config = config[modality]
        modality_train_config["modality"] = modality
        train(modality_train_config, version_name)

    # Discover deviations in brain features
    get_discovery_data(
        model_config=config,
        version_name=version_name,
        if_low_entropy=if_low_entropy,
        entropy_threshold=entropy_threshold,
    )

    # Perform statistical tests on the discovered deviations

    for modality in modality_list:
        perform_stat_tests(
            version_name=version_name,
            modality=modality,
        )


if __name__ == "__main__":
    # Configuration
    num_blrt_repetitions = 1
    if_low_entropy = True
    entropy_threshold = 0.2
    version_name = f"test_entropy_{entropy_threshold}"

    # Set up logging
    # Get paths from environment variables with defaults
    data_store_path = Path(os.getenv("ABCD_DATA_ROOT", "./abcd_data"))
    analysis_root_path = Path(os.getenv("ANALYSIS_ROOT", "./analysis_output"))

    version_path = Path(
        analysis_root_path,
        version_name,
    )
    version_path.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = version_path / "experiment.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # Also log to console
        ],
    )

    logging.info("Starting experiment with version: %s", version_name)
    logging.info("Log file saved to: %s", log_file)

    hyperparameter_config = {
        "cortical_thickness": {
            "hidden_dim": [10],
            "latent_dim": 5,
            "batch_size": 64,
            "learning_rate": 0.005,
            "epochs": 1000,
        },
        "cortical_surface_area": {
            "hidden_dim": [10],
            "latent_dim": 5,
            "batch_size": 64,
            "learning_rate": 0.005,
            "epochs": 1000,
        },
        "cortical_volume": {
            "hidden_dim": [10],
            "latent_dim": 5,
            "batch_size": 64,
            "learning_rate": 0.005,
            "epochs": 1000,
        },
        "subcortical_volume": {
            "hidden_dim": [10],
            "latent_dim": 5,
            "batch_size": 64,
            "learning_rate": 0.005,
            "epochs": 1000,
        },
    }

    main(
        version_name=version_name,
        num_blrt_repetitions=num_blrt_repetitions,
        config=hyperparameter_config,
        if_low_entropy=if_low_entropy,
        entropy_threshold=entropy_threshold,
    )

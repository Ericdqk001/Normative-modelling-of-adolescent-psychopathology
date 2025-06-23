from src.LCA.lca_cbcl import perform_lca
from src.modelling.train.train import train
from src.preprocess.deconfound import deconfound
from src.preprocess.prepare_data import prepare_data
from src.preprocess.split import split


def main(
    version_name: str = "test",
    num_blrt_repetitions: int = 1,
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

    modality_list = [
        "cortical_thickness",
        "cortical_surface_area",
        "cortical_volume",
        "subcortical_volume",
    ]

    for modality in modality_list:
        modality_train_config = modality_train_config[modality]
        modality_train_config["modality"] = modality
        train(modality_train_config)


if __name__ == "__main__":
    modality_train_config = {
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

    version_name = "test"
    num_blrt_repetitions = 1
    main(
        version_name=version_name,
        num_blrt_repetitions=num_blrt_repetitions,
    )

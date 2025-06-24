import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_discovery(
    imaging_features,
    lca_class_membership,
    features,
    data_splits,
    if_low_entropy=True,
    entropy_threshold=0.2,
):
    """Prepare scaled train/test datasets and group indicator DataFrame.

    Filters out subjects with high entropy if specified. The resulting datasets and
    output_data are used for computing and storing deviation scores
    in "./compute_deviations.py".

    Args:
        imaging_features (pd.DataFrame): DataFrame of imaging features.
        lca_class_membership (pd.DataFrame): DataFrame with LCA class membership
        and entropy.
        features (list): List of feature column names to use.
        data_splits (dict): Dictionary of subject splits for train/test/groups.
        if_low_entropy (bool, optional): Whether to exclude high-entropy subjects.
        Defaults to True.
        entropy_threshold (float, optional): Entropy cutoff for exclusion.
        Defaults to 0.2.

    Returns:
        tuple: (train_dataset, test_dataset, input_dim, output_data)
            - train_dataset (pd.DataFrame): Scaled training data.
            - test_dataset (pd.DataFrame): Scaled test data.
            - input_dim (int): Number of input features.
            - discovery_data (pd.DataFrame): Binary group indicator DataFrame for
            test subjects.
    """
    imaging_features_lca_class = imaging_features.join(
        lca_class_membership,
        how="left",
    )

    train_subs = data_splits["train"]
    test_subs = data_splits["total_test"]
    low_symp_test_subs = data_splits["low_symptom_test"]
    inter_test_subs = data_splits["internalising_test"]
    exter_test_subs = data_splits["externalising_test"]
    high_test_subs = data_splits["high_symptom_test"]

    if if_low_entropy:
        # Remove subjects with high entropy
        high_entropy_subs = imaging_features_lca_class.loc[
            imaging_features_lca_class["shannon_entropy"] > entropy_threshold
        ].index

        # Filter all subject lists in one step
        subject_lists = [
            "train_subs",
            "test_subs",
            "low_symp_test_subs",
            "inter_test_subs",
            "exter_test_subs",
            "high_test_subs",
        ]
        for name in subject_lists:
            locals()[name] = [
                sub for sub in locals()[name] if sub not in high_entropy_subs
            ]

    scaler = StandardScaler()

    train_dataset = imaging_features_lca_class.loc[
        train_subs,
        features,
    ]

    train_dataset_scaled = scaler.fit_transform(train_dataset)

    # Convert the numpy array back to a DataFrame
    train_dataset = pd.DataFrame(
        train_dataset_scaled,
        index=train_dataset.index,
        columns=train_dataset.columns,
    )

    test_dataset = imaging_features_lca_class.loc[
        test_subs,
        features,
    ]

    test_dataset_scaled = scaler.transform(test_dataset)

    test_dataset = pd.DataFrame(
        test_dataset_scaled,
        index=test_dataset.index,
        columns=test_dataset.columns,
    )

    input_dim = train_dataset.shape[1]

    group_dict = {
        "low_symp_test_subs": low_symp_test_subs,
        "inter_test_subs": inter_test_subs,
        "exter_test_subs": exter_test_subs,
        "high_test_subs": high_test_subs,
    }

    # Prepare an output DataFrame with binary indicators for each test group
    discovery_data = pd.DataFrame(
        {k: pd.Series(test_subs).isin(v).astype(int) for k, v in group_dict.items()},
        index=test_subs,
    )

    return (
        train_dataset,
        test_dataset,
        input_dim,
        discovery_data,
    )


def compute_deviations(
    model,
    test_dataset=None,
    discovery_data=None,
) -> pd.DataFrame:
    """Compute whole-brain and regional reconstruction deviations for test subjects.

    Uses the model to predict reconstructed imaging features,
    then calculates mean squared error (MSE) for the whole brain
    and for each brain region.
    Results are added as new columns to discovery_data.

    Args:
        model: Trained model with a pred_recon method for reconstruction.
        test_dataset (pd.DataFrame, optional): Scaled test dataset. Defaults to None.
        discovery_data (pd.DataFrame, optional): DataFrame to store deviation results.
        Defaults to None.

    Returns:
        pd.DataFrame: discovery_data with added columns for whole-brain
        and regional deviations.
    """
    test_prediction = model.pred_recon(
        test_dataset,
        DEVICE,
    )

    discovery_data["whole_brain_deviation"] = whole_brain_deviation(
        test_dataset.to_numpy(),
        test_prediction,
    )

    # Record reconstruction deviation for each brain region

    for i in range(test_prediction.shape[1]):
        discovery_data[f"regional_deviation_{i}"] = regional_deviation(
            test_dataset.to_numpy()[:, i],
            test_prediction[:, i],
        )

    return discovery_data


def whole_brain_deviation(x, x_pred):
    dev = np.mean((x - x_pred) ** 2, axis=1)

    return dev


def regional_deviation(x, x_pred):
    dev = (x - x_pred) ** 2

    return dev

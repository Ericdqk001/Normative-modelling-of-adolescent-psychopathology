# %%
import json
import logging
from pathlib import Path

import pandas as pd
from statsmodels.formula.api import ols


def deconfound(
    version_name: str = "test",
) -> None:
    # %%
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
        "mri_all_features_post_combat_rescaled.csv",
    )

    imaging_features = pd.read_csv(
        imaging_features_path,
        index_col=0,
        low_memory=False,
    )

    imaging_data_splits_path = Path(
        processed_data_path,
        "imaging_data_splits.json",
    )

    with open(imaging_data_splits_path, "r") as f:
        imaging_data_splits = json.load(f)

    features_of_interest_path = Path(
        processed_data_path,
        "features_of_interest.json",
    )

    with open(features_of_interest_path, "r") as f:
        features_of_interest = json.load(f)

    imaging_feature_list = (
        features_of_interest["cortical_thickness"]
        + features_of_interest["cortical_volume"]
        + features_of_interest["cortical_surface_area"]
        + features_of_interest["subcortical_volume"]
    )

    # %%
    ### Regressing out other confounders by sets and modality because the global feature
    # covariate is different for each modality

    train_set = imaging_features.loc[imaging_data_splits["train"]]

    val_set = imaging_features.loc[imaging_data_splits["val"]]

    test_set = imaging_features.loc[imaging_data_splits["total_test"]]

    logging.info("Regressing out confounders for imaging features...")

    for imaging_feature in imaging_feature_list:
        # Define and fit the model
        formula = f"{imaging_feature} ~ interview_age + C(demo_sex_v2) + C(demo_comb_income_v2)"

        # Add covariates based on the imaging feature modality
        if imaging_feature in features_of_interest["cortical_thickness"]:
            formula += " + smri_thick_cdk_mean"
        elif imaging_feature in features_of_interest["cortical_volume"]:
            formula += " + smri_vol_scs_intracranialv"
        elif imaging_feature in features_of_interest["cortical_surface_area"]:
            formula += " + smri_area_cdk_total"
        elif imaging_feature in features_of_interest["subcortical_volume"]:
            formula += " + smri_vol_scs_intracranialv"

        model = ols(formula, train_set).fit()

        print(model.summary())

        # Calculate residuals for the training set
        train_set[imaging_feature] = model.resid

        # Predict on the validation set and calculate residuals
        val_predictions = model.predict(val_set)
        val_set[imaging_feature] = val_set[imaging_feature] - val_predictions

        # Predict on the test set and calculate residuals
        test_predictions = model.predict(test_set)
        test_set[imaging_feature] = test_set[imaging_feature] - test_predictions

    # Combine all sets into one DataFrame
    imaging_features_resid_df = pd.concat(
        [train_set, val_set, test_set],
        axis=0,
    ).reindex(imaging_features.index)

    # Save the deconfounded imaging features
    imaging_features_resid_path = Path(
        processed_data_path,
        "mri_all_features_post_deconfound.csv",
    )

    imaging_features_resid_df.to_csv(
        imaging_features_resid_path,
        index=True,
    )


if __name__ == "__main__":
    deconfound()

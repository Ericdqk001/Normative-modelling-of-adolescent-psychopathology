import json
import logging
import os
import random
from pathlib import Path

import pandas as pd


def split(
    version_name: str = "test",
):
    logging.info("-----------------------")

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

    # load prepared data from "preprocess/prepare_data.py"
    imaging_features_path = Path(
        processed_data_path,
        "mri_all_features_post_combat_rescaled.csv",
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

    imaging_features_lca_class = imaging_features.join(
        lca_class_membership["predicted_class"],
        how="left",
    )

    imaging_low_symptom = imaging_features_lca_class[
        imaging_features_lca_class["predicted_class"] == 0
    ]

    logging.info("Sample size of imaging_low_symptom: %d", len(imaging_low_symptom))

    imaging_interalising = imaging_features_lca_class[
        imaging_features_lca_class["predicted_class"] == 1
    ]

    logging.info("Sample size of imaging_interalising: %d", len(imaging_interalising))

    imaging_externalising = imaging_features_lca_class[
        imaging_features_lca_class["predicted_class"] == 2
    ]

    logging.info("Sample size of imaging_externalising: %d", len(imaging_externalising))

    imaging_high_symptom = imaging_features_lca_class[
        imaging_features_lca_class["predicted_class"] == 3
    ]

    logging.info("Sample size of t1w_high_symptom: %d", len(imaging_high_symptom))

    # Note: Psychiatric diagnosis filtering removed for public release
    # Users can implement their own control group filtering if needed
    
    logging.info(
        "Sample size of imaging_low_symptom: %d",
        len(imaging_low_symptom),
    )

    cbcl_scales = [
        "cbcl_scr_syn_aggressive_t",
        "cbcl_scr_syn_rulebreak_t",
        "cbcl_scr_syn_attention_t",
        "cbcl_scr_syn_thought_t",
        "cbcl_scr_syn_social_t",
        "cbcl_scr_syn_somatic_t",
        "cbcl_scr_syn_withdep_t",
        "cbcl_scr_syn_anxdep_t",
    ]

    # Filter out rows where any cbcl_scales column has value 1
    imaging_low_symptom = imaging_low_symptom[
        ~(imaging_low_symptom[cbcl_scales] == 1).any(axis=1)
    ]

    logging.info(
        "Sample size of imaging_low_symptom after removing rows with any cbcl_scales == 1: %d",
        len(imaging_low_symptom),
    )

    (
        imaging_low_symptom_train_subs,
        imaging_low_symptom_val_subs,
        imaging_low_symptom_test_subs,
    ) = split_low_symptom(imaging_low_symptom)

    imaging_data_splits = {
        "train": imaging_low_symptom_train_subs,
        "val": imaging_low_symptom_val_subs,
        "total_test": imaging_low_symptom_test_subs
        + imaging_interalising.index.to_list()
        + imaging_externalising.index.to_list()
        + imaging_high_symptom.index.to_list(),
        "low_symptom_test": imaging_low_symptom_test_subs,
        "internalising_test": imaging_interalising.index.to_list(),
        "externalising_test": imaging_externalising.index.to_list(),
        "high_symptom_test": imaging_high_symptom.index.to_list(),
    }

    results_path = Path(
        processed_data_path,
        "imaging_data_splits.json",
    )

    with open(
        results_path,
        "w",
    ) as f:
        json.dump(imaging_data_splits, f)

    for set, subs in imaging_data_splits.items():
        logging.info(
            "Sample size of %s: %d",
            set,
            len(subs),
        )


def split_low_symptom(
    data,
    train_ratio=0.8,
    val_ratio=0.1,
    stratify_var=["demo_sex_v2", "demo_comb_income_v2"],
    random_seed=42,
):
    """Split data indices into train, validation, and test sets with stratification.

    Stratifies by the specified variables, shuffles within each stratum, and splits
    according to the provided ratios.

    Args:
        data (pd.DataFrame): Input DataFrame containing the data to split.
        train_ratio (float, optional): Proportion for training set. Defaults to 0.8.
        val_ratio (float, optional): Proportion for validation set. Defaults to 0.1.
        stratify_var (list, optional): List of column names to stratify by.
        Defaults to ["demo_sex_v2", "demo_comb_income_v2"].
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: Lists of indices for train, validation, and test sets.
    """
    train_subs = []
    val_subs = []
    test_subs = []

    for combination in data[stratify_var].drop_duplicates().values:
        sex, income = combination

        strata = data[
            (data["demo_sex_v2"] == sex) & (data["demo_comb_income_v2"] == income)
        ].index.to_list()

        random.seed(random_seed)

        random.shuffle(strata)

        train_size = int(len(strata) * train_ratio)

        val_size = int(len(strata) * val_ratio)

        train_subs.extend(strata[:train_size])

        val_subs.extend(strata[train_size : train_size + val_size])

        test_subs.extend(strata[train_size + val_size :])

    return train_subs, val_subs, test_subs

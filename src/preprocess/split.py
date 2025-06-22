import json
import random
from pathlib import Path
from typing import List

import pandas as pd


def split(
    ct_features: pd.DataFrame,
    fmri_features: pd.DataFrame,
):

    # load CBCL data

    cbcl_LCA_path = Path(
        "data",
        "LCA",
        "cbcl_final_class_member.csv",
    )

    cbcl_LCA = pd.read_csv(
        cbcl_LCA_path,
        index_col=0,
        low_memory=False,
    )

    # Load psychiatric diagnoses

    psych_dx_path = Path(
        "data",
        "liza_data",
        "all_psych_dx_r5.csv",
    )

    psych_dx = pd.read_csv(
        psych_dx_path,
        index_col=0,
        low_memory=False,
    )

    cbcl_columns_to_join = [
        "cbcl_scr_syn_anxdep_t",
        "cbcl_scr_syn_withdep_t",
        "cbcl_scr_syn_somatic_t",
        "cbcl_scr_syn_social_t",
        "cbcl_scr_syn_thought_t",
        "cbcl_scr_syn_attention_t",
        "cbcl_scr_syn_rulebreak_t",
        "cbcl_scr_syn_aggressive_t",
        "cbcl_scr_syn_internal_t",
        "cbcl_scr_syn_external_t",
        "cbcl_scr_syn_totprob_t",
        "entropy",
    ]

    ct_features = ct_features.join(
        cbcl_LCA[cbcl_columns_to_join],
        how="left",
    ).join(
        psych_dx,
        how="left",
    )

    fmri_features = fmri_features.join(
        cbcl_LCA[cbcl_columns_to_join],
        how="left",
    ).join(
        psych_dx,
        how="left",
    )

    t1w_low_symptom = ct_features[ct_features["predicted_class"] == 1]

    print("t1w_low_symptom")
    print(len(t1w_low_symptom))

    t1w_internalising = ct_features[ct_features["predicted_class"] == 2]

    print("t1w_internalising")
    print(len(t1w_internalising))

    t1w_externalising = ct_features[ct_features["predicted_class"] == 3]

    print("t1w_externalising")
    print(len(t1w_externalising))

    t1w_high_symptom = ct_features[ct_features["predicted_class"] == 4]

    print("t1w_high_symptom")
    print(len(t1w_high_symptom))

    rsfmri_low_symptom = fmri_features[fmri_features["predicted_class"] == 1]

    print("rsfmri_low_symptom")
    print(len(rsfmri_low_symptom))

    rsfmri_internalising = fmri_features[fmri_features["predicted_class"] == 2]

    print("rsfmri_internalising")
    print(len(rsfmri_internalising))

    rsfmri_externalising = fmri_features[fmri_features["predicted_class"] == 3]

    print("rsfmri_externalising")
    print(len(rsfmri_externalising))

    rsfmri_high_symptom = fmri_features[fmri_features["predicted_class"] == 4]

    print("rsfmri_high_symptom")
    print(len(rsfmri_high_symptom))

    def split_low_symptom(
        data: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        stratify_var: List = ["demo_sex_v2", "demo_comb_income_v2"],
        random_seed: int = 42,
    ):

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

    ### Remove subjects with any psychiatric diagnoses or positive for any of the CBCL
    # syndrome scales (Tested: works)

    t1w_low_symptom = t1w_low_symptom[t1w_low_symptom["psych_dx"] == "control"]

    rsfmri_low_symptom = rsfmri_low_symptom[rsfmri_low_symptom["psych_dx"] == "control"]

    t1w_low_symptom = t1w_low_symptom[
        ~t1w_low_symptom[cbcl_columns_to_join[:-1]].eq(2).any(axis=1)
    ]

    rsfmri_low_symptom = rsfmri_low_symptom[
        ~rsfmri_low_symptom[cbcl_columns_to_join[:-1]].eq(2).any(axis=1)
    ]

    t1w_low_symptom_train_subs, t1w_low_symptom_val_subs, t1w_low_symptom_test_subs = (
        split_low_symptom(t1w_low_symptom)
    )

    (
        rsfmri_low_symptom_train_subs,
        rsfmri_low_symptom_val_subs,
        rsfmri_low_symptom_test_subs,
    ) = split_low_symptom(rsfmri_low_symptom)

    t1w_data_splits = {
        "train": t1w_low_symptom_train_subs,
        "val": t1w_low_symptom_val_subs,
        "total_test": t1w_low_symptom_test_subs
        + t1w_internalising.index.to_list()
        + t1w_externalising.index.to_list()
        + t1w_high_symptom.index.to_list(),
        "low_symptom_test": t1w_low_symptom_test_subs,
        "internalising_test": t1w_internalising.index.to_list(),
        "externalising_test": t1w_externalising.index.to_list(),
        "high_symptom_test": t1w_high_symptom.index.to_list(),
    }

    rsfmri_data_splits = {
        "train": rsfmri_low_symptom_train_subs,
        "val": rsfmri_low_symptom_val_subs,
        "total_test": rsfmri_low_symptom_test_subs
        + rsfmri_internalising.index.to_list()
        + rsfmri_externalising.index.to_list()
        + rsfmri_high_symptom.index.to_list(),
        "low_symptom_test": rsfmri_low_symptom_test_subs,
        "internalising_test": rsfmri_internalising.index.to_list(),
        "externalising_test": rsfmri_externalising.index.to_list(),
        "high_symptom_test": rsfmri_high_symptom.index.to_list(),
    }

    data_splits = {
        "structural": t1w_data_splits,
        "functional": rsfmri_data_splits,
    }

    processed_data_path = Path(
        "data",
        "processed_data",
    )

    with open(Path(processed_data_path, "data_splits.json"), "w") as f:
        json.dump(data_splits, f)

    for key, value in t1w_data_splits.items():
        print(f"Length of {key} in t1w_data_splits: {len(value)}")

    print("\n")

    for key, value in rsfmri_data_splits.items():
        print(f"Length of {key} in rsfmri_data_splits: {len(value)}")

    return t1w_data_splits, rsfmri_data_splits


# if __name__ == "__main__":
#     processed_data_path = Path(
#         "data",
#         "processed_data",
#     )

#     t1w_cortical_features_resid = Path(
#         processed_data_path,
#         "t1w_cortical_features_post_combat.csv",
#     )

#     rsfmri_features_resid = Path(
#         processed_data_path,
#         "gordon_cor_subcortical_post_combat.csv",
#     )

#     t1w_cortical_features_resid = pd.read_csv(t1w_cortical_features_resid, index_col=0)

#     rsfmri_features_resid = pd.read_csv(rsfmri_features_resid, index_col=0)

#     split(
#         t1w_cortical_features_resid,
#         rsfmri_features_resid,
#     )

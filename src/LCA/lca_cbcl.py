from pathlib import Path

import numpy as np
import pandas as pd
from stepmix.bootstrap import blrt_sweep
from stepmix.stepmix import StepMix


def perform_lca(
    version_name: str = "test",
):
    """Perform latent class analysis (LCA).

    Saves:
        - 'lca_model_stats.csv': Model fit statistics
        (BIC, AIC, log-likelihood, entropy, etc.) and BLRT p-values.
        - 'lca_class_parameters.csv': variable-specific probabilities for each class,
        class proportions.
        - 'lca_class_member_entropy.csv': Individual predicted class, class membership.

    Args:
        version_name (str, optional): Subfolder name for processed data. Defaults to
        "test".
    """
    # Load processed data from "preprocess/prepare_data.py"
    data_store_path = Path(
        "/",
        "Volumes",
        "GenScotDepression",
    )

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

    rescaled_features_post_combat = pd.read_csv(
        Path(
            processed_data_path,
            "mri_all_features_post_combat_rescaled.csv",
        ),
        index_col=0,
        low_memory=False,
    )

    data_path = Path(
        analysis_root_path,
        "data",
    )

    cbcl_variables_path = Path(
        data_path,
        "cbcl_8_dim_t.csv",
    )

    cbcl_variables_df = pd.read_csv(
        cbcl_variables_path,
        index_col=0,
        low_memory=False,
    )

    cbcl_variable_names = cbcl_variables_df["var_name"].tolist()

    # %%
    # Perform StepMix LCA analysis

    data = rescaled_features_post_combat[cbcl_variable_names].copy()

    min_classes = 2
    max_classes = 6

    results_dict = {
        "n_classes": [],
        "bic": [],
        "aic": [],
        "log_likelihood": [],
        "abic": [],
        "relative_entropy": [],
        "caic": [],
    }

    for n_classes in range(min_classes, max_classes + 1):
        model = StepMix(
            n_components=n_classes,
            measurement="binary",
            n_steps=1,
            verbose=1,
            random_state=42,
        )
        model.fit(data)

        results_dict["n_classes"].append(n_classes)
        results_dict["bic"].append(model.bic(data))
        results_dict["aic"].append(model.aic(data))
        results_dict["relative_entropy"].append(model.relative_entropy(data))
        results_dict["log_likelihood"].append(model.score(data) * data.shape[0])
        results_dict["abic"].append(model.sabic(data))
        results_dict["caic"].append(model.caic(data))

    num_classes = 4

    model = StepMix(
        n_components=num_classes,
        measurement="binary",
        n_steps=1,
        verbose=1,
        random_state=42,
    )

    blrt_results = blrt_sweep(
        model,
        data,
        low=min_classes,
        high=max_classes,
        n_repetitions=1,
    )

    # Prepend a null value for the first class
    p_values = [None] + blrt_results["p"].values.tolist()

    results_df = pd.DataFrame(results_dict)
    results_df["blrt_p_value"] = p_values

    # Save model stats to CSV
    results_df.to_csv(
        Path(
            processed_data_path,
            "lca_model_stats.csv",
        ),
        index=False,
    )

    # Predict class membership and calculate posterior probabilities
    model.fit(data)

    # Swap class numbers so the largest class is first
    model.permute_classes(np.array([1, 0, 2, 3]))

    params_df = model.get_parameters_df().reset_index()

    df_pis = params_df[params_df["param"] == "pis"]

    df_pis_wide = df_pis.pivot(
        index="class_no",
        columns="variable",
        values="value",
    )

    df_weights = params_df[params_df["model_name"] == "class_weights"][
        ["class_no", "value"]
    ]
    df_weights = df_weights.rename(columns={"value": "class_proportion"}).set_index(
        "class_no"
    )

    param_df_wide = df_pis_wide.join(df_weights)

    # Save parameters to CSV
    param_df_wide.to_csv(
        Path(
            processed_data_path,
            "lca_class_parameters.csv",
        ),
        index=True,
    )

    predicted_class = model.predict(data)
    posterior_probs = model.predict_proba(data)

    posterior_probs_df = pd.DataFrame(
        posterior_probs,
        columns=[f"post_prob_class_{i}" for i in range(num_classes)],
        index=data.index,
    )

    # Apply to each row of the posterior probability DataFrame
    cbcl_results_df = pd.DataFrame(
        index=data.index,
    )

    cbcl_results_df["shannon_entropy"] = posterior_probs_df.apply(
        compute_shannon_entropy, axis=1
    )

    cbcl_results_df["predicted_class"] = predicted_class

    # Save the predicted class and shannon entropy to CSV
    cbcl_results_df.to_csv(
        Path(
            processed_data_path,
            "lca_class_member_entropy.csv",
        ),
        index=True,
    )


# Compute individual entropy
def compute_shannon_entropy(posterior_probs_row):
    """Compute the Shannon entropy of a probability distribution for each individual.

    Entropy is calculated as:
        H(p) = -∑ p_i * log(p_i)
    where p_i are the posterior probabilities.

    Args:
        posterior_probs_row (np.ndarray): Array of posterior probabilities for one
        individual.

    Returns:
        float: Entropy value.
    """
    # Replace zeros to avoid log(0)
    safe_probs = np.where(
        posterior_probs_row == 0,
        1e-10,
        posterior_probs_row,
    )

    entropy = -np.sum(
        safe_probs * np.log(safe_probs),
    )

    return entropy

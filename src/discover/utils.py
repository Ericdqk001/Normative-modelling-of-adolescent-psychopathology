import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import fdrcorrection

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

    # Create discovery_data DataFrame with binary group indicators
    discovery_data = pd.DataFrame(index=test_subs)

    # Initialize all columns with 0
    discovery_data["low_symp_test_subs"] = 0
    discovery_data["inter_test_subs"] = 0
    discovery_data["exter_test_subs"] = 0
    discovery_data["high_test_subs"] = 0

    # Set 1 for subjects belonging to each group
    discovery_data.loc[
        discovery_data.index.isin(low_symp_test_subs), "low_symp_test_subs"
    ] = 1
    discovery_data.loc[
        discovery_data.index.isin(inter_test_subs), "inter_test_subs"
    ] = 1
    discovery_data.loc[
        discovery_data.index.isin(exter_test_subs), "exter_test_subs"
    ] = 1
    discovery_data.loc[discovery_data.index.isin(high_test_subs), "high_test_subs"] = 1

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

    # Validate dimensions match
    if test_prediction.shape[1] != len(test_dataset.columns):
        raise ValueError(
            f"Prediction shape {test_prediction.shape[1]} does not match "
            f"test dataset features {len(test_dataset.columns)}"
        )

    discovery_data["whole_brain_deviation"] = whole_brain_deviation(
        test_dataset.to_numpy(),
        test_prediction,
    )

    # Record reconstruction deviation for each brain region using actual feature names
    for i, feature_name in enumerate(test_dataset.columns):
        discovery_data[f"regional_deviation_{feature_name}"] = regional_deviation(
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


def extract_groups_for_metric(discovery_data, metric_name):
    """Extract group data for a specific deviation metric.

    Args:
        discovery_data (pd.DataFrame): DataFrame containing group indicators
        and metrics.
        metric_name (str): Name of the metric column to extract.

    Returns:
        dict: Dictionary with group names as keys and metric values as values.
    """
    groups = {
        "control": discovery_data[metric_name][
            discovery_data["low_symp_test_subs"] == 1
        ].values,
        "inter_test": discovery_data[metric_name][
            discovery_data["inter_test_subs"] == 1
        ].values,
        "exter_test": discovery_data[metric_name][
            discovery_data["exter_test_subs"] == 1
        ].values,
        "high_test": discovery_data[metric_name][
            discovery_data["high_test_subs"] == 1
        ].values,
    }

    return groups


def test_single_metric_assumptions(discovery_data, metric_name):
    """Test normality and equal variance assumptions for one metric.

    Args:
        discovery_data (pd.DataFrame): DataFrame containing group indicators
        and metrics.
        metric_name (str): Name of the metric column to test.

    Returns:
        pd.DataFrame: DataFrame with test results for normality and equal variance.
    """
    groups = extract_groups_for_metric(discovery_data, metric_name)

    # Initialize list to store results for DataFrame
    results = []

    # Test for normality using the Shapiro-Wilk test
    for group_name, values in groups.items():
        if len(values) > 0:  # Only test if group has data
            shapiro_stat, shapiro_p = stats.shapiro(values)
            results.append(
                {
                    "metric": metric_name,
                    "group": group_name,
                    "test": "Shapiro-Wilk",
                    "statistic": shapiro_stat,
                    "p_value": shapiro_p,
                }
            )

    # Test for equal variances using Levene's test
    # Only include groups with data
    valid_groups = [values for values in groups.values() if len(values) > 0]
    if len(valid_groups) >= 2:  # Need at least 2 groups for Levene's test
        levene_stat, levene_p = stats.levene(
            *valid_groups,
            center="median",  # Recommended when distributions are not symmetrical
        )
        results.append(
            {
                "metric": metric_name,
                "group": "All",
                "test": "Levene",
                "statistic": levene_stat,
                "p_value": levene_p,
            }
        )

    # Convert results list to a DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def perform_single_mannwhitney_u(control_group, test_group):
    """Perform Mann-Whitney U test between two groups.

    Args:
        control_group (np.array): Control group values.
        test_group (np.array): Test group values.

    Returns:
        tuple: (U-statistic, p-value)
    """
    if len(control_group) == 0 or len(test_group) == 0:
        return np.nan, np.nan

    u_statistic, p_value = stats.mannwhitneyu(
        control_group, test_group, alternative="two-sided"
    )

    return u_statistic, p_value


def test_U_test_assumptions(discovery_data):
    """Check assumptions of normality and equal variance for all deviation metrics.

    Tests all whole_brain_deviation and regional_deviation columns in discovery_data
    for normality (Shapiro-Wilk) and equal variance (Levene's test) across groups.

    Args:
        discovery_data (pd.DataFrame): DataFrame containing group indicators and
        deviation metrics.

    Returns:
        pd.DataFrame: Combined test results for all metrics with columns:
                     ['metric', 'group', 'test', 'statistic', 'p_value']
    """
    # Identify all deviation metric columns
    deviation_columns = [
        col
        for col in discovery_data.columns
        if col.startswith(("whole_brain_deviation", "regional_deviation_"))
    ]

    all_results = []

    # Test assumptions for each deviation metric
    for metric in deviation_columns:
        metric_results = test_single_metric_assumptions(discovery_data, metric)
        all_results.append(metric_results)

    # Combine all results into a single DataFrame
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
    else:
        combined_results = pd.DataFrame(
            columns=["metric", "group", "test", "statistic", "p_value"]
        )

    return combined_results


def perform_U_test(discovery_data):
    """Perform U tests for all deviation metrics across test groups vs control.

    Compares each test group (inter_test, exter_test, high_test) against control group
    for all deviation metrics, then applies FDR correction across all tests.

    Args:
        discovery_data (pd.DataFrame): DataFrame containing group indicators
        and deviation metrics.

    Returns:
        pd.DataFrame: Results with columns:
        ['metric', 'test_group', 'U_statistic', 'p_value', 'p_value_FDR_corrected']
    """
    # Identify all deviation metric columns
    deviation_columns = [
        col
        for col in discovery_data.columns
        if col.startswith(("whole_brain_deviation", "regional_deviation_"))
    ]

    # Define test groups (excluding control)
    test_groups = ["inter_test", "exter_test", "high_test"]

    all_results = []
    all_p_values = []

    # Perform U-tests for each metric and each test group vs control
    for metric in deviation_columns:
        groups = extract_groups_for_metric(discovery_data, metric)
        control_group = groups["control"]

        for test_group_name in test_groups:
            test_group = groups[test_group_name]

            # Perform Mann-Whitney U test
            u_statistic, p_value = perform_single_mannwhitney_u(
                control_group, test_group
            )

            all_results.append(
                {
                    "metric": metric,
                    "test_group": test_group_name,
                    "U_statistic": u_statistic,
                    "p_value": p_value,
                }
            )

            # Collect p-values for FDR correction (exclude NaN values)
            if not np.isnan(p_value):
                all_p_values.append(p_value)
            else:
                all_p_values.append(np.nan)

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Apply FDR correction to all p-values
    if len(all_p_values) > 0:
        # Remove NaN values for FDR correction
        valid_p_indices = [i for i, p in enumerate(all_p_values) if not np.isnan(p)]
        valid_p_values = [all_p_values[i] for i in valid_p_indices]

        if len(valid_p_values) > 0:
            # Apply FDR correction
            reject, p_corrected = fdrcorrection(valid_p_values, alpha=0.05)

            # Create corrected p-values array with NaN for invalid entries
            p_corrected_full = np.full(len(all_p_values), np.nan)
            for idx, corrected_p in zip(valid_p_indices, p_corrected):
                p_corrected_full[idx] = corrected_p

            results_df["p_value_FDR_corrected"] = p_corrected_full
        else:
            results_df["p_value_FDR_corrected"] = np.nan
    else:
        results_df["p_value_FDR_corrected"] = np.nan

    return results_df

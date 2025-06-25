import logging

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

        print(f"Removing {len(high_entropy_subs)} high entropy subjects")

        # Filter each subject list individually
        test_subs = [sub for sub in test_subs if sub not in high_entropy_subs]
        low_symp_test_subs = [
            sub for sub in low_symp_test_subs if sub not in high_entropy_subs
        ]
        inter_test_subs = [
            sub for sub in inter_test_subs if sub not in high_entropy_subs
        ]
        exter_test_subs = [
            sub for sub in exter_test_subs if sub not in high_entropy_subs
        ]
        high_test_subs = [sub for sub in high_test_subs if sub not in high_entropy_subs]

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

    # Need at least 2 groups for Levene's test
    if len(valid_groups) >= 2:
        levene_stat, levene_p = stats.levene(
            *valid_groups,
            center="median",
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
    # Identify all regional deviation metric columns
    regional_deviation_columns = [
        col
        for col in discovery_data.columns
        if col.startswith(("regional_deviation_",))
    ]

    # Define test groups (excluding control)
    test_groups = [
        "inter_test",
        "exter_test",
        "high_test",
    ]

    all_results = []

    for test_group_name in test_groups:
        group_results = []
        for metric in regional_deviation_columns:
            # Extract control and test group data for the current metric
            groups = extract_groups_for_metric(discovery_data, metric)
            control_group = groups["control"]
            test_group = groups[test_group_name]

            # Perform Mann-Whitney U test
            u_statistic, p_value = perform_single_mannwhitney_u(
                control_group, test_group
            )

            group_results.append(
                {
                    "metric": metric,
                    "test_group": test_group_name,
                    "U_statistic": u_statistic,
                    "p_value": p_value,
                }
            )

        # Apply FDR correction to the p-values for the test group
        if group_results:
            group_results_df = pd.DataFrame(group_results)

            # Check for NaN p-values
            if group_results_df["p_value"].isna().any():
                logging.error(
                    "NaN p-values found in group %s. Stopping processing.",
                    test_group_name,
                )
                raise ValueError(f"NaN p-values found in group {test_group_name}")

            # Apply FDR correction to the p-values
            rejected, p_corrected = fdrcorrection(
                group_results_df["p_value"].values, alpha=0.05
            )

            group_results_df["p_value_FDR_corrected"] = p_corrected
            group_results_df["significant"] = rejected

            all_results.append(group_results_df)

        # Apply U-test to whole-brain deviation
        whole_brain_metric = "whole_brain_deviation"

        if whole_brain_metric in discovery_data.columns:
            groups = extract_groups_for_metric(discovery_data, whole_brain_metric)
            control_group = groups["control"]
            test_group = groups[test_group_name]

            u_statistic, p_value = perform_single_mannwhitney_u(
                control_group, test_group
            )

            # Check for NaN p-value
            if np.isnan(p_value):
                logging.error(
                    "NaN p-value found for whole-brain deviation in group %s. Stopping processing.",
                    test_group_name,
                )
                raise ValueError(
                    f"NaN p-value found for whole-brain deviation in group {test_group_name}"
                )

            group_results_df = pd.DataFrame(
                {
                    "metric": whole_brain_metric,
                    "test_group": test_group_name,
                    "U_statistic": u_statistic,
                    "p_value": p_value,
                    # Single feature, so no need for correction
                    "p_value_FDR_corrected": p_value,
                },
                index=[0],
            )

            if p_value < 0.05:
                rejected = True
            else:
                rejected = False

            group_results_df["significant"] = rejected

            all_results.append(group_results_df)

    # Combine all results and return
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
    else:
        combined_results = pd.DataFrame(
            columns=[
                "metric",
                "test_group",
                "U_statistic",
                "p_value",
                "p_value_FDR_corrected",
                "significant",
            ]
        )

    return combined_results


def test_hemisphere_differences(discovery_data):
    """Test for significant differences between lh/rh regional deviations within test groups only.

    Tests whether left and right hemisphere regional deviations differ significantly within
    each test group (inter_test, exter_test, high_test). Control group is excluded from
    this analysis.

    Args:
        discovery_data (pd.DataFrame): DataFrame containing group indicators and
        regional deviation metrics.

    Returns:
        pd.DataFrame: Results with columns:
        ['region', 'test_group', 'U_statistic', 'p_value', 'p_value_FDR_corrected', 'significant']
    """
    # Identify regional deviation columns
    regional_columns = [
        col for col in discovery_data.columns if col.startswith("regional_deviation_")
    ]

    # Separate lh and rh columns and find bilateral pairs
    lh_columns = [col for col in regional_columns if col.endswith("lh")]
    rh_columns = [col for col in regional_columns if col.endswith("rh")]

    # Create region mapping by removing hemisphere suffix
    lh_regions = {}
    rh_regions = {}

    for col in lh_columns:
        # Extract region name by removing "regional_deviation_" prefix and "lh" suffix
        region = col.replace("regional_deviation_", "").replace("lh", "")
        lh_regions[region] = col

    for col in rh_columns:
        # Extract region name by removing "regional_deviation_" prefix and "rh" suffix
        region = col.replace("regional_deviation_", "").replace("rh", "")
        rh_regions[region] = col

    # Find bilateral regions (present in both hemispheres)
    bilateral_regions = set(lh_regions.keys()) & set(rh_regions.keys())

    logging.info(
        "Found %d bilateral regional pairs for hemisphere testing",
        len(bilateral_regions),
    )

    # Define test groups (excluding control)
    test_groups = ["inter_test", "exter_test", "high_test"]

    all_results = []

    # Test hemisphere differences for each test group separately
    for test_group_name in test_groups:
        group_results = []
        group_p_values = []

        for region in bilateral_regions:
            lh_col = lh_regions[region]
            rh_col = rh_regions[region]

            # Get test group subjects only
            test_group_mask = discovery_data[f"{test_group_name}_subs"] == 1
            lh_values = discovery_data[lh_col][test_group_mask].values
            rh_values = discovery_data[rh_col][test_group_mask].values

            # Perform Mann-Whitney U test between lh and rh within this test group
            u_statistic, p_value = perform_single_mannwhitney_u(lh_values, rh_values)

            group_results.append(
                {
                    "region": region,
                    "test_group": test_group_name,
                    "U_statistic": u_statistic,
                    "p_value": p_value,
                }
            )

            # Collect p-values for FDR correction
            if not np.isnan(p_value):
                group_p_values.append(p_value)
            else:
                group_p_values.append(np.nan)

        # Apply FDR correction within this test group
        if group_results:
            group_results_df = pd.DataFrame(group_results)

            # Check for NaN p-values
            if group_results_df["p_value"].isna().any():
                logging.error(
                    "NaN p-values found in hemisphere testing for group %s. Stopping processing.",
                    test_group_name,
                )
                raise ValueError(
                    f"NaN p-values found in hemisphere testing for group {test_group_name}"
                )

            # Apply FDR correction
            rejected, p_corrected = fdrcorrection(
                group_results_df["p_value"].values, alpha=0.05
            )

            group_results_df["p_value_FDR_corrected"] = p_corrected
            group_results_df["significant"] = rejected

            all_results.append(group_results_df)

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
    else:
        combined_results = pd.DataFrame(
            columns=[
                "region",
                "test_group",
                "U_statistic",
                "p_value",
                "p_value_FDR_corrected",
                "significant",
            ]
        )

    return combined_results


def create_averaged_regional_deviations(discovery_data, hemisphere_test_results):
    """Average bilateral deviations where no significant lh/rh differences exist in any test group.

    For regions where no test group shows significant hemisphere differences, creates averaged
    regional deviation columns. For regions where any test group shows significant differences,
    keeps separate lh/rh columns.

    Args:
        discovery_data (pd.DataFrame): DataFrame containing group indicators and
        regional deviation metrics.
        hemisphere_test_results (pd.DataFrame): Results from test_hemisphere_differences.

    Returns:
        pd.DataFrame: Modified discovery_data with averaged bilateral regions where appropriate.
    """
    # Create a copy to avoid modifying the original
    result_data = discovery_data.copy()

    # Find regions that show significant differences in ANY test group
    significant_regions = set(
        hemisphere_test_results[hemisphere_test_results["significant"] == True][
            "region"
        ].values
    )

    # Get all bilateral regions from hemisphere test results
    all_tested_regions = set(hemisphere_test_results["region"].values)

    # Regions to average = tested regions - significant regions
    regions_to_average = all_tested_regions - significant_regions

    logging.info(
        "Regions with significant hemisphere differences (keeping separate): %d",
        len(significant_regions),
    )
    logging.info(
        "Regions without significant hemisphere differences (averaging): %d",
        len(regions_to_average),
    )

    # Average the non-significant bilateral regions
    for region in regions_to_average:
        lh_col = f"regional_deviation_{region}lh"
        rh_col = f"regional_deviation_{region}rh"
        avg_col = f"regional_deviation_{region}_avg"

        if lh_col in result_data.columns and rh_col in result_data.columns:
            # Create averaged column for all subjects (including control)
            result_data[avg_col] = (result_data[lh_col] + result_data[rh_col]) / 2

            # Remove the original lh/rh columns
            result_data = result_data.drop(columns=[lh_col, rh_col])

            logging.info("Averaged region: %s", region)

    # Log summary
    remaining_lh_cols = len(
        [
            col
            for col in result_data.columns
            if col.startswith("regional_deviation_") and col.endswith("lh")
        ]
    )
    remaining_rh_cols = len(
        [
            col
            for col in result_data.columns
            if col.startswith("regional_deviation_") and col.endswith("rh")
        ]
    )
    avg_cols = len(
        [
            col
            for col in result_data.columns
            if col.startswith("regional_deviation_") and col.endswith("_avg")
        ]
    )

    logging.info(
        "Final regional deviation columns: %d lh, %d rh, %d averaged",
        remaining_lh_cols,
        remaining_rh_cols,
        avg_cols,
    )

    return result_data

import logging
from pathlib import Path

import pandas as pd

from src.discover.utils import perform_U_test, test_U_test_assumptions


def perform_stat_tests(
    version_name: str = "test",
    modality: str = "cortical_thickness",
):
    """Perform statistical tests on discovery data and save results.

    Args:
        version_name (str): Version name for data paths. Defaults to "test".
        modality (str): Brain modality to analyze. Defaults to "cortical_thickness".
    """
    logging.info("-----------------------")
    logging.info("Performing statistical tests on discovery data")
    logging.info("Modality: %s", modality)

    # Set up paths
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

    results_path = Path(
        analysis_root_path,
        version_name,
        "results",
    )

    # Ensure results directory exists
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    # Load discovery data
    discovery_data_save_path = Path(
        results_path,
        f"discovery_data_{modality}.csv",
    )

    logging.info("Loading discovery data from: %s", discovery_data_save_path)
    discovery_data = pd.read_csv(
        discovery_data_save_path,
        index_col=0,
        low_memory=False,
    )

    # Examine data structure
    logging.info("Discovery data shape: %s", discovery_data.shape)

    # Check group sizes
    group_columns = [
        "low_symp_test_subs",
        "inter_test_subs",
        "exter_test_subs",
        "high_test_subs",
    ]
    for group in group_columns:
        if group in discovery_data.columns:
            group_size = discovery_data[group].sum()
            logging.info("Group %s size: %d", group, group_size)

    # Identify deviation columns
    deviation_columns = [
        col
        for col in discovery_data.columns
        if col.startswith(("whole_brain_deviation", "regional_deviation_"))
    ]
    logging.info("Found %d deviation metrics", len(deviation_columns))

    # Perform assumption testing
    logging.info("Testing statistical assumptions...")
    try:
        assumption_results = test_U_test_assumptions(discovery_data)
        logging.info("Assumption testing completed successfully.")
        logging.info("Assumption results shape: %s", assumption_results.shape)

        # Summary of normality tests
        normality_results = assumption_results[
            assumption_results["test"] == "Shapiro-Wilk"
        ]
        if len(normality_results) > 0:
            failed_normality = (normality_results["p_value"] < 0.05).sum()
            total_normality = len(normality_results)
            logging.info(
                "Normality tests: %d/%d failed (p < 0.05)",
                failed_normality,
                total_normality,
            )

        # Summary of variance tests
        variance_results = assumption_results[assumption_results["test"] == "Levene"]
        if len(variance_results) > 0:
            failed_variance = (variance_results["p_value"] < 0.05).sum()
            total_variance = len(variance_results)
            logging.info(
                "Equal variance tests: %d/%d failed (p < 0.05)",
                failed_variance,
                total_variance,
            )

    except Exception as e:
        logging.error("Error in assumption testing: %s", str(e))
        raise

    # Perform U-tests
    logging.info("Performing Mann-Whitney U tests...")
    try:
        u_test_results = perform_U_test(discovery_data)
        logging.info("U-test completed successfully.")
        logging.info("U-test results shape: %s", u_test_results.shape)

        # Summary of significant results
        significant_uncorrected = (u_test_results["p_value"] < 0.05).sum()
        total_tests = len(u_test_results)
        logging.info(
            "Significant results (uncorrected): %d/%d",
            significant_uncorrected,
            total_tests,
        )

        if "p_value_FDR_corrected" in u_test_results.columns:
            significant_corrected = (
                u_test_results["p_value_FDR_corrected"] < 0.05
            ).sum()
            logging.info(
                "Significant results (FDR-corrected): %d/%d",
                significant_corrected,
                total_tests,
            )

    except Exception as e:
        logging.error("Error in U-test: %s", str(e))
        raise

    # Save results

    assumption_save_path = Path(
        results_path,
        f"assumption_test_results_{modality}.csv",
    )
    assumption_results.to_csv(
        assumption_save_path,
        index=False,
    )
    logging.info(
        "Assumption test results saved to: %s",
        assumption_save_path,
    )

    # Save U-test results
    u_test_save_path = Path(
        results_path,
        f"u_test_results_{modality}.csv",
    )
    u_test_results.to_csv(
        u_test_save_path,
        index=False,
    )
    logging.info(
        "U-test results saved to: %s",
        u_test_save_path,
    )

    logging.info(
        "Statistical testing completed successfully for %s",
        modality,
    )


if __name__ == "__main__":
    # Test the function
    perform_stat_tests(
        version_name="test",
        modality="cortical_thickness",
    )

from pathlib import Path

import pandas as pd

features_set = [
    "cortical_thickness",
    "cortical_volume",
    "rsfmri",
]

tune_resuls_path = Path(
    "data",
    "tune_results",
    "VAE",
)

for feature in features_set:

    VAE_volume_results = Path(
        tune_resuls_path,
        f"VAE_{feature}_UCL_hyper_tune_results.csv",
    )

    VAE_volume_results = pd.read_csv(
        VAE_volume_results,
        index_col=0,
        low_memory=False,
    )

    # Find the index of the row with the maximum average_separation
    max_average_separation_index = VAE_volume_results["average_separation"].idxmax()

    # Retrieve the corresponding config value

    max_separation = VAE_volume_results.loc[
        max_average_separation_index, "average_separation"
    ]

    best_config = VAE_volume_results.loc[max_average_separation_index, "config"]

    print(f"Results for {feature} features:")

    print("\n")

    print(
        f"The best config is {best_config} with average separation of {max_separation}"
    )

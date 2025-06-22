from pathlib import Path

import pandas as pd
from stepmix.stepmix import StepMix

wave = "baseline_year_1_arm_1"
version_name: str = "test"

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
        f"mri_all_features_post_combat_rescaled-{wave}.csv",
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

n_classes = 4

# Fit basic LCA model (no continuous latent traits)
model = StepMix(
    n_components=n_classes,
    measurement="categorical",
    # n_steps=3,
    verbose=1,
    random_state=42,
)

model.fit(data)

# probs = model.predict_proba(data)

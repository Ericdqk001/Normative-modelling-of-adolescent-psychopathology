import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess(
    wave: str = "baseline_year_1_arm_1",
    version_name: str = "test",
):
    # %%
    logging.info("-----------------------")
    logging.info("Processing wave: %s", wave)

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

    data_path = Path(
        analysis_root_path,
        "data",
    )

    if not processed_data_path.exists():
        processed_data_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    core_data_path = Path(
        data_store_path,
        "data",
        "abcd",
        "release5.1",
        "core",
    )

    imaging_path = Path(
        core_data_path,
        "imaging",
    )

    general_info_path = Path(
        core_data_path,
        "abcd-general",
    )

    # For biological sex (demo_sex_v2)
    demographics_path = Path(
        general_info_path,
        "abcd_p_demo.csv",
    )

    demographics = pd.read_csv(
        demographics_path,
        index_col=0,
        low_memory=False,
    )

    # Select the baseline year 1 demographics data
    demographics_bl = demographics[demographics.eventname == "baseline_year_1_arm_1"]

    demographics_bl.demo_sex_v2.value_counts()

    inter_sex_subs = demographics_bl[demographics_bl.demo_sex_v2 == 3].index

    # Recommended image inclusion (NDA 4.0 abcd_imgincl01)
    mri_y_qc_incl_path = Path(
        imaging_path,
        "mri_y_qc_incl.csv",
    )

    mri_y_qc_incl = pd.read_csv(
        mri_y_qc_incl_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_qc_incl = mri_y_qc_incl[mri_y_qc_incl.eventname == wave]

    logging.info(
        "Sample size with MRI recommended inclusion %d", mri_y_qc_incl.shape[0]
    )

    # Remove subjects with intersex from the imaging data
    mri_y_qc_incl = mri_y_qc_incl[~mri_y_qc_incl.index.isin(inter_sex_subs)]

    logging.info(
        "Remove intersex subjects from the imaging data, number = %d",
        len(inter_sex_subs),
    )

    ### Remove imaging data with data quality issues, overall MRI clinical report is used
    # here as well.

    # First, we apply quality control to T1 weighted images (for structural features).
    # Conditions for inclusion:
    # 1. T1w data recommended for inclusion (YES)
    # 2. dmri data recommended for inclusion (YES)
    # 3. Overall MRI clinical report score < 3, which excludes subjects with neurological issues.

    logging.info("Quality Control Criteria:")
    logging.info("T1 data recommended for inclusion = 1")
    logging.info("dMRI data recommended for inclusion = 1")
    logging.info("Overall MRI clinical report score < 3")

    mri_clin_report_path = Path(
        imaging_path,
        "mri_y_qc_clfind.csv",
    )

    mri_clin_report = pd.read_csv(
        mri_clin_report_path,
        index_col=0,
        low_memory=False,
    )

    mri_clin_report_bl = mri_clin_report[mri_clin_report.eventname == wave]

    qc_passed_indices = list(
        mri_y_qc_incl[
            (mri_y_qc_incl.imgincl_t1w_include == 1)
            & (mri_y_qc_incl.imgincl_dmri_include == 1)
        ].index
    )

    qc_passed_mask = mri_clin_report_bl.index.isin(qc_passed_indices)

    logging.info(
        "Sample size after QC passed, number = %d",
        mri_clin_report_bl[qc_passed_mask].shape[0],
    )

    score_mask = mri_clin_report_bl.mrif_score < 3

    subs_pass = mri_clin_report_bl[qc_passed_mask & score_mask]

    logging.info(
        "sample size after QC passed and clinical report (score < 3), number = %d",
        subs_pass.shape[0],
    )

    ### Now prepare the imaging data

    mri_y_smr_thk_dst_path = Path(
        imaging_path,
        "mri_y_smr_thk_dsk.csv",
    )

    mri_y_smr_thk_dst = pd.read_csv(
        mri_y_smr_thk_dst_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_smr_vol_dst_path = Path(
        imaging_path,
        "mri_y_smr_vol_dsk.csv",
    )

    mri_y_smr_vol_dst = pd.read_csv(
        mri_y_smr_vol_dst_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_smr_area_dst_path = Path(
        imaging_path,
        "mri_y_smr_area_dsk.csv",
    )

    mri_y_smr_area_dst = pd.read_csv(
        mri_y_smr_area_dst_path,
        index_col=0,
        low_memory=False,
    )

    mir_y_smr_vol_aseg_path = Path(
        imaging_path,
        "mri_y_smr_vol_aseg.csv",
    )

    mri_y_smr_vol_aseg = pd.read_csv(
        mir_y_smr_vol_aseg_path,
        index_col=0,
        low_memory=False,
    )

    # Select the data for the subjects who passed the quality control and drop
    # subjects with missing data

    # Cortical thickness data
    mri_y_smr_thk_dst = mri_y_smr_thk_dst[mri_y_smr_thk_dst.eventname == wave]

    logging.info(
        "Sample size with T1w cortical thickness data, number = %d",
        mri_y_smr_thk_dst.shape[0],
    )

    t1w_cortical_thickness_pass = mri_y_smr_thk_dst[
        mri_y_smr_thk_dst.index.isin(subs_pass.index)
    ].dropna()

    logging.info(
        "Sample size with complete CT data after QC, number = %d",
        t1w_cortical_thickness_pass.shape[0],
    )

    # Cortical volume data
    mri_y_smr_vol_dst = mri_y_smr_vol_dst[mri_y_smr_vol_dst.eventname == wave]

    logging.info(
        "Sample size with T1w cortical volume data, number = %d",
        mri_y_smr_vol_dst.shape[0],
    )

    t1w_cortical_volume_pass = mri_y_smr_vol_dst[
        mri_y_smr_vol_dst.index.isin(subs_pass.index)
    ].dropna()

    logging.info(
        "Sample size with complete CV data after QC, number = %d",
        t1w_cortical_volume_pass.shape[0],
    )

    # Cortical surface area data

    mri_y_smr_area_dst = mri_y_smr_area_dst[mri_y_smr_area_dst.eventname == wave]

    logging.info(
        "Sample size with T1w cortical surface area data, number = %d",
        mri_y_smr_area_dst.shape[0],
    )

    t1w_cortical_surface_area_pass = mri_y_smr_area_dst[
        mri_y_smr_area_dst.index.isin(subs_pass.index)
    ].dropna()

    logging.info(
        "Sample size with complete SA data after QC, number = %d",
        t1w_cortical_surface_area_pass.shape[0],
    )

    # Subcortical volume

    t1w_subcortical_volume = mri_y_smr_vol_aseg[mri_y_smr_vol_aseg.eventname == wave]

    logging.info(
        "Sample size with T1w subcortical volume data, number = %d",
        t1w_subcortical_volume.shape[0],
    )

    t1w_subcortical_volume_pass = t1w_subcortical_volume[
        t1w_subcortical_volume.index.isin(subs_pass.index)
    ]

    # NOTE: These columns were dropped because they had all missing values or all zeros

    subcortical_all_zeros_cols = [
        "smri_vol_scs_lesionlh",
        "smri_vol_scs_lesionrh",
        "smri_vol_scs_wmhintlh",
        "smri_vol_scs_wmhintrh",
    ]

    t1w_subcortical_volume_pass = t1w_subcortical_volume_pass.drop(
        columns=subcortical_all_zeros_cols
    ).dropna()

    logging.info("Subcortical all zeros columns dropped")
    logging.info("Column names: %s", subcortical_all_zeros_cols)

    logging.info(
        "Sample size with complete subcortical volume data after QC, number = %d",
        t1w_subcortical_volume_pass.shape[0],
    )

    # Combine all the modalities
    mri_all_features = pd.concat(
        [
            t1w_cortical_thickness_pass,
            t1w_cortical_volume_pass,
            t1w_cortical_surface_area_pass,
            t1w_subcortical_volume_pass,
        ],
        axis=1,
    )

    logging.info(
        "Sample size with all imaging features, number = %d", mri_all_features.shape[0]
    )

    # Drop eventname column
    mri_all_features = mri_all_features.drop(columns="eventname")

    ### Add covariates to be considered in the analysis

    logging.info("Adding covariates to the imaging features")

    # For site information (imaging device ID)
    mri_y_adm_info_path = Path(
        imaging_path,
        "mri_y_adm_info.csv",
    )

    mri_y_adm_info = pd.read_csv(
        mri_y_adm_info_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_adm_info = mri_y_adm_info[mri_y_adm_info.eventname == wave]

    le = LabelEncoder()

    # Using .fit_transform function to fit label
    # encoder and return encoded label
    label = le.fit_transform(mri_y_adm_info["mri_info_deviceserialnumber"])

    logging.info("Add covariate: mri_info_deviceserialnumber")

    mri_y_adm_info["img_device_label"] = label

    logging.info(
        "Using LabelEncoder to encode the imaging device ID is error-free, Checked"
    )

    # For interview_age (in months)
    abcd_y_lt_path = Path(
        general_info_path,
        "abcd_y_lt.csv",
    )

    abcd_y_lt = pd.read_csv(
        abcd_y_lt_path,
        index_col=0,
        low_memory=False,
    )

    abcd_y_lt = abcd_y_lt[abcd_y_lt.eventname == wave]

    # Add family ID

    genetics_path = Path(
        core_data_path,
        "genetics",
    )

    genetics_relatedness_path = Path(
        genetics_path,
        "gen_y_pihat.csv",
    )

    genetics_relatedness = pd.read_csv(
        genetics_relatedness_path,
        index_col=0,
        low_memory=False,
    )

    family_id = genetics_relatedness["rel_family_id"]

    # Add household income

    household_income = demographics_bl["demo_comb_income_v2"].copy()

    # Not available category (777:refused to answer, 999: don't know, missing values)

    household_income = household_income.replace(
        [777, 999],
        np.nan,
    )

    print(
        "Subjects who either refused to answer or don't know their income are set to NA"
    )

    series_list = [
        demographics_bl.demo_sex_v2,
        mri_y_adm_info.img_device_label,
        abcd_y_lt.interview_age,
        family_id,
        household_income,
    ]

    covariates = pd.concat(series_list, axis=1).dropna()

    # Join the covariates to the brain features

    mri_all_features_cov = mri_all_features.join(
        covariates,
        how="left",
    ).dropna()

    logging.info(
        "Sample size with all imaging features and covariates, number = %d",
        mri_all_features_cov.shape[0],
    )

    # %%
    # Joining CBCL scales

    mental_health_path = Path(
        core_data_path,
        "mental-health",
    )

    cbcl_path = Path(
        mental_health_path,
        "mh_p_cbcl.csv",
    )

    cbcl_df = pd.read_csv(
        cbcl_path,
        index_col=0,
        low_memory=False,
    )

    cbcl_df = cbcl_df[cbcl_df["eventname"] == wave]

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

    logging.info("CBCL variable names: %s", cbcl_variable_names)

    # Select the CBCL variables of interest
    cbcl_scales = cbcl_df[cbcl_variable_names]

    cbcl_scales = cbcl_scales.dropna()

    logging.info(
        "Sample size with CBCL scales, number = %d",
        cbcl_scales.shape[0],
    )

    # Create binary variables for the CBCL scales for LCA analysis

    cbcl_binary_scales = cbcl_scales.copy()

    cbcl_binary_scales = (cbcl_binary_scales >= 65).astype(int)

    # Add one here because LCA expects 1/2 rather than 0/1

    cbcl_binary_scales += 1

    # Join the CBCL scales to the imaging features and covariates

    mri_all_features_cov_cbcl = mri_all_features_cov.join(
        cbcl_binary_scales,
        how="left",
    ).dropna()

    logging.info(
        "Sample size with all imaging features, covariates and CBCL scales, number = %d",
        mri_all_features_cov_cbcl.shape[0],
    )

    # %%

    # %%
    # %% Keep unrelated subjects

    seed = 42

    logging.info("Keeping unrelated subjects, random seed = %d", seed)

    mri_all_features_cov_cbcl_unrelated = mri_all_features_cov_cbcl.loc[
        mri_all_features_cov_cbcl.groupby(["rel_family_id"]).apply(
            lambda x: x.sample(n=1, random_state=seed).index[0]
        ),
    ]

    logging.info("Keeping unrelated subjects is error-free, Checked")

    logging.info(
        "Sample size after keeping unrelated subjects, number = %d",
        mri_all_features_cov_cbcl_unrelated.shape[0],
    )
    # %%

    # Average the bilateral features

    lh_columns = [
        col for col in mri_all_features_cov_cbcl_unrelated.columns if col.endswith("lh")
    ]

    logging.info("Number of left hemisphere features: %d", len(lh_columns))

    rh_columns = [
        col for col in mri_all_features_cov_cbcl_unrelated.columns if col.endswith("rh")
    ]

    logging.info("Number of right hemisphere features: %d", len(rh_columns))

    # Check if the two lists match by region (assuming 'lh'/'rh' are prefixes)
    lh_regions = [col.replace("lh", "") for col in lh_columns]
    rh_regions = [col.replace("rh", "") for col in rh_columns]

    if lh_regions == rh_regions:
        logging.info("Left and right hemisphere feature lists MATCH by region.")

    else:
        logging.warning(
            "Left and right hemisphere feature lists DO NOT MATCH by region."
        )

    # Identify all other columns (covariates, unilateral features, PRS.)
    other_columns = [
        col
        for col in mri_all_features_cov_cbcl_unrelated.columns
        if col not in lh_columns + rh_columns
    ]

    def average_hemisphere_columns(df, lh_columns, rh_columns, other_columns):
        avg_cols = {
            f"img_{lh.replace('lh', '')}": (df[lh] + df[rh]) / 2
            for lh, rh in zip(lh_columns, rh_columns)
        }
        other_cols = df[other_columns]
        return pd.concat([pd.DataFrame(avg_cols), other_cols], axis=1)

    mri_all_features_cov_cbcl_unrelated_avg = average_hemisphere_columns(
        mri_all_features_cov_cbcl_unrelated,
        lh_columns,
        rh_columns,
        other_columns,
    )

    logging.info("Averaging bilateral features is error-free, Checked")

    # %%
    # Create features of interest list

    logging.info("Cortical thickness global features:")
    t1w_cortical_thickness_rois = list(t1w_cortical_thickness_pass.columns[1:-3])
    logging.info("%s", list(t1w_cortical_thickness_pass.columns[-3:]))

    # For cortical volume

    logging.info("Cortical volume global features:")
    t1w_cortical_volume_rois = list(t1w_cortical_volume_pass.columns[1:-3])
    logging.info("%s", list(t1w_cortical_volume_pass.columns[-3:]))

    # For surface area

    logging.info("Cortical surface area global features:")
    t1w_cortical_surface_area_rois = list(t1w_cortical_surface_area_pass.columns[1:-3])
    logging.info("%s", list(t1w_cortical_surface_area_pass.columns[-3:]))

    global_subcortical_features = [
        "smri_vol_scs_csf",
        "smri_vol_scs_wholeb",
        "smri_vol_scs_intracranialv",
        "smri_vol_scs_latventricles",
        "smri_vol_scs_allventricles",
        "smri_vol_scs_subcorticalgv",
        "smri_vol_scs_suprateialv",
        "smri_vol_scs_wmhint",
    ]

    logging.info("Subcortical volume global features:")
    logging.info("%s", global_subcortical_features)

    t1w_subcortical_volume_rois = [
        col
        for col in t1w_subcortical_volume_pass.columns
        if col not in global_subcortical_features and col != "eventname"
    ]

    def get_bilateral_and_unilateral_features(feature_list):
        """Returns bilateral and unilateral features from a list of features."""
        lh_roots = {f[:-2] for f in feature_list if f.endswith("lh")}
        rh_roots = {f[:-2] for f in feature_list if f.endswith("rh")}
        bilateral_roots = sorted(lh_roots & rh_roots)

        # Unilateral = present in only one hemisphere or has no suffix
        unilateral_features = [
            f for f in feature_list if (not f.endswith("lh") and not f.endswith("rh"))
        ]

        # Add prefix "img_" to bilateral roots
        bilateral_roots = [f"img_{root}" for root in bilateral_roots]

        return bilateral_roots, unilateral_features

    features_of_interest = {
        "bilateral_cortical_thickness": get_bilateral_and_unilateral_features(
            t1w_cortical_thickness_rois
        )[0],
        "bilateral_cortical_volume": get_bilateral_and_unilateral_features(
            t1w_cortical_volume_rois
        )[0],
        "bilateral_cortical_surface_area": get_bilateral_and_unilateral_features(
            t1w_cortical_surface_area_rois
        )[0],
        "bilateral_subcortical_volume": get_bilateral_and_unilateral_features(
            t1w_subcortical_volume_rois
        )[0],
        # Unilateral features are for performing GLM
        "unilateral_subcortical_features": get_bilateral_and_unilateral_features(
            t1w_subcortical_volume_rois
        )[1],
    }

    logging.info("Number of features for each modality:")
    for modality, features in features_of_interest.items():
        logging.info(f"{modality}: {len(features)} features")

    features_for_repeated_effects_path = Path(
        processed_data_path,
        "features_of_interest.json",
    )

    with open(features_for_repeated_effects_path, "w") as f:
        json.dump(features_of_interest, f)

    # %%
    # Apply neuroCombat to remove site effects

    imaging_feature_list = (
        features_of_interest["bilateral_cortical_thickness"]
        + features_of_interest["bilateral_cortical_volume"]
        + features_of_interest["bilateral_cortical_surface_area"]
        + features_of_interest["bilateral_subcortical_volume"]
        + features_of_interest["unilateral_subcortical_features"]
        + ["smri_thick_cdk_mean"]
        + ["smri_vol_scs_intracranialv"]
        + ["smri_area_cdk_total"]
    )

    other_feature_list = list(covariates.columns) + list(cbcl_binary_scales.columns)

    features_combat = neuroCombat(
        dat=np.array(mri_all_features_cov_cbcl_unrelated_avg[imaging_feature_list]).T,
        covars=mri_all_features_cov_cbcl_unrelated_avg[other_feature_list],
        batch_col="img_device_label",
        categorical_cols=["demo_sex_v2"],
        continuous_cols=["interview_age"],
    )["data"]

    features_post_combat = pd.DataFrame(
        data=features_combat.T, columns=imaging_feature_list
    ).set_index(mri_all_features_cov_cbcl_unrelated_avg.index)

    features_cov_post_combat = pd.concat(
        [
            features_post_combat,
            mri_all_features_cov_cbcl_unrelated_avg[other_feature_list],
        ],
        axis=1,
    )

    # %%
    # Standardize the continuous variables

    logging.info("Standardizing the continuous variables")

    categorical_variables = [
        "demo_sex_v2",
        "img_device_label",
        "rel_family_id",
        "demo_comb_income_v2",
    ] + list(cbcl_binary_scales.columns)

    for col in categorical_variables:
        if col in features_cov_post_combat.columns:
            features_cov_post_combat[col] = features_cov_post_combat[col].astype(
                "category"
            )

    logging.info("Make sure the following columns are categorical: ")
    logging.info(", ".join(categorical_variables))

    logging.info("Excluding the following columns from standardisation: ")
    logging.info(", ".join(categorical_variables))

    # Get columns to scale (everything else)
    cols_to_scale = [
        col
        for col in features_cov_post_combat.columns
        if col not in categorical_variables
    ]

    # Standardize selected columns
    scaler = StandardScaler()

    features_cov_post_combat[cols_to_scale] = scaler.fit_transform(
        features_cov_post_combat[cols_to_scale]
    )

    logging.info("Standardization of continuous variables is error-free, Checked")

    rescaled_features_post_combat = features_post_combat.copy()

    # This is for performing GLM (for unilateral features)
    rescaled_features_post_combat.to_csv(
        Path(
            processed_data_path,
            f"mri_all_features_post_combat_rescaled-{wave}.csv",
        ),
        index=True,
    )

    logging.info("Rescaled imaging features with PRS saved to CSV")

    logging.info(
        "Final Sample size for wave:%s %d",
        wave,
        rescaled_features_post_combat.shape[0],
    )


if __name__ == "__main__":
    all_img_waves = [
        "baseline_year_1_arm_1",
        # "2_year_follow_up_y_arm_1",
        # "4_year_follow_up_y_arm_1",
    ]

    # Process all waves
    for wave in all_img_waves:
        preprocess(wave=wave)

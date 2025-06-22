# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from sklearn.preprocessing import LabelEncoder
from statsmodels.formula.api import ols

from preprocess.scripts.split import split


def deconfound_image_exc_sex():
    # %%
    processed_data_path = Path(
        "data",
        "processed_data",
    )

    LCA_path = Path(
        "data",
        "LCA",
    )

    core_data_path = Path(
        "data",
        "raw_data",
        "core",
    )

    general_info_path = Path(
        core_data_path,
        "abcd-general",
    )

    demographics_path = Path(
        general_info_path,
        "abcd_p_demo.csv",
    )

    demographics = pd.read_csv(
        demographics_path,
        index_col=0,
        low_memory=False,
    )

    demographics_bl = demographics[demographics.eventname == "baseline_year_1_arm_1"]

    family_income = demographics_bl["demo_comb_income_v2"].copy()

    family_income = family_income.replace(777, 999)

    imaging_path = Path(
        core_data_path,
        "imaging",
    )

    # Load cbcl-LCA data
    cbcl_LCA_path = Path(
        LCA_path,
        "cbcl_class_member_prob.csv",
    )

    cbcl_LCA = pd.read_csv(
        cbcl_LCA_path,
        index_col=0,
        low_memory=False,
    )

    # %%
    ### Perform neuroCombat to harmonize the imaging data

    # from neuroCombat import neuroCombat

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

    abcd_y_lt_bl = abcd_y_lt[abcd_y_lt.eventname == "baseline_year_1_arm_1"]

    # For biological sex (demo_sex_v2)

    demographics_bl.demo_sex_v2.value_counts()

    # For site information
    mri_y_adm_info_path = Path(
        imaging_path,
        "mri_y_adm_info.csv",
    )

    mri_y_adm_info = pd.read_csv(
        mri_y_adm_info_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_adm_info_bl = mri_y_adm_info[
        mri_y_adm_info.eventname == "baseline_year_1_arm_1"
    ]

    # mri_y_adm_info_bl.mri_info_deviceserialnumber.unique()

    le = LabelEncoder()

    # Using .fit_transform function to fit label
    # encoder and return encoded label
    label = le.fit_transform(mri_y_adm_info_bl["mri_info_deviceserialnumber"])
    mri_y_adm_info_bl["label_site"] = label

    # For smri_vol_scs_intracranialv (intracranial volume)
    mri_y_smr_vol_aseg_path = Path(
        imaging_path,
        "mri_y_smr_vol_aseg.csv",
    )

    mri_y_smr_vol_aseg = pd.read_csv(
        mri_y_smr_vol_aseg_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_smr_vol_aseg_bl = mri_y_smr_vol_aseg[
        mri_y_smr_vol_aseg.eventname == "baseline_year_1_arm_1"
    ]

    # mri_y_smr_vol_aseg_bl.smri_vol_scs_intracranialv.describe()

    # Combine covariates together (interview_age, intra cranial volume, and site)
    # List of covariates to combine
    series_list = [
        demographics_bl.demo_sex_v2,
        cbcl_LCA.predicted_class,
        mri_y_adm_info_bl.label_site,
        mri_y_smr_vol_aseg_bl.smri_vol_scs_intracranialv,
        abcd_y_lt_bl.interview_age,
        family_income,
    ]

    covariates = pd.concat(series_list, axis=1)

    # %%
    ### Add the covariates to imaging data

    # Cortical thickness
    t1w_cortical_thickness_bl_pass_path = Path(
        processed_data_path,
        "t1w_cortical_thickness_bl_pass.csv",
    )

    t1w_cortical_thickness_bl_pass = pd.read_csv(
        t1w_cortical_thickness_bl_pass_path,
        index_col=0,
        low_memory=False,
    )

    # Cortical volume
    t1w_cortical_volume_bl_pass_path = Path(
        processed_data_path,
        "t1w_cortical_volume_bl_pass.csv",
    )

    t1w_cortical_volume_bl_pass = pd.read_csv(
        t1w_cortical_volume_bl_pass_path,
        index_col=0,
        low_memory=False,
    )

    t1w_cortical_volume_bl_pass = t1w_cortical_volume_bl_pass.drop(
        columns=["eventname"]
    )

    # Cortical surface area

    t1w_cortical_surface_area_bl_pass_path = Path(
        processed_data_path,
        "t1w_cortical_surface_area_bl_pass.csv",
    )

    t1w_cortical_surface_area_bl_pass = pd.read_csv(
        t1w_cortical_surface_area_bl_pass_path,
        index_col=0,
        low_memory=False,
    )

    t1w_cortical_surface_area_bl_pass = t1w_cortical_surface_area_bl_pass.drop(
        columns=["eventname"]
    )

    # Join the covariates to the cortical features (no missing data here)

    t1w_cortical_features_bl_pass = (
        t1w_cortical_thickness_bl_pass.join(
            t1w_cortical_volume_bl_pass,
            how="left",
        )
        .join(
            t1w_cortical_surface_area_bl_pass,
            how="left",
        )
        .join(
            covariates,
            how="left",
        )
    ).dropna()
    # %%

    gordon_cor_subcortical_bl_pass_path = Path(
        processed_data_path,
        "gordon_cor_subcortical_bl_pass.csv",
    )

    gordon_cor_subcortical_bl_pass = pd.read_csv(
        gordon_cor_subcortical_bl_pass_path,
        index_col=0,
        low_memory=False,
    )

    # Join the covariates to the rs-fMRI data
    gordon_cor_subcortical_bl_pass = gordon_cor_subcortical_bl_pass.join(
        covariates, how="left"
    ).dropna()

    # %%
    ### Apply NeuroCombat to the imaging data

    # NeuroCombat for cortical thickness
    brain_features_of_interest_path = Path(
        processed_data_path,
        "brain_features_of_interest.json",
    )

    with open(brain_features_of_interest_path, "r") as f:
        brain_features_of_interest = json.load(f)

    t1w_cortical_thickness_rois = brain_features_of_interest[
        "t1w_cortical_thickness_rois"
    ]

    t1w_cortical_volume_rois = brain_features_of_interest["t1w_cortical_volume_rois"]

    t1w_cortical_surface_area_rois = brain_features_of_interest[
        "t1w_cortical_surface_area_rois"
    ]

    # Add intracranial volume here to be combated as well as other brain features
    t1w_brain_features_list = (
        t1w_cortical_thickness_rois
        + t1w_cortical_volume_rois
        + t1w_cortical_surface_area_rois
        + ["smri_vol_scs_intracranialv"]
    )

    t1w_cortical_features_list = (
        t1w_cortical_thickness_rois
        + t1w_cortical_volume_rois
        + t1w_cortical_surface_area_rois
    )

    t1w_cortical_features_covariates = t1w_cortical_features_bl_pass[
        covariates.columns
    ].drop(
        "smri_vol_scs_intracranialv",
        axis=1,
    )

    cortical_features_combat = neuroCombat(
        dat=np.array(t1w_cortical_features_bl_pass[t1w_brain_features_list]).T,
        covars=t1w_cortical_features_covariates,
        batch_col="label_site",
        categorical_cols=["demo_sex_v2"],
        continuous_cols=["interview_age"],
    )["data"]

    cortical_features_post_combat = pd.DataFrame(
        data=cortical_features_combat.T, columns=t1w_brain_features_list
    ).set_index(t1w_cortical_features_bl_pass.index)

    # Drop intracranial volume because we have the post-combat ones now
    covariates_no_intracranialv = covariates.drop(
        "smri_vol_scs_intracranialv",
        axis=1,
    )

    cortical_features_post_combat_covar = cortical_features_post_combat.join(
        covariates_no_intracranialv,
        how="left",
    )

    # NeuroCombat for rs-fMRI
    gordon_net_subcor_no_dup = brain_features_of_interest["gordon_net_subcor_no_dup"]

    rsfmri_brain_features = gordon_net_subcor_no_dup + ["smri_vol_scs_intracranialv"]

    rsfmri_covariates_no_intracranialv = gordon_cor_subcortical_bl_pass[
        covariates.columns
    ].drop(
        "smri_vol_scs_intracranialv",
        axis=1,
    )

    rsfmri_combat = neuroCombat(
        dat=np.array(gordon_cor_subcortical_bl_pass[rsfmri_brain_features]).T,
        covars=rsfmri_covariates_no_intracranialv,
        batch_col="label_site",
        categorical_cols=["demo_sex_v2"],
        continuous_cols=["interview_age"],
    )["data"]

    rsfmri_post_combat = pd.DataFrame(
        data=rsfmri_combat.T, columns=rsfmri_brain_features
    ).set_index(gordon_cor_subcortical_bl_pass.index)

    rsfmri_post_combat_covars = rsfmri_post_combat.join(
        covariates_no_intracranialv,
        how="left",
    )

    # Not get the data splits first

    t1w_data_splits, rsfmri_data_splits = split(
        t1w_cortical_features_bl_pass,
        gordon_cor_subcortical_bl_pass,
    )

    # %%
    ### Now regressing out other confounders by sets,
    # train the regression models on the training set and apply to the validation and test sets

    # For cortical features
    # Fit the model on the training set

    cortical_features_post_combat_covar_train = cortical_features_post_combat_covar.loc[
        t1w_data_splits["train"]
    ]

    cortical_features_post_combat_covar_val = cortical_features_post_combat_covar.loc[
        t1w_data_splits["val"]
    ]

    cortical_features_post_combat_covar_test = cortical_features_post_combat_covar.loc[
        t1w_data_splits["total_test"]
    ]

    cortical_features_resid_train_df = pd.DataFrame(
        index=cortical_features_post_combat_covar_train.index,
        columns=t1w_cortical_features_list,
    )

    cortical_features_resid_val_df = pd.DataFrame(
        index=cortical_features_post_combat_covar_val.index,
        columns=t1w_cortical_features_list,
    )

    cortical_features_resid_test_df = pd.DataFrame(
        index=cortical_features_post_combat_covar_test.index,
        columns=t1w_cortical_features_list,
    )

    for img_feature in t1w_cortical_features_list:
        # Define and fit the model
        formula = "%s ~ smri_vol_scs_intracranialv + interview_age" % img_feature
        model = ols(formula, cortical_features_post_combat_covar_train).fit()

        # Calculate residuals for the training set
        cortical_features_resid_train_df[img_feature] = model.resid

        # Predict on the validation set and calculate residuals
        val_predictions = model.predict(cortical_features_post_combat_covar_val)
        cortical_features_resid_val_df[img_feature] = (
            cortical_features_post_combat_covar_val[img_feature] - val_predictions
        )

        # Predict on the test set and calculate residuals
        test_predictions = model.predict(cortical_features_post_combat_covar_test)
        cortical_features_resid_test_df[img_feature] = (
            cortical_features_post_combat_covar_test[img_feature] - test_predictions
        )

    # Combine all residuals into one DataFrame
    cortical_features_resid_df = pd.concat(
        [
            cortical_features_resid_train_df,
            cortical_features_resid_val_df,
            cortical_features_resid_test_df,
        ],
        axis=0,
    ).reindex(cortical_features_post_combat_covar.index)

    cortical_features_resid_df_covars = cortical_features_resid_df.join(
        covariates_no_intracranialv,
        how="left",
    )

    cortical_features_resid_df_covars = cortical_features_resid_df_covars.assign(
        smri_vol_scs_intracranialv=cortical_features_post_combat_covar[
            "smri_vol_scs_intracranialv"
        ]
    )

    # For rs-fMRI

    rsfmri_post_combat_covar_train = rsfmri_post_combat_covars.loc[
        rsfmri_data_splits["train"]
    ]

    rsfmri_post_combat_covar_val = rsfmri_post_combat_covars.loc[
        rsfmri_data_splits["val"]
    ]

    rsfmri_post_combat_covar_test = rsfmri_post_combat_covars.loc[
        rsfmri_data_splits["total_test"]
    ]

    rsfmri_resid_train_df = pd.DataFrame(
        index=rsfmri_post_combat_covar_train.index,
        columns=gordon_net_subcor_no_dup,
    )

    rsfmri_resid_val_df = pd.DataFrame(
        index=rsfmri_post_combat_covar_val.index,
        columns=gordon_net_subcor_no_dup,
    )

    rsfmri_resid_test_df = pd.DataFrame(
        index=rsfmri_post_combat_covar_test.index,
        columns=gordon_net_subcor_no_dup,
    )

    for img_feature in gordon_net_subcor_no_dup:
        formula = "%s ~ smri_vol_scs_intracranialv + interview_age" % img_feature
        model = ols(formula, rsfmri_post_combat_covar_train).fit()

        rsfmri_resid_train_df[img_feature] = model.resid

        val_predictions = model.predict(rsfmri_post_combat_covar_val)
        rsfmri_resid_val_df[img_feature] = (
            rsfmri_post_combat_covar_val[img_feature] - val_predictions
        )

        test_predictions = model.predict(rsfmri_post_combat_covar_test)

        rsfmri_resid_test_df[img_feature] = (
            rsfmri_post_combat_covar_test[img_feature] - test_predictions
        )

    rsfmri_resid_df = pd.concat(
        [
            rsfmri_resid_train_df,
            rsfmri_resid_val_df,
            rsfmri_resid_test_df,
        ],
        axis=0,
    ).reindex(rsfmri_post_combat_covars.index)

    rsfmri_resid_df_covars = rsfmri_resid_df.join(
        covariates_no_intracranialv,
        how="left",
    )

    rsfmri_resid_df_covars = rsfmri_resid_df_covars.assign(
        smri_vol_scs_intracranialv=rsfmri_post_combat_covars[
            "smri_vol_scs_intracranialv"
        ]
    )

    # NOTE the effect of de-confounding is tested in the test folder

    # %%
    ### Save the deconfounded data

    cortical_features_resid_df_covars.to_csv(
        Path(processed_data_path, "t1w_cortical_features_resid_exc_sex.csv"),
        index=True,
    )

    rsfmri_resid_df_covars.to_csv(
        Path(processed_data_path, "gordon_cor_subcortical_resid_exc_sex.csv"),
        index=True,
    )

    all_brain_features_resid = cortical_features_resid_df.merge(
        rsfmri_resid_df, left_index=True, right_index=True, how="outer"
    ).join(
        covariates_no_intracranialv,
        how="inner",
    )

    all_brain_features_resid.to_csv(
        Path(
            processed_data_path,
            "all_brain_features_resid_exc_sex.csv",
        ),
        index=True,
    )


if __name__ == "__main__":
    deconfound_image_exc_sex()

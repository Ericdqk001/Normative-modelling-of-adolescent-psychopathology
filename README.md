# Normative Modelling of Adolescent Psychopathology

This repository contains the complete pipeline for normative modelling of adolescent psychopathology using brain imaging data from the ABCD Study. This work was conducted as part of an MRes research project at University College London. The project uses Variational Autoencoders (VAEs) to learn normative patterns in brain structure and identifies deviations associated with different psychopathology subgroups.

## Overview

The pipeline consists of several key components:

1. **Data Preprocessing**: Quality control, covariate handling, and feature preparation
2. **Latent Class Analysis (LCA)**: Identification of psychopathology subgroups from CBCL data
3. **Normative Modelling**: VAE-based learning of normal brain patterns
4. **Discovery Analysis**: Detection and statistical analysis of brain deviations
5. **Statistical Testing**: Comprehensive analysis with multiple comparison correction

## Requirements

- Python 3.10+
- ABCD Study data access

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ericdqk001/Normative-modelling-of-adolescent-psychopathology.git
cd Normative-modelling-of-adolescent-psychopathology
```

### 2. Install the package

```bash
# Install in development mode
pip install -e .
```

## Data Setup

This project requires ABCD Study data. You'll need:

1. **Neuroimaging data**: Cortical thickness, surface area, volume, and subcortical volume
2. **Behavioral data**: CBCL (Child Behavior Checklist) scores
3. **Demographics**: Age, sex, income, family relationships
4. **Quality control**: MRI quality metrics

### Environment Variables

Configure data paths using environment variables:

```bash
# Required: Path to ABCD data containing release5.1 folder
export ABCD_DATA_ROOT="/path/to/your/abcd/data"

# Required: Path for analysis outputs
export ANALYSIS_ROOT="/path/to/your/analysis/output"
```

**ABCD Data Structure**: Your `ABCD_DATA_ROOT` should contain the `release5.1` folder with the standard ABCD directory structure:

```bash
$ABCD_DATA_ROOT/
└── release5.1/
    └── core/
        ├── imaging/
        ├── mental_health/
        └── ... (other ABCD data folders)
```

## Usage

### 1. Install the package

First, install the package in development mode:

```bash
pip install -e .
```

### 2. Set environment variables

```bash
export ABCD_DATA_ROOT="/path/to/your/abcd/data"
export ANALYSIS_ROOT="/path/to/your/analysis/output"
```

### 3. Run the pipeline

Execute the complete pipeline using Python's module syntax:

```bash
python -m src.main
```

**Note**: The project is structured as a Python package with relative imports. You must either:

- Install with `pip install -e .` and run `python -m src.main`

Do NOT run `python src/main.py` directly as this will cause import errors.

## Pipeline Components

### 1. Data Preprocessing (`src/preprocess/`)

- **prepare_data.py**: Main data preparation with quality control
- **deconfound.py**: Removes age, sex, and income effects
- **split.py**: Creates stratified train/validation/test splits

### 2. Latent Class Analysis (`src/LCA/`)

- **lca_cbcl.py**: Identifies psychopathology subgroups (2-6 classes)
- Includes model selection with BLRT and entropy calculations

### 3. VAE Modelling (`src/modelling/`)

- **models/VAE.py**: Variational Autoencoder implementation
- **train/train.py**: Training pipeline with early stopping

### 4. Discovery Analysis (`src/discover/`)

- **compute_deviations.py**: Calculates brain feature deviations
- **stat_tests.py**: Statistical testing with FDR correction
- **utils.py**: Utility functions for analysis

## Output Files

The pipeline generates several output files:

### Model Files

- `VAE_model_weights_{modality}.pt`: Trained VAE models for each brain modality

### LCA Results

- `lca_model_stats.csv`: Model fit statistics (BIC, AIC, log-likelihood, entropy, BLRT p-values)
- `lca_class_parameters.csv`: Variable-specific probabilities for each class
- `lca_class_member_entropy.csv`: Individual predicted class and entropy scores

### Data Files

- `mri_all_features_post_combat_rescaled.csv`: Main processed imaging features with CBCL scales
- `features_of_interest.json`: Brain features organized by modality
- `imaging_data_splits.json`: Train/validation/test subject splits
- `mri_all_features_post_deconfound.csv`: Deconfounded imaging features

### Discovery Results

- `discovery_data_{modality}.csv`: Brain deviation measurements for each modality

### Statistical Test Results

- `hemisphere_test_results_{modality}.csv`: Bilateral hemisphere comparisons
- `assumption_test_results_{modality}.csv`: Statistical assumption testing
- `u_test_results_{modality}.csv`: Mann-Whitney U test results

### Logs

- `experiment.log`: Detailed execution logs

## Key Features

### Robust Statistical Analysis

- Assumption testing (normality and equal variance)
- Mann-Whitney U tests for non-parametric comparisons
- FDR correction for multiple comparisons
- Hemisphere difference testing for bilateral regions
- Averaging regions across hemisphere without lateral differences in deviations

### Comprehensive Preprocessing

- neuroCombat for site effect removal
- Quality control filtering
- Family relationship handling
- Covariate deconfounding

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
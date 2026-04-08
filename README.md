# NGBoost Pipeline

Predict **mean primary particle size (μm)** from composition & sintering parameters using **NGBoost** with cross‑validation, testing, and SHAP‑based interpretation. No dataset is included—use your own CSVs with the required columns below.&#x20;

## 1) Install

```bash
# Python 3.12.11 (Anaconda), ngboost 0.5.6
conda create -n ngb-sem python=3.12.11 -y
conda activate ngb-sem
pip install -U pip
pip install numpy pandas scikit-learn ngboost==0.5.6 shap matplotlib seaborn joblib matimpute
python -c "import sys, ngboost; print(sys.version); print(ngboost.__version__)"

```

## 2) Prepare your data

Place CSVs under `./dataset/`:

* **Train input**: `Li-rich data collection_full.csv` (or `_half.csv`)
* **(Optional) Test input**: `Li-rich_test_full.csv` (or `_half.csv`)
  Required columns (exact names):
  `Li_fraction_in_TM_layer, Ni_fraction, Co_fraction, Mn_fraction, first_sintering_temperature, first_sintering_time, second_sintering_temperature, second_sintering_time, mean_primary_particle_size, doi`. The imputer expects and preserves `doi`.&#x20;

## 3) Impute missing values

```bash
# include rows even if target value is missing
python imputer.py --dataset full

# exclude rows with missing target value
python imputer.py --dataset half
```

This writes **four** training CSVs (Mat/Mean/KNN/MICE). Note: paths in `imputer.py` use Windows style; adjust to POSIX if needed.&#x20;

> **Rename** the imputed files so training can find them:
> `Li-rich_train_<SUFFIX>_<DATASET_TYPE>.csv` where `<SUFFIX> ∈ {MatImputer, KNN Imputer, MICE Imputer, Mean Imputer}` and `<DATASET_TYPE> ∈ {full, half}`.&#x20;

## 4) (Optional) Make a 143‑row sample set

```bash
python 143_selection.py
# outputs to ./sampled_sets/*_sample143.csv
```

Then run with `--dataset_dir ./sampled_sets --dataset_type full_sample143`.

## 5) Train / Test

```bash
# Train models for all 4 imputation variants
python main.py --dataset_dir ./dataset --dataset_type full --mode train

# Test saved models on a test CSV named Li-rich_test_full.csv (or _half/_full_sample143)
python main.py --dataset_dir ./dataset --dataset_type full --mode test
```

Training uses features above, a `PowerTransformer`, and **NGBRegressor(Normal)** with 10‑fold CV; artifacts go to `models/models_<dataset_type>/` and plots to `./plots/`. Test computes metrics and calibration curves if the test CSV includes the target.&#x20;

## 6) Interpret (SHAP)

```bash
# after training on 'full' or 'half'
python model_interpretation.py --dataset_dir ./dataset --dataset_type full
```

Generates feature importance, SHAP summaries, dependence, interactions, and extra plots under `plots/shap/`.&#x20;

## Outputs (where to look)

* **Models & scalers**: `models/models_<dataset_type>/`
* **Training/validation plots**: parity, metrics, loss curves, fold/final uncertainty; **test**: uncertainty heatmap + combined calibration curves.

**Notes**

* Expected train file names: `Li-rich_train_<SUFFIX>_<dataset_type>.csv`. Expected test name: `Li-rich_test_<dataset_type>.csv`.&#x20;
* `main.py` options: `--dataset_dir`, `--dataset_type {full, half, full_sample143}`, `--mode {train, test, both}`.&#x20;
* 
 ## Use a trained MatImputer model to predict on a new experimental data CSV:

```bash
import os
import joblib
import pandas as pd

dataset_type = "full"
model_dir    = f"models/models_{dataset_type}"
model_path   = os.path.join(model_dir, "best_ngb_default_model_MatImputer.pkl")
scaler_path  = os.path.join(model_dir, "scaler_MatImputer.pkl")
test_csv     = "experimental_test.csv"                 # your SEM experiment data
out_csv      = "experimental_test_with_predictions.csv"

ngb_model = joblib.load(model_path)
scaler    = joblib.load(scaler_path)

df_test = pd.read_csv(test_csv)
features = [
    'Li_fraction_in_TM_layer', 'Ni_fraction', 'Co_fraction', 'Mn_fraction',
    'first_sintering_temperature', 'first_sintering_time',
    'second_sintering_temperature', 'second_sintering_time'
]
X_test        = df_test[features]
X_test_scaled = scaler.transform(X_test)
y_dist        = ngb_model.pred_dist(X_test_scaled)

df_test['predicted_mean'] = y_dist.loc
df_test['predicted_std']  = y_dist.scale
df_test['lower_95']       = y_dist.ppf(0.025)
df_test['upper_95']       = y_dist.ppf(0.975)

print(df_test[['predicted_mean','predicted_std','lower_95','upper_95']].head())
df_test.to_csv(out_csv, index=False)
print(f"\nSaved predictions to {out_csv}")
```
** This work is published in Advanced Sciene **

Uncertainty‐Quantified Primary Particle Size Prediction in Li‐Rich NCM Materials via Machine Learning and Chemistry‐Aware Imputation (https://doi.org/10.1002/advs.202515694)


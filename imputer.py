import os
import argparse
import pandas as pd
import warnings
import numpy as np
import random

# Set a fixed PYTHONHASHSEED environment variable for consistent hashing behavior.
# Note: This must be set before any libraries that use hashing are imported.
os.environ["PYTHONHASHSEED"] = "42"

# Global random seed
seed = 42
np.random.seed(seed)
random.seed(seed)

from matimpute import MatImputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

def restore_merge_save(imputed_df, filename, df_impute_reset, non_impute_cols_all, df_impute):
    imputed_df['orig_order'] = df_impute_reset['orig_order']
    imputed_df = imputed_df.sort_values('orig_order').drop(columns='orig_order')
    
    if len(imputed_df) != len(df_impute_reset):
        raise ValueError(f"Row count mismatch in {filename} imputation!")
    

    imputed_df.index = df_impute.index
    imputed_df = imputed_df.join(non_impute_cols_all)
    

    np.random.seed(seed)
    random.seed(seed)
    imputed_df = imputed_df.sample(frac=1, random_state=seed)
    imputed_df.index.name = 'index'
    
    imputed_df.to_csv(filename, index=True)

def main(args):
    if args.dataset.lower() == "full":
        input_file = r".\dataset\Li-rich data collection_full.csv"
    elif args.dataset.lower() == "half":
        input_file = r".\dataset\Li-rich data collection_half.csv"
    else:
        raise ValueError("Dataset must be either 'full' or 'half'.")

    df = pd.read_csv(input_file, low_memory=False, encoding="utf-8-sig")
    if 'index' not in df.columns:
        df = df.reset_index()
    df = df.set_index('index')
    
    non_impute_cols = df[['doi']].copy()
    df.drop(['doi'], axis=1, inplace=True)
    
    cols_for_impute = [
        'Li_fraction_in_TM_layer', 'Ni_fraction', 'Co_fraction', 'Mn_fraction', 
        'first_sintering_temperature', 'first_sintering_time', 
        'second_sintering_temperature', 'second_sintering_time', 'mean_primary_particle_size'
    ]
    
    df_impute = df[cols_for_impute].copy()
    non_impute_cols_all = non_impute_cols.copy()
    
    df_impute_reset = df_impute.reset_index(drop=True)
    df_impute_reset['orig_order'] = range(len(df_impute_reset))
    
    dataset_suffix = args.dataset.lower()

    # ---- MatImputer ----
    mat_impute = MatImputer()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
        imputed_array_mat = mat_impute.transform(df_impute_reset[cols_for_impute])
    df_imputed_mat = pd.DataFrame(imputed_array_mat, columns=cols_for_impute)
    restore_merge_save(df_imputed_mat,
                       fr".\dataset\Li-rich_train_mat_imputation_{dataset_suffix}.csv",
                       df_impute_reset, non_impute_cols_all, df_impute)
    
    # ---- SimpleImputer (Mean Strategy) ----
    mean_imputer = SimpleImputer(strategy='mean')
    imputed_array_mean = mean_imputer.fit_transform(df_impute_reset[cols_for_impute])
    df_imputed_mean = pd.DataFrame(imputed_array_mean, columns=cols_for_impute)
    restore_merge_save(df_imputed_mean,
                       fr".\dataset\Li-rich_train_mean_imputation_{dataset_suffix}.csv",
                       df_impute_reset, non_impute_cols_all, df_impute)
    
    # ---- KNN Imputer ----
    knn_imputer = KNNImputer(n_neighbors=10)
    imputed_array_knn = knn_imputer.fit_transform(df_impute_reset[cols_for_impute])
    df_imputed_knn = pd.DataFrame(imputed_array_knn, columns=cols_for_impute)
    restore_merge_save(df_imputed_knn,
                       fr".\dataset\Li-rich_train_knn_imputation_{dataset_suffix}.csv",
                       df_impute_reset, non_impute_cols_all, df_impute)
    
    # ---- Iterative Imputer (MICE) ----
    mice_imputer = IterativeImputer(random_state=seed, max_iter=10000)
    imputed_array_mice = mice_imputer.fit_transform(df_impute_reset[cols_for_impute])
    df_imputed_mice = pd.DataFrame(imputed_array_mice, columns=cols_for_impute)
    restore_merge_save(df_imputed_mice,
                       fr".\dataset\Li-rich_train_mice_imputation_{dataset_suffix}.csv",
                       df_impute_reset, non_impute_cols_all, df_impute)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Impute missing values in the Li-rich dataset using various imputation methods."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["full", "half"],
        help="Specify which dataset to use: 'full' for Li-rich data collection_full.csv or 'half' for Li-rich data collection_half.csv"
    )
    args = parser.parse_args()
    main(args)

    # Example usage:
    # python imputer.py --dataset full ( for imputation including the  missing target values)
    # python imputer.py --dataset half ( for imputation excluding the missing target values)

import os
import io
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    mean_absolute_error
)


from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor


from utils import*

plot_config = PlotConfig(
    title_fontsize=12,
    label_fontsize=12,
    tick_fontsize=12,
    legend_fontsize=12,
    line_width=3,
    marker_size=25,
    dpi_value=500
)

plt.rcParams["font.family"] = "serif"




EWMA_SPAN = 150  
MIN_FOLDS_PER_POINT = 5 

def _ewma(y, span=EWMA_SPAN):
    """Exponential weighted moving average of a 1D list/array."""
    if y is None or len(y) == 0:
        return np.array([])
    return pd.Series(y).ewm(span=span, adjust=False).mean().values


def _plot_fold_loss_with_ewma(fd, suffix, config, filename):
    """One plot per fold: raw + EWMA for train/val."""
    it = fd['iterations']; tr = fd['train_losses']; va = fd['val_losses']
    tr_s = _ewma(tr); va_s = _ewma(va)

    plt.figure(figsize=(6.0, 4.0), dpi=config.dpi_value)
    plt.plot(it, tr, linewidth=1, alpha=0.35, label='Train (raw)')
    plt.plot(it, va, linewidth=1, alpha=0.35, label='Val (raw)')
    plt.plot(it, tr_s, linewidth=2.0, label='Train (EWMA)')
    plt.plot(it, va_s, linewidth=2.0, label='Val (EWMA)')
    plt.xlabel('Boosting iteration', fontsize=config.label_fontsize)
    plt.ylabel('Loss', fontsize=config.label_fontsize)
    plt.title(f'Fold {fd["fold"]}: Loss vs. Iteration ({suffix})', fontsize=config.title_fontsize)
    plt.legend(fontsize=config.legend_fontsize)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def compute_nll(y_true, y_dist):
    """
    Compute the Negative Log Likelihood (NLL) for a set of true values
    and the corresponding predicted probability distribution.
    """
    return -np.mean(y_dist.logpdf(y_true))


def compute_calibration_curve(y_true, y_dist, num_quantiles=101):
    """
    Computes the empirical coverage vs. predicted quantile curve and
    calculates the AUCE (area under the calibration error curve).

    For each quantile in [0,1], it computes the fraction of y_true below the predicted
    q-th percentile. AUCE is computed as the integral of the absolute difference between
    the empirical coverage and the quantile.
    """
    quantiles = np.linspace(0, 1, num_quantiles)
    empirical_coverage = []
    for q in quantiles:
        predicted_q = y_dist.ppf(q)
        empirical_coverage.append(np.mean(y_true <= predicted_q))
    empirical_coverage = np.array(empirical_coverage)
    errors = np.abs(empirical_coverage - quantiles)
    auce = np.trapz(errors, quantiles)
    return quantiles, empirical_coverage, auce


def train_phase(args):
    """
    Train models for each training dataset suffix. For each, cross-validation is run,
    the final model is trained on the entire training set, the best model and its scaler
    are saved, and relevant plots—including parity plots with uncertainty for each fold and the final model—are generated.
    """
    training_suffixes = ["MatImputer", "KNN Imputer", "MICE Imputer", "Mean Imputer"]
    features = [
        'Li_fraction_in_TM_layer', 'Ni_fraction', 'Co_fraction', 'Mn_fraction',
        'first_sintering_temperature', 'first_sintering_time',
        'second_sintering_temperature', 'second_sintering_time'
    ]

    for suffix in training_suffixes:
        print(f"\n=== Processing Training Dataset: {suffix} ===")
        train_file = os.path.join(args.dataset_dir, f"Li-rich_train_{suffix}_{args.dataset_type}.csv")
        df_train = pd.read_csv(train_file)
        X = df_train[features]
        y = df_train['mean_primary_particle_size']

        scaler = PowerTransformer()
        X_scaled = scaler.fit_transform(X)


        mse_scores, mape_scores, r2_scores, nll_scores = [], [], [], []
        fold_data = []
        fold_loss_history = []
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        print("Running cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            ngb = NGBRegressor(
                Dist=Normal,
                Base=DecisionTreeRegressor(random_state=42),
                n_estimators=5000,
                learning_rate=0.00999999999,
                verbose=True,
                verbose_eval=1,
                validation_fraction=0.15,
                random_state=42,
                minibatch_frac=0.5,
                col_sample=0.5,
                early_stopping_rounds=True,
            )

            log_buffer = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = log_buffer
            ngb.fit(X_train, y_train)
            sys.stdout = old_stdout
            logs = log_buffer.getvalue()

            iterations, train_losses, val_losses = [], [], []
            for line in logs.splitlines():
                if line.startswith("[iter"):
                    try:
                        prefix = line.split("]")[0]
                        iter_str = prefix.replace("[iter", "").strip()
                        iteration = int(iter_str)
                        iterations.append(iteration)
                        parts = line.split()
                        t_loss, v_loss = None, None
                        for part in parts:
                            if part.startswith("loss="):
                                t_loss = float(part.replace("loss=", ""))
                            if part.startswith("val_loss="):
                                v_loss = float(part.replace("val_loss=", ""))
                        if t_loss is not None and v_loss is not None:
                            train_losses.append(t_loss)
                            val_losses.append(v_loss)
                    except Exception:
                        continue
            fold_loss_history.append({
                'fold': fold,
                'iterations': iterations,
                'train_losses': train_losses,
                'val_losses': val_losses
            })

            y_dist = ngb.pred_dist(X_val)
            y_pred = y_dist.loc
            y_std = y_dist.scale

            mse = mean_squared_error(y_val, y_pred)
            mape = mean_absolute_percentage_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            nll = compute_nll(y_val, y_dist)

            mse_scores.append(mse)
            mape_scores.append(mape)
            r2_scores.append(r2)
            nll_scores.append(nll)

            fold_data.append({
                'fold': fold,
                'y_true': y_val.tolist(),
                'y_pred': y_pred.tolist(),
                'y_std': y_std.tolist(), 
                'mse': mse,
                'mape': mape,
                'r2': r2,
                'nll': nll
            })
            print(f"Fold {fold}: MSE = {mse:.4f}, MAPE = {mape:.4f}, R² = {r2:.4f}, NLL = {nll:.4f}")

        avg_MSE = np.mean(mse_scores)
        avg_MAPE = np.mean(mape_scores)
        avg_R2 = np.mean(r2_scores)
        avg_NLL = np.mean(nll_scores)


        os.makedirs("plots", exist_ok=True)
        for fd in fold_loss_history:
            _plot_fold_loss_with_ewma(
                fd, suffix, plot_config,
                filename=f"plots/loss_fold{fd['fold']}_ewma_{suffix}_{args.dataset_type}.png"
            )

        end_gaps = []
        for fd in fold_loss_history:
            if len(fd['val_losses']) and len(fd['train_losses']):
                end_gaps.append(fd['val_losses'][-1] - fd['train_losses'][-1])
        if len(end_gaps):
            q1, med, q3 = np.percentile(end_gaps, [25, 50, 75])
            print(f"End-of-training G_f summary across folds: median={med:.4f}, IQR=[{q1:.4f}, {q3:.4f}]")


        print(f"Average CV Metrics for {suffix}:")
        print(f"  MSE  = {avg_MSE:.4f}")
        print(f"  MAPE = {avg_MAPE:.4f}")
        print(f"  R²   = {avg_R2:.4f}")
        print(f"  NLL  = {avg_NLL:.4f}")


        plot_cv_parity(fold_data, y, suffix, avg_R2, plot_config,
                       filename=f"plots/CV_parity_{suffix}_{args.dataset_type}.png")
        folds = np.arange(1, len(mse_scores) + 1)
        plot_cv_metrics_bar(folds, mse_scores, avg_MSE, mape_scores, avg_MAPE,
                            r2_scores, avg_R2, nll_scores, avg_NLL, plot_config,
                            filename=f"plots/CV_metrics_{suffix}_{args.dataset_type}.png")

        for fold_dict in fold_data:
            plot_fold_parity_with_uncertainty(
                fold_dict,
                suffix,
                plot_config,
                filename=f"plots/Parity_with_uncertainty_fold{fold_dict['fold']}_{suffix}_{args.dataset_type}.png"
            )

        final_ngb = NGBRegressor(
            Dist=Normal,
            Base=DecisionTreeRegressor(random_state=42),
            n_estimators=5000,
            learning_rate=0.00999999999,
            verbose=True,
            verbose_eval=1,
            validation_fraction=0.15,
            random_state=42,
            minibatch_frac=0.5,
            col_sample=0.5,
            early_stopping_rounds=True,
        )
        log_buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = log_buffer
        final_ngb.fit(X_scaled, y)
        sys.stdout = old_stdout
        final_logs = log_buffer.getvalue()
        final_iterations, final_train_losses, final_val_losses = [], [], []
        for line in final_logs.splitlines():
            if line.startswith("[iter"):
                try:
                    prefix = line.split("]")[0]
                    iter_str = prefix.replace("[iter", "").strip()
                    iteration = int(iter_str)
                    final_iterations.append(iteration)
                    parts = line.split()
                    t_loss, v_loss = None, None
                    for part in parts:
                        if part.startswith("loss="):
                            t_loss = float(part.replace("loss=", ""))
                        if part.startswith("val_loss="):
                            v_loss = float(part.replace("val_loss=", ""))
                    if t_loss is not None and v_loss is not None:
                        final_train_losses.append(t_loss)
                        final_val_losses.append(v_loss)
                except Exception:
                    continue

        model_folder = os.path.join("models", f"models_{args.dataset_type}")
        os.makedirs(model_folder, exist_ok=True)
        model_filepath = os.path.join(model_folder, f"best_ngb_default_model_{suffix}.pkl")
        joblib.dump(final_ngb, model_filepath)
        scaler_filepath = os.path.join(model_folder, f"scaler_{suffix}.pkl")
        joblib.dump(scaler, scaler_filepath)
        print(f"Final model for {suffix} saved as {model_filepath}.")
        print(f"Scaler for {suffix} saved as {scaler_filepath}.")

        plot_loss_curves(fold_loss_history, final_iterations, final_train_losses, final_val_losses,
                         plot_config, filename=f"plots/loss_curves_{suffix}_{args.dataset_type}.png")

        final_y_dist = final_ngb.pred_dist(X_scaled)
        final_y_pred = final_y_dist.loc
        final_y_std = final_y_dist.scale
        plot_final_parity_with_uncertainty(
            y,
            final_y_pred,
            final_y_std,
            suffix,
            plot_config,
            filename=f"plots/Final_parity_with_uncertainty_{suffix}_{args.dataset_type}.png"
        )


def test_phase(args):
    """
    Load previously saved models and their corresponding scalers for each dataset suffix,
    then evaluate them on the test set. Plots and metrics are saved.
    """
    training_suffixes = ["MatImputer", "KNN Imputer", "MICE Imputer", "Mean Imputer"]
    features = [
        'Li_fraction_in_TM_layer', 'Ni_fraction', 'Co_fraction', 'Mn_fraction',
        'first_sintering_temperature', 'first_sintering_time',
        'second_sintering_temperature', 'second_sintering_time'
    ]
    calibration_results = {}
    model_folder = os.path.join("models", f"models_{args.dataset_type}")

    for suffix in training_suffixes:
        model_filepath = os.path.join(model_folder, f"best_ngb_default_model_{suffix}.pkl")
        scaler_filepath = os.path.join(model_folder, f"scaler_{suffix}.pkl")
        if not os.path.exists(model_filepath) or not os.path.exists(scaler_filepath):
            print(f"Model or scaler for {suffix} not found in {model_folder}. Please run training first.")
            continue

        final_model = joblib.load(model_filepath)
        scaler = joblib.load(scaler_filepath)
        print(f"Loaded final model for {suffix} from {model_filepath}.")

        test_file = os.path.join(args.dataset_dir, f"Li-rich_test_{args.dataset_type}.csv")
        df_test = pd.read_csv(test_file)
        X_test = df_test[features]
        X_test_scaled = scaler.transform(X_test)

        y_test_dist = final_model.pred_dist(X_test_scaled)
        y_test_pred = y_test_dist.loc

        df_test['predicted_mean'] = y_test_pred
        df_test['predicted_std'] = y_test_dist.scale
        df_test['lower_bound'] = y_test_dist.ppf(0.025)
        df_test['upper_bound'] = y_test_dist.ppf(1 - 0.025)

        print(f"Test set predictions for {suffix} (first 5 rows):")
        print(df_test[['predicted_mean', 'predicted_std', 'lower_bound', 'upper_bound']].head())

        if 'mean_primary_particle_size' in df_test.columns:
            y_test_true = df_test['mean_primary_particle_size']
            mse_test  = mean_squared_error(y_test_true, y_test_pred)
            mae_test  = mean_absolute_error(y_test_true, y_test_pred)
            rmse_test = np.sqrt(mse_test)
            mape_test = mean_absolute_percentage_error(y_test_true, y_test_pred)
            r2_test   = r2_score(y_test_true, y_test_pred)
            nll_test  = compute_nll(y_test_true, y_test_dist)

            print(f"\nTest Set Metrics for {suffix}:")
            print(f"  MSE  : {mse_test:.4f}")
            print(f"  MAE  : {mae_test:.4f}")
            print(f"  RMSE : {rmse_test:.4f}")
            print(f"  MAPE : {mape_test:.4f}")
            print(f"  R²   : {r2_test:.4f}")
            print(f"  NLL  : {nll_test:.4f}")

            metrics_annotation = (
                f"MSE: {mse_test:.4f}\n"
                f"MAE: {mae_test:.4f}\n"
                f"RMSE: {rmse_test:.4f}\n"
                f"MAPE: {mape_test:.4f}\n"
                f"R²: {r2_test:.4f}\n"
                f"NLL: {nll_test:.4f}"
            )
            plot_uncertainty_heatmap(
                y_test_true,
                y_test_pred,
                df_test['predicted_std'],
                metrics_annotation,
                plot_config,
                filename=f"plots/uncertainty_heatmap_{suffix}_{args.dataset_type}.png"
            )

            quantiles, empirical_coverage, auce = compute_calibration_curve(y_test_true, y_test_dist, num_quantiles=101)
            calibration_results[suffix] = (quantiles, empirical_coverage, auce)


    plot_combined_calibration_curves(
        calibration_results,
        training_suffixes,
        plot_config,
        filename=f"plots/combined_calibration_curves_{args.dataset_type}.png"
    )


def run_experiment(args):
    if args.mode == "train":
        train_phase(args)
    elif args.mode == "test":
        test_phase(args)
    elif args.mode == "both":
        train_phase(args)
        test_phase(args)

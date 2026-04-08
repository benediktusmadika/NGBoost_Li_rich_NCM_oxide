import os
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor

# --------------------------------------------------
# Command-line argument parser.
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="NGBoost SHAP Interpretation with Extra Plots")
    parser.add_argument('--dataset_dir', type=str, default='./dataset',
                        help='Directory containing the dataset CSV files')
    parser.add_argument('--dataset_type', type=str, choices=['full', 'half'], default='half',
                        help='Dataset type: "full" or "half"')
    parser.add_argument('--suffix', type=str, default='mat_imputation',
                        help='Training dataset suffix (e.g., mat_imputation, knn_imputation, etc.)')
    return parser.parse_args()

# --------------------------------------------------
# Create/get folder for saving SHAP figures.
# --------------------------------------------------
def get_shap_folder():
    shap_folder = os.path.join("plots", "shap")
    os.makedirs(shap_folder, exist_ok=True)
    return shap_folder

# --------------------------------------------------
# Load trained model and scaler.
# --------------------------------------------------
def load_model_and_scaler(suffix, dataset_type):
    model_folder = os.path.join("models", f"models_{dataset_type}")
    model_path = os.path.join(model_folder, f"best_ngb_default_model_{suffix}.pkl")
    scaler_path = os.path.join(model_folder, f"scaler_{suffix}.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model or scaler for {suffix} not found in {model_folder}.")
        return None, None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# --------------------------------------------------
# Load training data.
# --------------------------------------------------
def load_training_data(suffix, dataset_dir, dataset_type, features):
    train_file = os.path.join(dataset_dir, f"Li-rich_train_{suffix}_{dataset_type}.csv")
    df = pd.read_csv(train_file)
    X = df[features]
    y = df['mean_primary_particle_size']
    return X, y

# --------------------------------------------------
# Non-SHAP Plot: Feature Importance.
# --------------------------------------------------
def plot_feature_importance(model, features_short, suffix, dataset_type):
    feature_importance_loc = model.feature_importances_[0]
    feature_importance_scale = model.feature_importances_[1]
    
    df_loc = pd.DataFrame({'feature': features_short, 'importance': feature_importance_loc})
    df_loc.sort_values('importance', ascending=False, inplace=True)
    df_scale = pd.DataFrame({'feature': features_short, 'importance': feature_importance_scale})
    df_scale.sort_values('importance', ascending=False, inplace=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Feature Importance for Distribution Parameters", fontsize=17)
    sns.barplot(x='importance', y='feature', data=df_loc, color="skyblue", ax=ax1)
    ax1.set_title('loc (Mean)')
    sns.barplot(x='importance', y='feature', data=df_scale, color="skyblue", ax=ax2)
    ax2.set_title('scale (Uncertainty)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("plots", exist_ok=True)
    out_file = os.path.join("plots", f"Feature_importance_{suffix}_{dataset_type}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {out_file}")

# --------------------------------------------------
# Basic SHAP Summary Plot.
# --------------------------------------------------
def plot_shap_summary(model, X, features_short, suffix, dataset_type, param='loc'):
    shap_folder = get_shap_folder()
    model_output = 0 if param == 'loc' else 1
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output=model_output)
    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=features_short, show=False)
    out_file = os.path.join(shap_folder, f"SHAP_summary_{param}_{suffix}_{dataset_type}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot for {param} saved to {out_file}")

# --------------------------------------------------
# Advanced SHAP Analysis: Summary + Dependence Plot.
# --------------------------------------------------
def plot_shap_advanced(model, X, features_short, suffix, dataset_type, param='loc'):
    shap_folder = get_shap_folder()
    model_output = 0 if param == 'loc' else 1
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output=model_output)
    shap_values = explainer.shap_values(X)
    
    # Advanced summary plot.
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=features_short, show=False)
    summary_file = os.path.join(shap_folder, f"SHAP_summary_adv_{param}_{suffix}_{dataset_type}.png")
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Advanced SHAP summary plot for {param} saved to {summary_file}")
    
    # Determine the top feature based on average absolute SHAP value.
    mean_abs_shap = np.abs(shap_values).mean(0)
    top_feature_idx = np.argmax(mean_abs_shap)
    top_feature = features_short[top_feature_idx]
    
    # Dependence plot for the top feature.
    plt.figure()
    shap.dependence_plot(top_feature, shap_values, X, feature_names=features_short,
                           interaction_index="auto", show=False)
    depend_file = os.path.join(shap_folder, f"SHAP_dependence_{param}_{top_feature}_{suffix}_{dataset_type}.png")
    plt.savefig(depend_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Advanced SHAP dependence plot for {param} and feature '{top_feature}' saved to {depend_file}")

# --------------------------------------------------
# SHAP Interaction Plot.
# --------------------------------------------------
def plot_shap_interaction(model, X, features_short, suffix, dataset_type, param='loc'):
    shap_folder = get_shap_folder()
    model_output = 0 if param == 'loc' else 1
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output=model_output)
    interaction_values = explainer.shap_interaction_values(X)
    
    plt.figure()
    features_np = np.array(features_short)
    shap.summary_plot(
        interaction_values,
        X,
        feature_names=features_np,
        show=False,
        plot_size=(14, 6),
        max_display=len(features_np)
    )
    plt.tight_layout()
    out_file = os.path.join(shap_folder, f"SHAP_interaction_{param}_{suffix}_{dataset_type}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP interaction plot for {param} saved to {out_file}")

# --------------------------------------------------
# Extra SHAP Plots:
#   - Bar plot
#   - Beeswarm plot
#   - Interaction matrix (with color bar and bottom x-axis labels)
#   - Layered violin and violin summary
#   - Heatmap
# --------------------------------------------------
def plot_extra_shap_plots(model, X, features_short, suffix, dataset_type, param='loc'):
    shap_folder = get_shap_folder()
    model_output = 0 if param == 'loc' else 1
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output=model_output)
    shap_values = explainer.shap_values(X)
    
    # Wrap raw shap values in an Explanation object.
    expl = shap.Explanation(values=shap_values, data=X, feature_names=features_short)
    
    # Extra: SHAP bar plot.
    plt.figure()
    shap.plots.bar(expl, show=False)
    bar_file = os.path.join(shap_folder, f"SHAP_bar_extra_{param}_{suffix}_{dataset_type}.png")
    plt.savefig(bar_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Extra SHAP bar plot saved to {bar_file}")
    
    # Extra: SHAP beeswarm plot.
    plt.figure()
    shap.plots.beeswarm(expl, show=False)
    beeswarm_file = os.path.join(shap_folder, f"SHAP_beeswarm_extra_{param}_{suffix}_{dataset_type}.png")
    plt.savefig(beeswarm_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Extra SHAP beeswarm plot saved to {beeswarm_file}")
    
    # Extra: SHAP interaction matrix.
    shap_interaction_values = explainer.shap_interaction_values(X)
    interaction_matrix = np.abs(shap_interaction_values).sum(0)
    for i in range(interaction_matrix.shape[0]):
        interaction_matrix[i, i] = 0
    inds = np.argsort(-interaction_matrix.sum(0))[:12]
    sorted_ia_matrix = interaction_matrix[inds, :][:, inds]
    
    plt.figure(figsize=(12, 12))
    # Use imshow and add a color bar.
    im = plt.imshow(sorted_ia_matrix, aspect='auto', cmap='viridis')
    cbar = plt.colorbar(im)
    cbar.set_label("Absolute Interaction", rotation=270, labelpad=15)
    # Set y-axis and x-axis tick labels.
    plt.yticks(range(sorted_ia_matrix.shape[0]), [features_short[i] for i in inds], rotation=50.4, horizontalalignment="right")
    plt.xticks(range(sorted_ia_matrix.shape[0]), [features_short[i] for i in inds], rotation=50.4, horizontalalignment="left")
    # Do not set x-axis ticks on top; leave them at the bottom.
    plt.tight_layout()
    interaction_file = os.path.join(shap_folder, f"SHAP_interaction_matrix_{param}_{suffix}_{dataset_type}.png")
    plt.savefig(interaction_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP interaction matrix plot saved to {interaction_file}")
    
    # Extra: SHAP summary plot with layered violin plot type.
    if X.shape[0] > 1000:
        X_subset = X[:1000]
        shap_values_subset = np.array(shap_values)[:1000]
    else:
        X_subset = X
        shap_values_subset = shap_values
    plt.figure()
    shap.summary_plot(shap_values_subset, X_subset, feature_names=features_short,
                      plot_type="layered_violin", color="coolwarm", show=False)
    layered_violin_file = os.path.join(shap_folder, f"SHAP_layered_violin_{param}_{suffix}_{dataset_type}.png")
    plt.savefig(layered_violin_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP layered violin plot saved to {layered_violin_file}")
    
    # Extra: SHAP summary plot with violin plot type.
    plt.figure()
    shap.summary_plot(shap_values_subset, X_subset, feature_names=features_short, plot_type="violin", show=False)
    violin_file = os.path.join(shap_folder, f"SHAP_violin_{param}_{suffix}_{dataset_type}.png")
    plt.savefig(violin_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP violin plot saved to {violin_file}")
    
    # Extra: SHAP heatmap.
    plt.figure()
    shap.plots.heatmap(expl, show=False)
    heatmap_file = os.path.join(shap_folder, f"SHAP_heatmap_{param}_{suffix}_{dataset_type}.png")
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP heatmap plot saved to {heatmap_file}")

# --------------------------------------------------
# Main entry point.
# --------------------------------------------------
def main():
    args = parse_args()
    
    # Define the feature list (as used during training).
    features = [
        'Li_fraction_in_TM_layer', 'Ni_fraction', 'Co_fraction', 'Mn_fraction',
        'first_sintering_temperature', 'first_sintering_time',
        'second_sintering_temperature', 'second_sintering_time'
    ]


    # 2. Define a parallel “short” list for plotting:
    features_short = [
        'Li_frac', 'Ni_frac', 'Co_frac', 'Mn_frac',
        'first_sinter_T', 'first_sinter_t',
        'second_sinter_T', 'second_sinter_t'
    ]
    # Process all four suffixes.
    training_suffixes = ["MatImputer", "KNN Imputer", "MICE Imputer", "Mean Imputer"]
    
    # Initialize SHAP's JavaScript visualization (useful for interactive environments).
    shap.initjs()
    
    for suffix in training_suffixes:
        print(f"\nProcessing model interpretation for dataset '{suffix}' ({args.dataset_type})")
        model, scaler = load_model_and_scaler(suffix, args.dataset_type)
        if model is None or scaler is None:
            print(f"Skipping {suffix} because the model or scaler was not found.")
            continue
        
        # Load training data as a DataFrame and convert to a NumPy array.
        X_df, y = load_training_data(suffix, args.dataset_dir, args.dataset_type, features)
        X = X_df.values  # SHAP prefers NumPy arrays
        # Scale the data using the saved scaler.
        X_scaled = scaler.transform(X)
        
        # Non-SHAP plot: feature importance.
        plot_feature_importance(model, features, suffix, args.dataset_type)
        
        # Generate basic SHAP summary plots for "loc" and "scale".
        plot_shap_summary(model, X_scaled, features, suffix, args.dataset_type, param='loc')
        plot_shap_summary(model, X_scaled, features, suffix, args.dataset_type, param='scale')
        
        # Advanced SHAP analysis: summary and dependence plots.
        plot_shap_advanced(model, X_scaled, features, suffix, args.dataset_type, param='loc')
        plot_shap_advanced(model, X_scaled, features, suffix, args.dataset_type, param='scale')
        
        # SHAP interaction plots.
        plot_shap_interaction(model, X_scaled, features, suffix, args.dataset_type, param='loc')
        plot_shap_interaction(model, X_scaled, features, suffix, args.dataset_type, param='scale')
        
        # Generate extra SHAP plots.
        plot_extra_shap_plots(model, X_scaled, features, suffix, args.dataset_type, param='loc')
        plot_extra_shap_plots(model, X_scaled, features, suffix, args.dataset_type, param='scale')

if __name__ == "__main__":
    main()

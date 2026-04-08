import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable



@dataclass
class PlotConfig:
   
    title_fontsize: int = 15
    label_fontsize: int = 15
    tick_fontsize: int = 20
    legend_fontsize: int = 15
    annotation_fontsize: int = 15
    colorbar_label_fontsize: int = 15
    colorbar_tick_fontsize: int = 15


    line_width: float = 3.0
    marker_size: int = 20
    dpi_value: int = 400

    axis_margin: float = 0.2         
    axis_margin_mode: str = "absolute" 


@dataclass
class ScaleRanges:
    """Shared scales for parity/uncertainty plots."""
    xy_min: float
    xy_max: float
    std_min: Optional[float] = None
    std_max: Optional[float] = None



_GLOBAL_PARITY_CACHE: Dict[str, ScaleRanges] = {} 


def _apply_font_rc_params(cfg: PlotConfig) -> None:
    rcParams.update({
        "font.size": cfg.tick_fontsize,  
        "axes.titlesize": cfg.title_fontsize,
        "axes.labelsize": cfg.label_fontsize,
        "xtick.labelsize": cfg.tick_fontsize,
        "ytick.labelsize": cfg.tick_fontsize,
        "legend.fontsize": cfg.legend_fontsize,
    })



def _enforce_tick_label_sizes(ax: plt.Axes, size: int) -> None:
    ax.tick_params(axis='both', which='both', labelsize=size)
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lab.set_fontsize(size)


def _with_margin(lo: float, hi: float, margin: float = 0.02, mode: str = "relative") -> Tuple[float, float]:
    """
    Expand [lo, hi] by a relative fraction or an absolute value.
    """
    if mode == "absolute":
        return lo - margin, hi + margin
    rng = hi - lo
    if rng <= 0:
        return lo - margin, hi + margin
    return lo - rng * margin, hi + rng * margin


def _set_cached_parity_scales(suffix: str, scales: ScaleRanges) -> None:
    _GLOBAL_PARITY_CACHE[suffix] = scales


def _get_cached_parity_scales(suffix: str) -> Optional[ScaleRanges]:
    return _GLOBAL_PARITY_CACHE.get(suffix)


def compute_global_scales(
    folds: List[dict],
    extra_xy: Optional[List[np.ndarray]] = None,
    extra_std: Optional[List[np.ndarray]] = None,
    margin: float = 0.02,
    mode: str = "relative",
) -> ScaleRanges:
    """
    Collects y_true/y_pred (and y_std if present) across all folds (and any extras)
    to build one shared axis range and one shared colorbar range.
    """
    xy_vals: List[np.ndarray] = []
    std_vals: List[np.ndarray] = []

    for d in folds:
        if 'y_true' in d and d['y_true'] is not None:
            xy_vals.append(np.asarray(d['y_true']).ravel())
        if 'y_pred' in d and d['y_pred'] is not None:
            xy_vals.append(np.asarray(d['y_pred']).ravel())
        if 'y_std' in d and d['y_std'] is not None:
            std_vals.append(np.asarray(d['y_std']).ravel())

    if extra_xy:
        for arr in extra_xy:
            xy_vals.append(np.asarray(arr).ravel())

    if extra_std:
        for arr in extra_std:
            std_vals.append(np.asarray(arr).ravel())

    if not xy_vals:
        raise ValueError("No data found to compute global scales.")

    xy_min = min(v.min() for v in xy_vals)
    xy_max = max(v.max() for v in xy_vals)
    xy_min, xy_max = _with_margin(xy_min, xy_max, margin=margin, mode=mode)

    std_min = std_max = None
    if std_vals:
        std_min = min(v.min() for v in std_vals)
        std_max = max(v.max() for v in std_vals)
        std_min, std_max = _with_margin(std_min, std_max, margin=0.02, mode="relative")

    return ScaleRanges(xy_min=xy_min, xy_max=xy_max, std_min=std_min, std_max=std_max)


def plot_cv_parity(
    fold_data,
    y,
    suffix,
    avg_R2,
    config: PlotConfig,
    filename: str = None,
    scales: Optional[ScaleRanges] = None
):
    """
    CV parity plot (all folds). Global x/y limits come strictly from FULL training 'y'
    with an ABSOLUTE ±config.axis_margin. Limits are cached per suffix for reuse.
    """
    _apply_font_rc_params(config)

    y = np.asarray(y).ravel()
    y_lo, y_hi = y.min(), y.max()
    xy_min, xy_max = _with_margin(y_lo, y_hi, margin=config.axis_margin, mode="absolute")

    std_vals = [np.asarray(d['y_std']).ravel() for d in fold_data if d.get('y_std') is not None]
    std_min = std_max = None
    if len(std_vals):
        s_min = min(v.min() for v in std_vals)
        s_max = max(v.max() for v in std_vals)
        std_min, std_max = _with_margin(s_min, s_max, margin=0.02, mode="relative")

    scales = ScaleRanges(xy_min=xy_min, xy_max=xy_max, std_min=std_min, std_max=std_max)
    _set_cached_parity_scales(suffix, scales)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=config.dpi_value)
    colors = sns.color_palette("Dark2", len(fold_data))

    for i, d in enumerate(fold_data):
        ax.scatter(
            d['y_true'], d['y_pred'],
            s=config.marker_size, alpha=0.7,
            label=f'Fold {d["fold"]}', color=colors[i],
            edgecolor='k', linewidth=0.5
        )

    ax.plot([scales.xy_min, scales.xy_max], [scales.xy_min, scales.xy_max],
            color='red', linewidth=config.line_width, label='Ideal')

    ax.set_xlim(scales.xy_min, scales.xy_max)
    ax.set_ylim(scales.xy_min, scales.xy_max)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('Actual Value (μm)', fontsize=config.label_fontsize)
    ax.set_ylabel('Predicted Values (μm)', fontsize=config.label_fontsize)
    ax.set_title(f'CV Parity Plot - {suffix}', fontsize=config.title_fontsize)

    leg = ax.legend(loc='upper left', prop={'size': config.legend_fontsize})
    _enforce_tick_label_sizes(ax, config.tick_fontsize)
    ax.grid(alpha=0.25, linewidth=0.8)

    ax.annotate(
        f"Avg. R² = {avg_R2:.4f}",
        xy=(0.98, 0.02), xycoords='axes fraction',
        fontsize=config.annotation_fontsize, ha="right", va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8)
    )

    if not filename:
        filename = f"plots/CV_parity_{suffix}.png"
    os.makedirs("plots", exist_ok=True)
    fig.tight_layout()
    fig.savefig(filename, dpi=config.dpi_value)
    plt.close(fig)


def plot_cv_metrics_bar(
    folds, mse_scores, avg_MSE, mape_scores, avg_MAPE,
    r2_scores, avg_R2, nll_scores, avg_NLL,
    config: PlotConfig, filename: str = None
):
    _apply_font_rc_params(config)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=config.dpi_value)
    axes = axes.ravel()

    panels = [
        ("MSE per Fold", "MSE", folds, mse_scores, f"Avg. MSE = {avg_MSE:.4f}"),
        ("MAPE per Fold", "MAPE", folds, mape_scores, f"Avg. MAPE = {avg_MAPE:.4f}"),
        ("R² per Fold", "R² Score", folds, r2_scores, f"Avg. R² = {avg_R2:.4f}"),
        ("NLL per Fold", "NLL", folds, nll_scores, f"Avg. NLL = {avg_NLL:.4f}"),
    ]

    for ax, (title, ylabel, x, y, text) in zip(axes, panels):
        ax.bar(x, y)
        ax.set_xlabel("Fold", fontsize=config.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=config.label_fontsize)
        ax.set_title(title, fontsize=config.title_fontsize)
        _enforce_tick_label_sizes(ax, config.tick_fontsize)
        ax.text(
            0.95, 0.90, text, transform=ax.transAxes,
            fontsize=config.annotation_fontsize, ha="right",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8)
        )

    fig.tight_layout()
    if not filename:
        filename = "plots/CV_metrics_bar.png"
    os.makedirs("plots", exist_ok=True)
    fig.savefig(filename, dpi=config.dpi_value)
    plt.close(fig)


def plot_loss_curves(
    fold_loss_history,
    final_iterations,
    final_train_losses,
    final_val_losses,
    config: PlotConfig,
    filename: str = None
):
    _apply_font_rc_params(config)

    MS = 3.5

    fig, ax = plt.subplots(figsize=(10, 6), dpi=config.dpi_value)
    palette = sns.color_palette("Dark2", len(fold_loss_history))

    for idx, fd in enumerate(fold_loss_history):
        color = palette[idx % len(palette)]
        iters = fd['iterations']
        t_losses = fd['train_losses']
        v_losses = fd['val_losses']
        if iters and t_losses and v_losses:
            ax.plot(
                iters, t_losses,
                marker='o', markersize=MS, markeredgewidth=0.8,
                linestyle='-', color=color, alpha=0.9,
                label=f'Fold {fd["fold"]} Train Loss'
            )
            ax.plot(
                iters, v_losses,
                marker='x', markersize=MS, markeredgewidth=0.8,
                linestyle='--', color=color, alpha=0.9,
                label=f'Fold {fd["fold"]} Val Loss'
            )

    if final_iterations and final_train_losses and final_val_losses:
        ax.plot(
            final_iterations, final_train_losses,
            marker='o', markersize=MS, markeredgewidth=0.8,
            linestyle='-', color='black', linewidth=config.line_width,
            label='Final Model Train Loss'
        )
        ax.plot(
            final_iterations, final_val_losses,
            marker='x', markersize=MS, markeredgewidth=0.8,
            linestyle='--', color='black', linewidth=config.line_width,
            label='Final Model Val Loss'
        )

    ax.set_xlabel('Iteration', fontsize=config.label_fontsize)
    ax.set_ylabel('Loss', fontsize=config.label_fontsize)
    ax.set_title(
        'Iteration vs. Loss (Training & Validation) for Each Fold and Final Model',
        fontsize=config.title_fontsize
    )

    try:
        leg = ax.legend(
            loc='upper right', ncol=2, fancybox=True, framealpha=0.8,
            fontsize=11, title_fontsize=11
        )
    except TypeError:
        leg = ax.legend(
            loc='upper right', ncol=2, fancybox=True, framealpha=0.8,
            prop={'size': 11}
        )
        if leg.get_title():
            leg.get_title().set_fontsize(14)

    _enforce_tick_label_sizes(ax, config.tick_fontsize)
    ax.xaxis.get_offset_text().set_fontsize(config.tick_fontsize)
    ax.yaxis.get_offset_text().set_fontsize(config.tick_fontsize)

    fig.tight_layout()
    if not filename:
        filename = "plots/loss_curves.png"
    os.makedirs("plots", exist_ok=True)
    fig.savefig(filename, dpi=config.dpi_value)
    plt.close(fig)



def plot_uncertainty_heatmap(
    y_test_true, y_test_pred, predicted_std, metrics_annotation,
    config: PlotConfig, filename: str = None
):
    _apply_font_rc_params(config)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=config.dpi_value)
    sc = ax.scatter(
        y_test_true, y_test_pred,
        c=predicted_std, cmap='viridis',
        alpha=0.7, edgecolor='k', linewidth=0.5,
        s=config.marker_size
    )

    y_true_min, y_true_max = np.min(y_test_true), np.max(y_test_true)
    ax.plot([y_true_min, y_true_max], [y_true_min, y_true_max],
            'r--', linewidth=config.line_width, label='Ideal Fit')

    ax.set_xlabel('Actual Values (μm)', fontsize=config.label_fontsize)
    ax.set_ylabel('Predicted Values (μm)', fontsize=config.label_fontsize)
    ax.set_title('Heatmap (Test Set)', fontsize=config.title_fontsize)
    _enforce_tick_label_sizes(ax, config.tick_fontsize)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label('Predicted Std/Uncertainty (μm)', fontsize=config.colorbar_label_fontsize)
    cb.ax.tick_params(labelsize=config.colorbar_tick_fontsize)
    cb.ax.yaxis.get_offset_text().set_fontsize(config.colorbar_tick_fontsize)

    ax.text(
        0.02, 0.98, metrics_annotation,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=config.annotation_fontsize,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8)
    )

    if not filename:
        filename = "plots/uncertainty_heatmap.png"
    os.makedirs("plots", exist_ok=True)
    fig.tight_layout()
    fig.savefig(filename, dpi=config.dpi_value)
    plt.close(fig)


def plot_combined_calibration_curves(
    calibration_results, training_suffixes, config: PlotConfig, filename: str = None
):
    _apply_font_rc_params(config)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=config.dpi_value)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=config.line_width, label='Perfect Calibration')
    colors = sns.color_palette("Dark2", len(training_suffixes))

    for idx, suffix in enumerate(training_suffixes):
        if suffix not in calibration_results:
            continue
        quantiles, empirical_coverage, auce = calibration_results[suffix]
        ax.plot(quantiles, empirical_coverage, color=colors[idx],
                linewidth=config.line_width, label=f'{suffix} (AUCE={auce:.4f})')

    ax.set_xlabel('Predicted Quantile', fontsize=config.label_fontsize)
    ax.set_ylabel('Empirical Coverage', fontsize=config.label_fontsize)
    ax.set_title('Combined Calibration Curves Comparison (Test Set)',
                 fontsize=config.title_fontsize)
    leg = ax.legend(loc='best', prop={'size': config.legend_fontsize})
    _enforce_tick_label_sizes(ax, config.tick_fontsize)

    if not filename:
        filename = "plots/combined_calibration_curves.png"
    os.makedirs("plots", exist_ok=True)
    fig.tight_layout()
    fig.savefig(filename, dpi=config.dpi_value)
    plt.close(fig)


def plot_fold_parity_with_uncertainty(
    fold_dict,
    suffix,
    config: PlotConfig,
    filename: str = None,
    scales: Optional[ScaleRanges] = None
):
    """
    Parity scatter for one fold; points colored by predicted uncertainty.
    Uses shared axis limits (from cache) and shared colorbar range if available.
    If no cached scales are present, falls back to local limits with ±config.axis_margin (absolute).
    """
    _apply_font_rc_params(config)

    y_true = np.asarray(fold_dict['y_true'])
    y_pred = np.asarray(fold_dict['y_pred'])
    y_std  = fold_dict.get('y_std', None)
    y_std  = None if y_std is None else np.asarray(y_std)
    fold   = fold_dict['fold']

    eff_scales = scales or _get_cached_parity_scales(suffix)
    if eff_scales is None:
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        xy_min, xy_max = _with_margin(lo, hi, margin=config.axis_margin, mode="absolute")
        std_min = std_max = None
        if y_std is not None and y_std.size:
            s_min, s_max = float(y_std.min()), float(y_std.max())
            std_min, std_max = _with_margin(s_min, s_max, margin=0.02, mode="relative")
        eff_scales = ScaleRanges(xy_min=xy_min, xy_max=xy_max, std_min=std_min, std_max=std_max)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=config.dpi_value)
    sc = ax.scatter(
        y_true, y_pred,
        c=(y_std if y_std is not None else None),
        cmap='viridis',
        s=config.marker_size, alpha=0.7,
        edgecolor='k', linewidth=0.5,
        vmin=(eff_scales.std_min if eff_scales.std_min is not None else None),
        vmax=(eff_scales.std_max if eff_scales.std_max is not None else None),
    )

    ax.plot([eff_scales.xy_min, eff_scales.xy_max],
            [eff_scales.xy_min, eff_scales.xy_max],
            'r--', linewidth=config.line_width, label='Ideal')

    ax.set_xlim(eff_scales.xy_min, eff_scales.xy_max)
    ax.set_ylim(eff_scales.xy_min, eff_scales.xy_max)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('Actual Values (μm)', fontsize=config.label_fontsize)
    ax.set_ylabel('Predicted Values (μm)', fontsize=config.label_fontsize)
    ax.set_title(f'Parity Plot with Uncertainty (Fold {fold} - {suffix})',
                 fontsize=config.title_fontsize)

    _enforce_tick_label_sizes(ax, config.tick_fontsize)
    ax.grid(alpha=0.25, linewidth=0.8)

    if y_std is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label('Predicted Std/Uncertainty (μm)', fontsize=config.colorbar_label_fontsize)
        cbar.ax.tick_params(labelsize=config.colorbar_tick_fontsize)
        cbar.ax.yaxis.get_offset_text().set_fontsize(config.colorbar_tick_fontsize)

    leg = ax.legend(prop={'size': config.legend_fontsize})
    if not filename:
        filename = f"plots/Parity_with_uncertainty_fold{fold}_{suffix}.png"
    os.makedirs("plots", exist_ok=True)
    fig.tight_layout()
    fig.savefig(filename, dpi=config.dpi_value)
    plt.close(fig)


def plot_final_parity_with_uncertainty(
    y_true, y_pred, y_std, suffix,
    config: PlotConfig,
    filename: str = None,
    scales: Optional[ScaleRanges] = None
):
    """
    Final model parity plot with uncertainty. Reuses cached scales from CV phase
    (computed off FULL training y) if available; otherwise falls back to local
    limits expanded by absolute margin.
    """
    _apply_font_rc_params(config)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_std  = np.asarray(y_std)

    eff_scales = scales or _get_cached_parity_scales(suffix)
    if eff_scales is None:
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        xy_min, xy_max = _with_margin(lo, hi, margin=config.axis_margin, mode="absolute")
        s_min, s_max = float(y_std.min()), float(y_std.max())
        std_min, std_max = _with_margin(s_min, s_max, margin=0.02, mode="relative")
        eff_scales = ScaleRanges(xy_min=xy_min, xy_max=xy_max, std_min=std_min, std_max=std_max)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=config.dpi_value)
    sc = ax.scatter(
        y_true, y_pred,
        c=y_std, cmap='viridis',
        s=config.marker_size, alpha=0.7,
        edgecolor='k', linewidth=0.5,
        vmin=(eff_scales.std_min if eff_scales.std_min is not None else None),
        vmax=(eff_scales.std_max if eff_scales.std_max is not None else None),
    )

    ax.plot([eff_scales.xy_min, eff_scales.xy_max],
            [eff_scales.xy_min, eff_scales.xy_max],
            'r--', linewidth=config.line_width, label='Ideal')

    ax.set_xlim(eff_scales.xy_min, eff_scales.xy_max)
    ax.set_ylim(eff_scales.xy_min, eff_scales.xy_max)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('Actual Values (μm)', fontsize=config.label_fontsize)
    ax.set_ylabel('Predicted Values (μm)', fontsize=config.label_fontsize)
    ax.set_title(f'Final Model Parity Plot with Uncertainty ({suffix})',
                 fontsize=config.title_fontsize)

    _enforce_tick_label_sizes(ax, config.tick_fontsize)
    ax.grid(alpha=0.25, linewidth=0.8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label('Predicted Std/Uncertainty (μm)', fontsize=config.colorbar_label_fontsize)
    cbar.ax.tick_params(labelsize=config.colorbar_tick_fontsize)
    cbar.ax.yaxis.get_offset_text().set_fontsize(config.colorbar_tick_fontsize)

    leg = ax.legend(prop={'size': config.legend_fontsize})

    if not filename:
        filename = f"plots/Final_parity_with_uncertainty_{suffix}.png"
    os.makedirs("plots", exist_ok=True)
    fig.tight_layout()
    fig.savefig(filename, dpi=config.dpi_value)
    plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde


def plot_comparison(pred, true, field_name, unit, mask=None,
                    field_range=None, nmae_case=None, nearwall_nmae=None, err_vmax=None):
    """
    Three-panel figure: prediction | reference | NMAE error map (%).

    Error map = |pred - true| / (y_max - y_min) × 100  (NMAE per cell).
    Per-case NMAE metrics are shown as a text annotation on the error panel.

    Args:
        pred          : 2D predicted array
        true          : 2D reference array (OpenFOAM)
        field_name    : field label (e.g. 'X-Velocity')
        unit          : physical unit (e.g. 'm/s', 'Pa')
        mask          : 2D bool array (True = airfoil interior, shown white)
        field_range   : y_max - y_min [physical unit]. If None, uses max(|true|).
        nmae_case     : per-case global NMAE (%) — shown as annotation
        nearwall_nmae : per-case near-wall NMAE (%) — shown as annotation
    """
    d = field_range if (field_range is not None and field_range != 0) else (np.max(np.abs(true)) or 1.0)
    error_nmae_map = np.abs(true - pred) / d * 100  # NMAE per cell (%)

    vmin = min(pred.min(), true.min())
    vmax = max(pred.max(), true.max())

    # Colormaps — airfoil interior shown in white
    cmap_field = plt.cm.viridis.copy()
    cmap_field.set_bad('white')
    cmap_err = plt.cm.Reds.copy()
    cmap_err.set_bad('white')

    if mask is not None:
        pred_plot = np.ma.masked_where(mask, pred)
        true_plot = np.ma.masked_where(mask, true)
        err_plot  = np.ma.masked_where(mask, error_nmae_map)
    else:
        pred_plot, true_plot, err_plot = pred, true, error_nmae_map

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    im0 = axes[0].imshow(pred_plot, cmap=cmap_field, origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Hybrid U-Net — {field_name}')
    plt.colorbar(im0, ax=axes[0], label=unit)

    im1 = axes[1].imshow(true_plot, cmap=cmap_field, origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'OpenFOAM — {field_name}')
    plt.colorbar(im1, ax=axes[1], label=unit)

    im2 = axes[2].imshow(err_plot, cmap=cmap_err, origin='lower', vmin=0, vmax=err_vmax)
    axes[2].set_title(f'NMAE map (%) — {field_name}', fontsize=10)
    plt.colorbar(im2, ax=axes[2], label='NMAE (%)')

    # Per-case NMAE annotations (bottom-left of error panel)
    lines = []
    if nmae_case is not None:
        lines.append(f"NMAE: {nmae_case:.2f}%")
    if nearwall_nmae is not None:
        lines.append(f"NW NMAE: {nearwall_nmae:.2f}%")
    if lines:
        axes[2].text(
            0.02, 0.03, "\n".join(lines),
            transform=axes[2].transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
        )

    plt.tight_layout()
    return fig


def plot_sdf(sdf, case_label=""):
    """
    Visualiza el campo SDF de un caso.

    Args:
        sdf        : array 2D del SDF normalizado (0 = interior airfoil)
        case_label : texto para el título (e.g. 'case 007')
    """
    cmap = plt.cm.plasma.copy()
    cmap.set_bad('white')

    airfoil_mask = (sdf == 0)
    sdf_plot = np.ma.masked_where(airfoil_mask, sdf)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(sdf_plot, cmap=cmap, origin='lower', vmin=0)
    ax.set_title(f'SDF (normalized) — {case_label}')
    plt.colorbar(im, ax=ax, label='SDF norm.')
    plt.tight_layout()
    return fig


def plot_predictions(fields, titles=None):
    """
    Muestra los tres campos predichos (Ux, Uy, P) en una sola figura.

    Args:
        fields : dict {"Ux": array, "Uy": array, "P": array}
        titles : dict opcional con títulos personalizados
    """
    default_titles = {
        "Ux": ("Hybrid U-Net X-Velocity", "m/s"),
        "Uy": ("Hybrid U-Net Y-Velocity", "m/s"),
        "P":  ("Hybrid U-Net Pressure",   "Pa"),
    }
    titles = titles or default_titles

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    for ax, (key, arr) in zip(axes, fields.items()):
        title, unit = titles[key]
        im = ax.imshow(arr, cmap='viridis', origin='lower')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label=unit)

    plt.tight_layout()
    return fig


def plot_training_history(history):
    """Grafica la curva de pérdida de entrenamiento y validación."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["loss"],     label="Train loss")
    ax.plot(history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (weighted MSE)")
    ax.legend()
    ax.set_title("Training curve")
    plt.tight_layout()
    return fig


def plot_nearwall_distribution(nmae_ux, nmae_uy, nmae_p):
    """
    Near-wall NMAE distribution across the test set (one value per case).

    NMAE = MAE / (y_max - y_min) × 100. Shows normalized histogram + KDE
    for Ux, Uy and P. The Mean line matches the value reported in evaluate.py.

    Args:
        nmae_ux : 1D array — near-wall NMAE (%) per test case for Ux
        nmae_uy : 1D array — near-wall NMAE (%) per test case for Uy
        nmae_p  : 1D array — near-wall NMAE (%) per test case for P
    """
    fields_data = [
        (nmae_ux, "Ux",       "steelblue"),
        (nmae_uy, "Uy",       "darkorange"),
        (nmae_p,  "Pressure", "seagreen"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    for ax, (errors, name, color) in zip(axes, fields_data):
        errors = errors[np.isfinite(errors)]
        if errors.size == 0:
            ax.set_title(f"{name} — no data")
            continue

        ax.hist(errors, bins=50, density=True, color=color, alpha=0.4, label="Histogram")

        if errors.size > 1:
            kde = gaussian_kde(errors)
            x = np.linspace(0, np.percentile(errors, 99), 300)
            ax.plot(x, kde(x), color=color, linewidth=2, label="KDE")

        mean_val = errors.mean()
        ax.axvline(mean_val, color='black', linestyle='--', linewidth=1.2,
                   label=f"Mean = {mean_val:.2f}%")

        ax.set_xlabel("Near-wall NMAE (%)")
        ax.set_ylabel("Density")
        ax.set_title(f"Near-wall NMAE — {name}")
        ax.legend(fontsize=8)

    plt.suptitle("Near-wall NMAE distribution (test set)", fontsize=12)
    plt.tight_layout()
    return fig

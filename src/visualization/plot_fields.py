import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(pred, true, field_name, unit, denom=None, nearwall_mae=None, mask=None):
    """
    Muestra en tres subplots: predicción, referencia y error porcentual.

    El error % se calcula según la fórmula del paper:
        error%[i,j] = |true[i,j] - pred[i,j]| / mean(|true|) × 100

    Args:
        pred         : array 2D predicho
        true         : array 2D de referencia (OpenFOAM)
        field_name   : nombre del campo (e.g. 'X-Velocity')
        unit         : unidad (e.g. 'm/s', 'Pa')
        denom        : denominador = mean(|true|). Si es None se calcula del propio caso.
        nearwall_mae : MAE% en la zona near-wall (float). Si se pasa, se muestra en el título.
        mask         : array 2D booleano (True = airfoil). Se muestra en blanco.
    """
    d = denom if denom is not None else np.mean(np.abs(true))
    d = d if d != 0 else 1.0
    error_pct = np.abs(true - pred) / d * 100

    vmin = min(pred.min(), true.min())
    vmax = max(pred.max(), true.max())

    # Colormaps con color blanco para la zona enmascarada (airfoil)
    cmap_field = plt.cm.viridis.copy()
    cmap_field.set_bad('white')
    cmap_err = plt.cm.Reds.copy()
    cmap_err.set_bad('white')

    if mask is not None:
        pred_plot = np.ma.masked_where(mask, pred)
        true_plot = np.ma.masked_where(mask, true)
        err_plot  = np.ma.masked_where(mask, error_pct)
    else:
        pred_plot, true_plot, err_plot = pred, true, error_pct

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    im0 = axes[0].imshow(pred_plot, cmap=cmap_field, origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Hybrid U-Net — {field_name}')
    plt.colorbar(im0, ax=axes[0], label=unit)

    im1 = axes[1].imshow(true_plot, cmap=cmap_field, origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'OpenFOAM — {field_name}')
    plt.colorbar(im1, ax=axes[1], label=unit)

    # MAE global calculado solo sobre el dominio fluido (excluye airfoil)
    flow_mask = ~mask if mask is not None else np.ones(error_pct.shape, dtype=bool)
    global_mae = error_pct[flow_mask].mean()

    nw_str = f"  |  Near-wall MAE = {nearwall_mae:.2f}%" if nearwall_mae is not None else ""
    im2 = axes[2].imshow(err_plot, cmap=cmap_err, origin='lower', vmin=0)
    axes[2].set_title(f'Error(%) — {field_name}  (Global MAE = {global_mae:.2f}%{nw_str})')
    plt.colorbar(im2, ax=axes[2], label='%')

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
    ax.set_title(f'SDF (normalizado) — {case_label}')
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
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida (weighted MSE)")
    ax.legend()
    ax.set_title("Curva de entrenamiento")
    plt.tight_layout()
    return fig

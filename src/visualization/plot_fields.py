import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(pred, true, field_name, unit):
    """
    Muestra en tres subplots: predicción, referencia y error.

    Args:
        pred       : array 2D predicho
        true       : array 2D de referencia (OpenFOAM)
        field_name : nombre del campo (e.g. 'X-Velocity')
        unit       : unidad (e.g. 'm/s', 'Pa')
    """
    diff = true - pred
    vmin = min(pred.min(), true.min())
    vmax = max(pred.max(), true.max())

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    im0 = axes[0].imshow(pred, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Hybrid U-Net — {field_name}')
    plt.colorbar(im0, ax=axes[0], label=unit)

    im1 = axes[1].imshow(true, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'OpenFOAM — {field_name}')
    plt.colorbar(im1, ax=axes[1], label=unit)

    im2 = axes[2].imshow(diff, cmap='coolwarm', origin='lower')
    axes[2].set_title(f'Error — {field_name}')
    plt.colorbar(im2, ax=axes[2], label=unit)

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

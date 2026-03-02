import numpy as np
import matplotlib.pyplot as plt


def _error_pct(pred, true):
    """Error porcentual: |true - pred| / mean(|true|) × 100"""
    d = np.mean(np.abs(true))
    d = d if d != 0 else 1.0
    return np.abs(true - pred) / d * 100


def plot_velocity_x(U_x_pred, u_x_test):
    """Gráfica de comparación para el campo de velocidad X."""
    error_pct = _error_pct(U_x_pred, u_x_test)
    vmin = min(U_x_pred.min(), u_x_test.min())
    vmax = max(U_x_pred.max(), u_x_test.max())

    fig = plt.figure(figsize=(8, 10))

    plt.subplot(3, 1, 1)
    plt.title('Hybrid U-Net X-Velocity field')
    plt.imshow(U_x_pred, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Velocity (m/s)')

    plt.subplot(3, 1, 2)
    plt.title('Original X-Velocity field')
    plt.imshow(u_x_test, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Velocity (m/s)')

    plt.subplot(3, 1, 3)
    plt.title(f'Hybrid U-Net X-Velocity Error  (Global MAE = {error_pct.mean():.2f}%)')
    plt.imshow(error_pct, cmap='Reds', origin='lower', vmin=0)
    plt.colorbar(label='%')

    plt.tight_layout()
    return fig


def plot_velocity_y(U_y_pred, u_y_test):
    """Gráfica de comparación para el campo de velocidad Y."""
    error_pct = _error_pct(U_y_pred, u_y_test)
    vmin = min(U_y_pred.min(), u_y_test.min())
    vmax = max(U_y_pred.max(), u_y_test.max())

    fig = plt.figure(figsize=(8, 10))

    plt.subplot(3, 1, 1)
    plt.title('Hybrid U-Net Y-Velocity field')
    plt.imshow(U_y_pred, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Velocity (m/s)')

    plt.subplot(3, 1, 2)
    plt.title('Original Y-Velocity field')
    plt.imshow(u_y_test, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Velocity (m/s)')

    plt.subplot(3, 1, 3)
    plt.title(f'Hybrid U-Net Y-Velocity Error  (Global MAE = {error_pct.mean():.2f}%)')
    plt.imshow(error_pct, cmap='Reds', origin='lower', vmin=0)
    plt.colorbar(label='%')

    plt.tight_layout()
    return fig


def plot_pressure(P_pred, p_test):
    """Gráfica de comparación para el campo de presión."""
    error_pct = _error_pct(P_pred, p_test)
    vmin = min(P_pred.min(), p_test.min())
    vmax = max(P_pred.max(), p_test.max())

    fig = plt.figure(figsize=(8, 10))

    plt.subplot(3, 1, 1)
    plt.title('Hybrid U-Net Pressure field')
    plt.imshow(P_pred, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Pressure (Pa)')

    plt.subplot(3, 1, 2)
    plt.title('Original Pressure field')
    plt.imshow(p_test, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Pressure (Pa)')

    plt.subplot(3, 1, 3)
    plt.title(f'Hybrid U-Net Pressure Error  (Global MAE = {error_pct.mean():.2f}%)')
    plt.imshow(error_pct, cmap='Reds', origin='lower', vmin=0)
    plt.colorbar(label='%')

    plt.tight_layout()
    return fig


def plot_predictions(U_x_pred, U_y_pred, P_pred, vmin_p=None, vmax_p=None):
    """Gráfica de los tres campos predichos."""
    fig = plt.figure(figsize=(8, 10))

    plt.subplot(3, 1, 1)
    plt.title('Hybrid U-Net X-Velocity field')
    plt.imshow(U_x_pred, cmap='viridis', origin='lower')
    plt.colorbar(label='Velocity (m/s)')

    plt.subplot(3, 1, 2)
    plt.title('Hybrid U-Net Y-Velocity field')
    plt.imshow(U_y_pred, cmap='viridis', origin='lower')
    plt.colorbar(label='Velocity (m/s)')

    plt.subplot(3, 1, 3)
    plt.title('Hybrid U-Net Pressure field')
    plt.imshow(P_pred, cmap='viridis', origin='lower', vmin=vmin_p, vmax=vmax_p)
    plt.colorbar(label='Pressure (Pa)')

    plt.tight_layout()
    return fig

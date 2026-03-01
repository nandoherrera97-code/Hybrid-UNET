import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- RUTAS -------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
DATA_RAW      = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR   = DATA_RAW / "results"
SDF_PKL       = DATA_RAW / "sdf_df.pkl"
OUTPUT_PKL    = DATA_RAW / "CNN_df.pkl"
PLOTS_SDF_DIR = DATA_RAW / "sdf"
PLOTS_CFD_DIR = DATA_RAW / "CFD_graphs"

# Limpiar y recrear carpetas de gráficas
for _d in (PLOTS_SDF_DIR, PLOTS_CFD_DIR):
    shutil.rmtree(_d, ignore_errors=True)
    _d.mkdir()

# ---------- LEER RESULTADOS CSV -----------------------------------------
rows = []
for carpeta in sorted(RESULTS_DIR.iterdir()):
    if not carpeta.is_dir():
        continue
    csv_path = carpeta / "postProcessing" / "resultados.csv"
    if not csv_path.exists():
        print(f"Archivo no encontrado: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    points = np.column_stack((df["Points:0"].values, df["Points:1"].values))
    u      = np.column_stack((df["U:0"].values,      df["U:1"].values))
    p      = df["p"].values
    rows.append({"Simulation": carpeta.name, "Points": points, "U": u, "P": p})

df1 = pd.DataFrame(rows)
print(f"Simulaciones cargadas: {len(df1)}")

# ---------- CARGAR SDF --------------------------------------------------
sdf_df = pd.read_pickle(SDF_PKL)

# Lookup rápido: nombre de simulación -> matriz SDF
sdf_lookup = {
    row["Simulation"]: np.array(row["SDF"])
    for _, row in sdf_df.iterrows()
}

# ---------- DISCRETIZACIÓN DEL ESPACIO ----------------------------------
x_min, x_max     = -0.5, 1.5
y_min, y_max     = -0.5, 0.5
resolution_x     = 200
resolution_y     = 80

x_lin = np.linspace(x_min, x_max, resolution_x)
y_lin = np.linspace(y_min, y_max, resolution_y)
xx, yy = np.meshgrid(x_lin, y_lin)          # shape (80, 200)
space_points = np.column_stack([xx.ravel(), yy.ravel()])  # (16000, 2)

# ---------- INTERPOLACIÓN Y ENSAMBLADO ----------------------------------
records = []
for _, row in df1.iterrows():
    sim    = row["Simulation"]
    points = np.array(row["Points"])
    U      = np.array(row["U"])
    P      = np.array(row["P"])

    # Interpolación lineal sobre la malla uniforme
    Ux = griddata(points, U[:, 0], space_points, method='linear').reshape(resolution_y, resolution_x)
    Uy = griddata(points, U[:, 1], space_points, method='linear').reshape(resolution_y, resolution_x)
    Pp = griddata(points, P,       space_points, method='linear').reshape(resolution_y, resolution_x)

    # Voltear para que Y crezca hacia arriba en visualización
    Ux = np.flipud(Ux)
    Uy = np.flipud(Uy)
    Pp = np.flipud(Pp)

    sdf_matrix = sdf_lookup.get(sim)
    if sdf_matrix is None:
        print(f"SDF no encontrado para simulación: {sim}")
        continue

    # ── Gráfica SDF ───────────────────────────────────────────────────────
    # Convención: fila 0 = y_max (top) → origin='upper'
    # y_lin[::-1] en contour mapea fila 0 → y_max correctamente
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(sdf_matrix, origin='upper',
                   extent=[x_min, x_max, y_min, y_max],
                   cmap='RdBu_r', aspect='equal')
    ax.contour(x_lin, y_lin[::-1], sdf_matrix, levels=[0], colors='k', linewidths=1)
    plt.colorbar(im, ax=ax, label='SDF [m]')
    ax.set_title(f'SDF — {sim}')
    ax.set_xlabel('x [m]');  ax.set_ylabel('y [m]')
    plt.tight_layout()
    plt.savefig(PLOTS_SDF_DIR / f'{sim}.png', dpi=120)
    plt.close(fig)

    # Celdas dentro de la geometría (SDF < 0) → 0; NaN fuera del casco convexo → 0
    mask = sdf_matrix < 0
    Ux[mask] = 0.0;  Ux = np.nan_to_num(Ux)
    Uy[mask] = 0.0;  Uy = np.nan_to_num(Uy)
    Pp[mask] = 0.0;  Pp = np.nan_to_num(Pp)

    # ── Gráfica Ux / Uy / P ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    campos = [
        ("Ux [m/s]", Ux, 'RdBu_r'),
        ("Uy [m/s]", Uy, 'RdBu_r'),
        ("P [Pa]",   Pp, 'viridis'),
    ]
    for ax, (titulo, campo, cmap) in zip(axes, campos):
        im = ax.imshow(campo, origin='upper',
                       extent=[x_min, x_max, y_min, y_max],
                       cmap=cmap, aspect='equal')
        ax.contour(x_lin, y_lin[::-1], sdf_matrix, levels=[0], colors='k', linewidths=0.8)
        plt.colorbar(im, ax=ax, label=titulo)
        ax.set_title(f'{titulo} — {sim}')
        ax.set_xlabel('x [m]');  ax.set_ylabel('y [m]')
    plt.tight_layout()
    plt.savefig(PLOTS_CFD_DIR / f'{sim}.png', dpi=120)
    plt.close(fig)

    records.append({
        "SDF":            sdf_matrix,
        "Ux_discretized": Ux,
        "Uy_discretized": Uy,
        "P_discretized":  Pp,
    })

CNN_df = pd.DataFrame(records)

# ---------- EXPORTAR ----------------------------------------------------
CNN_df.to_pickle(OUTPUT_PKL)
print(f"Dataset guardado en {OUTPUT_PKL}  ({len(CNN_df)} simulaciones)")

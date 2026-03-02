"""
Evaluación del modelo entrenado sobre el conjunto de test.

Uso:
    cd original
    python -m src.inference.evaluate
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.models.unet_hybrid import unet_model_multi_output
from src.visualization.plot_fields import (
    plot_velocity_x, plot_velocity_y, plot_pressure,
)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(config_path="configs/config.yaml"):
    cfg       = load_config(config_path)
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]
    eval_cfg  = cfg.get("evaluation", {})

    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Cargar y normalizar datos (igual que train.py) ----
    CNN_df    = pd.read_pickle(paths_cfg["dataset"])
    sdf_array = np.array(CNN_df["SDF"].tolist())
    sdf_array = np.maximum(sdf_array, 0)   # SDF < 0 (interior airfoil) → 0 exacto
    U_x_array = np.array(CNN_df["Ux_discretized"].tolist())
    U_y_array = np.array(CNN_df["Uy_discretized"].tolist())
    P_array   = np.array(CNN_df["P_discretized"].tolist())

    min_Ux, max_Ux = np.min(U_x_array), np.max(U_x_array)
    min_Uy, max_Uy = np.min(U_y_array), np.max(U_y_array)
    min_P,  max_P  = np.min(P_array),   np.max(P_array)

    sdf_array = (sdf_array - np.min(sdf_array)) / (np.max(sdf_array) - np.min(sdf_array))
    U_x_array = (U_x_array - np.min(U_x_array)) / (np.max(U_x_array) - np.min(U_x_array))
    U_y_array = (U_y_array - np.min(U_y_array)) / (np.max(U_y_array) - np.min(U_y_array))
    P_array   = (P_array   - np.min(P_array))   / (np.max(P_array)   - np.min(P_array))

    # ---- Mismo split que entrenamiento ----
    X_train, X_test, y_train_x, y_test_x = train_test_split(
        sdf_array, U_x_array,
        test_size=train_cfg["test_size"], random_state=train_cfg["random_state"]
    )
    _, _, y_train_y,        y_test_y        = train_test_split(
        sdf_array, U_y_array,
        test_size=train_cfg["test_size"], random_state=train_cfg["random_state"]
    )
    _, _, y_train_pressure, y_test_pressure = train_test_split(
        sdf_array, P_array,
        test_size=train_cfg["test_size"], random_state=train_cfg["random_state"]
    )

    X_test           = np.expand_dims(X_test,           axis=-1)
    y_test_x         = np.expand_dims(y_test_x,         axis=-1)
    y_test_y         = np.expand_dims(y_test_y,         axis=-1)
    y_test_pressure  = np.expand_dims(y_test_pressure,  axis=-1)

    # ---- Umbral near-wall ----
    nw_cells = eval_cfg.get("nearwall_cells", 2)

    # ---- Cargar modelo ----
    input_shape = tuple(cfg["model"]["input_shape"])
    modelo_unet_multi = unet_model_multi_output(input_shape=input_shape)
    modelo_unet_multi.load_weights(paths_cfg["saved_model"])
    print(f"Modelo cargado desde: {paths_cfg['saved_model']}")
    print(f"Casos de test: {len(X_test)}")
    print(f"Near-wall: {nw_cells} × Δ por caso (Δ = mín. SDF no nulo de cada geometría)\n")

    # ---- Bucle completo de inferencia con temporización y métricas ----
    mae_ux_list,    mae_uy_list,    mae_p_list    = [], [], []
    nw_mae_ux_list, nw_mae_uy_list, nw_mae_p_list = [], [], []
    mabs_ux_list,   mabs_uy_list,   mabs_p_list   = [], [], []

    inicio = time.time()

    for i in range(len(X_test)):
        sdf_test = X_test[i]
        u_x_test = np.flipud(y_test_x[i])
        u_y_test = np.flipud(y_test_y[i])
        p_test_i = np.flipud(y_test_pressure[i])

        predicted_velocity_x, predicted_velocity_y, predicted_pressure = modelo_unet_multi.predict(
            sdf_test.reshape(1, *input_shape), verbose=0
        )

        U_x_pred = np.flipud(predicted_velocity_x[0])
        U_y_pred = np.flipud(predicted_velocity_y[0])
        P_pred   = np.flipud(predicted_pressure[0])

        sdf_vis  = np.flipud(np.squeeze(sdf_test))
        u_x_test = np.squeeze(u_x_test)
        u_y_test = np.squeeze(u_y_test)
        p_test_i = np.squeeze(p_test_i)

        # Desnormalizar las predicciones
        u_x_test = u_x_test * (max_Ux - min_Ux) + min_Ux
        u_y_test = u_y_test * (max_Uy - min_Uy) + min_Uy
        p_test_i = p_test_i * (max_P  - min_P)  + min_P

        U_x_pred = U_x_pred * (max_Ux - min_Ux) + min_Ux
        U_y_pred = U_y_pred * (max_Uy - min_Uy) + min_Uy
        P_pred   = P_pred   * (max_P  - min_P)  + min_P

        # Imponer geometría
        U_x_pred[sdf_vis == 0] = 0
        U_y_pred[sdf_vis == 0] = 0
        P_pred[sdf_vis == 0]   = 0

        U_x_pred = np.squeeze(U_x_pred)
        U_y_pred = np.squeeze(U_y_pred)
        P_pred   = np.squeeze(P_pred)

        # ---- Métricas por caso ----
        mae_ux_list.append(np.mean(np.abs(u_x_test - U_x_pred)))
        mae_uy_list.append(np.mean(np.abs(u_y_test - U_y_pred)))
        mae_p_list.append( np.mean(np.abs(p_test_i - P_pred)))

        mabs_ux_list.append(np.mean(np.abs(u_x_test)))
        mabs_uy_list.append(np.mean(np.abs(u_y_test)))
        mabs_p_list.append( np.mean(np.abs(p_test_i)))

        # Δ por caso: mínimo SDF fluido de esta geometría → umbral adaptado a su malla
        sdf_flow_vals = sdf_vis[sdf_vis > 1e-6]
        delta_case    = sdf_flow_vals.min() if sdf_flow_vals.size > 0 else 1e-4
        nw_threshold  = nw_cells * delta_case
        mask_nw = (sdf_vis > 1e-6) & (sdf_vis < nw_threshold)
        if mask_nw.any():
            nw_mae_ux_list.append(np.mean(np.abs((u_x_test - U_x_pred)[mask_nw])))
            nw_mae_uy_list.append(np.mean(np.abs((u_y_test - U_y_pred)[mask_nw])))
            nw_mae_p_list.append( np.mean(np.abs((p_test_i - P_pred)[mask_nw])))

        fig_ux = plot_velocity_x(U_x_pred, u_x_test)
        fig_ux.savefig(results_dir / f"case_{i:03d}_Ux.png", dpi=100)
        plt.close(fig_ux)

        fig_uy = plot_velocity_y(U_y_pred, u_y_test)
        fig_uy.savefig(results_dir / f"case_{i:03d}_Uy.png", dpi=100)
        plt.close(fig_uy)

        fig_p = plot_pressure(P_pred, p_test_i)
        fig_p.savefig(results_dir / f"case_{i:03d}_P.png", dpi=100)
        plt.close(fig_p)

    fin = time.time()
    t_IA = fin - inicio

    # ---- Reporte de métricas (MAE% según fórmula del paper) ----
    # MAE% = mean(|ŷ - y|) / mean(|y|) × 100
    mae_ux = np.mean(mae_ux_list);  mabs_ux = np.mean(mabs_ux_list)
    mae_uy = np.mean(mae_uy_list);  mabs_uy = np.mean(mabs_uy_list)
    mae_p  = np.mean(mae_p_list);   mabs_p  = np.mean(mabs_p_list)
    nw_ux  = np.mean(nw_mae_ux_list) if nw_mae_ux_list else float("nan")
    nw_uy  = np.mean(nw_mae_uy_list) if nw_mae_uy_list else float("nan")
    nw_p   = np.mean(nw_mae_p_list)  if nw_mae_p_list  else float("nan")

    print("=" * 57)
    print(f"RESULTADOS  (near-wall: {nw_cells}×Δ por caso)")
    print("=" * 57)
    print(f"{'Campo':<5}  {'MAE global':>18}  {'MAE near-wall':>18}")
    print(f"{'-'*5}  {'-'*18}  {'-'*18}")
    print(f"{'Ux':<5}  {mae_ux:.4f} m/s ({mae_ux/mabs_ux*100:.2f}%)  {nw_ux:.4f} m/s ({nw_ux/mabs_ux*100:.2f}%)")
    print(f"{'Uy':<5}  {mae_uy:.4f} m/s ({mae_uy/mabs_uy*100:.2f}%)  {nw_uy:.4f} m/s ({nw_uy/mabs_uy*100:.2f}%)")
    print(f"{'P':<5}  {mae_p:.4f} Pa  ({mae_p/mabs_p*100:.2f}%)   {nw_p:.4f} Pa  ({nw_p/mabs_p*100:.2f}%)")

    # ---- Cálculo de speedup (cell 14 del notebook) ----
    t_OF = eval_cfg.get("t_openfoam", 2770.09)
    print(f"\nTiempo total inferencia : {t_IA:.2f} s  ({len(X_test)} casos)")
    print(f"El tiempo de ejecución obtenido es {t_OF / t_IA:.2f} veces menor que empleando softwares convencionales")
    reduccion = ((t_OF - t_IA) / t_OF) * 100
    print(f"El tiempo de ejecución obtenido es un {reduccion:.2f}% menor que empleando softwares convencionales")
    print(f"Gráficas guardadas en: {results_dir.resolve()}")


if __name__ == "__main__":
    evaluate()

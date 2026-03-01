"""
Evaluación del modelo entrenado sobre el conjunto de test.

Uso:
    cd Hybrid_UNET
    python -m src.inference.evaluate
"""

import time
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")  # Sin pantalla — guarda a archivo
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.unet_hybrid import unet_model_multi_output
from src.inference.predict import predict_fields
from src.visualization.plot_fields import plot_comparison, plot_sdf, plot_training_history


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate(config_path="configs/config.yaml"):
    cfg       = load_config(config_path)
    paths_cfg = cfg["paths"]

    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Cargar datos ----
    df = pd.read_pickle(paths_cfg["dataset"])
    sdf_array = np.array(df["SDF"].tolist())
    ux_array  = np.array(df["Ux_discretized"].tolist())
    uy_array  = np.array(df["Uy_discretized"].tolist())
    p_array   = np.array(df["P_discretized"].tolist())

    # ---- Cargar índices y parámetros de normalización persistidos en train.py ----
    split   = np.load(paths_cfg["split_indices"])
    nparams = np.load(paths_cfg["norm_params"])
    idx_test = split["idx_test"]

    # Parámetros de normalización (fit sobre train únicamente → sin leakage)
    norm_params = {
        "Ux": (float(nparams["ux_mn"]), float(nparams["ux_mx"])),
        "Uy": (float(nparams["uy_mn"]), float(nparams["uy_mx"])),
        "P":  (float(nparams["p_mn"]),  float(nparams["p_mx"])),
    }

    def apply_norm(arr, mn, mx):
        return (arr - mn) / (mx - mn) if mx != mn else np.zeros_like(arr)

    sdf_norm = apply_norm(sdf_array, float(nparams["sdf_mn"]), float(nparams["sdf_mx"]))
    ux_norm  = apply_norm(ux_array,  norm_params["Ux"][0], norm_params["Ux"][1])
    uy_norm  = apply_norm(uy_array,  norm_params["Uy"][0], norm_params["Uy"][1])
    p_norm   = apply_norm(p_array,   norm_params["P"][0],  norm_params["P"][1])

    X_test  = sdf_norm[idx_test]
    ux_test = ux_norm[idx_test]
    uy_test = uy_norm[idx_test]
    p_test  = p_norm[idx_test]

    # ---- Cargar modelo ----
    model = unet_model_multi_output(input_shape=tuple(cfg["model"]["input_shape"]))
    model.load_weights(paths_cfg["saved_model"])
    print(f"Modelo cargado desde: {paths_cfg['saved_model']}")
    print(f"Casos de test: {len(X_test)}\n")

    # ---- Umbral near-wall: N × resolución espacial del grid ----
    # La resolución espacial Δ = mínimo valor no nulo del SDF normalizado,
    # que corresponde a la distancia del centro de la primera celda a la pared.
    nw_cells     = cfg.get("evaluation", {}).get("nearwall_cells", 2)
    delta        = sdf_norm[sdf_norm > 1e-8].min()      # Δ: 1 celda adyacente a la pared
    nw_threshold = nw_cells * delta
    print(f"Resolución espacial Δ : {delta:.6f}  (SDF normalizado)")
    print(f"Near-wall threshold   : SDF < {nw_threshold:.6f}  ({nw_cells} × Δ)\n")

    # ---- Predicción y métricas ----
    mae_ux_list,    mae_uy_list,    mae_p_list    = [], [], []
    nw_mae_ux_list, nw_mae_uy_list, nw_mae_p_list = [], [], []
    # mean(|Y|) por caso — denominador de la fórmula MAE% del paper
    mabs_ux_list,   mabs_uy_list,   mabs_p_list   = [], [], []
    t_inicio = time.time()

    for i in range(len(X_test)):
        fields, sdf_vis = predict_fields(model, X_test[i], norm_params)

        ux_true = np.flipud(np.squeeze(ux_test[i])) * (norm_params["Ux"][1] - norm_params["Ux"][0]) + norm_params["Ux"][0]
        uy_true = np.flipud(np.squeeze(uy_test[i])) * (norm_params["Uy"][1] - norm_params["Uy"][0]) + norm_params["Uy"][0]
        p_true  = np.flipud(np.squeeze(p_test[i]))  * (norm_params["P"][1]  - norm_params["P"][0])  + norm_params["P"][0]

        # Máscaras: dominio fluido (excluye airfoil) y near-wall
        mask_flow = sdf_vis > 1e-6
        mask_nw   = mask_flow & (sdf_vis < nw_threshold)

        # MAE global — solo sobre el dominio fluido (excluye interior airfoil)
        mae_ux_list.append(np.mean(np.abs((ux_true - fields["Ux"])[mask_flow])))
        mae_uy_list.append(np.mean(np.abs((uy_true - fields["Uy"])[mask_flow])))
        mae_p_list.append( np.mean(np.abs((p_true  - fields["P"])[mask_flow])))

        # mean(|Y|) sobre dominio fluido — denominador según fórmula del paper
        mabs_ux_list.append(np.mean(np.abs(ux_true[mask_flow])))
        mabs_uy_list.append(np.mean(np.abs(uy_true[mask_flow])))
        mabs_p_list.append( np.mean(np.abs(p_true[mask_flow])))

        if mask_nw.any():
            nw_mae_ux_list.append(np.mean(np.abs((ux_true - fields["Ux"])[mask_nw])))
            nw_mae_uy_list.append(np.mean(np.abs((uy_true - fields["Uy"])[mask_nw])))
            nw_mae_p_list.append( np.mean(np.abs((p_true  - fields["P"])[mask_nw])))

        # Guardar SDF del caso
        fig_sdf = plot_sdf(sdf_vis, case_label=f"case {i:03d}")
        fig_sdf.savefig(results_dir / f"case_{i:03d}_SDF.png", dpi=100)
        plt.close(fig_sdf)

        # Guardar gráfica — denominador = mean(|Y|) del caso (consistente con paper)
        mabs_case = {"Ux": mabs_ux_list[-1], "Uy": mabs_uy_list[-1], "P": mabs_p_list[-1]}
        nw_maes   = {"Ux": nw_mae_ux_list,   "Uy": nw_mae_uy_list,   "P": nw_mae_p_list}
        for field_key, true_arr, label, unit in [
            ("Ux", ux_true, "X-Velocity", "m/s"),
            ("Uy", uy_true, "Y-Velocity", "m/s"),
            ("P",  p_true,  "Pressure",   "Pa"),
        ]:
            denom      = mabs_case[field_key]
            nw_mae_pct = nw_maes[field_key][-1] / denom * 100 if mask_nw.any() else None
            fig = plot_comparison(
                fields[field_key], true_arr, label, unit,
                denom=denom, nearwall_mae=nw_mae_pct,
                mask=(sdf_vis == 0),
            )
            fig.savefig(results_dir / f"case_{i:03d}_{field_key}.png", dpi=100)
            plt.close(fig)

    t_total = time.time() - t_inicio

    # ---- Reporte de métricas (MAE% según fórmula del paper) ----
    # MAE% = mean(|ŷ - y|) / mean(|y|) × 100
    mae_ux = np.mean(mae_ux_list);  mabs_ux = np.mean(mabs_ux_list)
    mae_uy = np.mean(mae_uy_list);  mabs_uy = np.mean(mabs_uy_list)
    mae_p  = np.mean(mae_p_list);   mabs_p  = np.mean(mabs_p_list)
    nw_ux  = np.mean(nw_mae_ux_list) if nw_mae_ux_list else float("nan")
    nw_uy  = np.mean(nw_mae_uy_list) if nw_mae_uy_list else float("nan")
    nw_p   = np.mean(nw_mae_p_list)  if nw_mae_p_list  else float("nan")

    print("=" * 57)
    print(f"RESULTADOS  (near-wall: SDF < {nw_threshold:.6f} = {nw_cells}×Δ)")
    print("=" * 57)
    print(f"{'Campo':<5}  {'MAE global':>18}  {'MAE near-wall':>18}")
    print(f"{'-'*5}  {'-'*18}  {'-'*18}")
    print(f"{'Ux':<5}  {mae_ux:.4f} m/s ({mae_ux/mabs_ux*100:.2f}%)  {nw_ux:.4f} m/s ({nw_ux/mabs_ux*100:.2f}%)")
    print(f"{'Uy':<5}  {mae_uy:.4f} m/s ({mae_uy/mabs_uy*100:.2f}%)  {nw_uy:.4f} m/s ({nw_uy/mabs_uy*100:.2f}%)")
    print(f"{'P':<5}  {mae_p:.4f} Pa  ({mae_p/mabs_p*100:.2f}%)   {nw_p:.4f} Pa  ({nw_p/mabs_p*100:.2f}%)")
    print(f"\nTiempo total inferencia : {t_total:.2f} s  ({len(X_test)} casos)")
    print(f"Gráficas guardadas en   : {results_dir.resolve()}")


if __name__ == "__main__":
    evaluate()

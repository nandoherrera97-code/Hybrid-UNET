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
from src.visualization.plot_fields import plot_comparison, plot_sdf, plot_nearwall_distribution


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
    sdf_array = np.maximum(sdf_array, 0)   # SDF < 0 (interior airfoil) → 0 exacto
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

    # ---- Listado de índices de test ----
    print(f"Índices de test ({len(idx_test)}): {sorted(idx_test.tolist())}")
    print()

    # ---- Cargar modelo ----
    model = unet_model_multi_output(input_shape=tuple(cfg["model"]["input_shape"]))
    model.load_weights(paths_cfg["saved_model"])
    print(f"Modelo cargado desde: {paths_cfg['saved_model']}")
    print(f"Casos de test: {len(X_test)}\n")

    # ---- Umbral near-wall ----
    nw_cells = cfg.get("evaluation", {}).get("nearwall_cells", 2)
    print(f"Near-wall: {nw_cells} × Δ por caso (Δ = mín. SDF no nulo de cada geometría)\n")

    # ---- Rangos físicos para NMAE = MAE / (y_max - y_min) × 100 ----
    field_ranges = {
        "Ux": norm_params["Ux"][1] - norm_params["Ux"][0],
        "Uy": norm_params["Uy"][1] - norm_params["Uy"][0],
        "P":  norm_params["P"][1]  - norm_params["P"][0],
    }

    def safe_nmae(mae_value, value_range):
        """NMAE (%) = MAE / (y_max - y_min) × 100. Devuelve nan si rango=0."""
        return mae_value / value_range * 100 if value_range != 0 else float("nan")

    # ---- Predicción y métricas ----
    mae_ux_list,    mae_uy_list,    mae_p_list    = [], [], []
    nw_mae_ux_list, nw_mae_uy_list, nw_mae_p_list = [], [], []
    t_inicio = time.time()
    t_pred_list = []

    for i in range(len(X_test)):
        t0 = time.time()
        fields, sdf_vis = predict_fields(model, X_test[i], norm_params)
        t_pred_list.append(time.time() - t0)

        ux_true = np.flipud(np.squeeze(ux_test[i])) * (norm_params["Ux"][1] - norm_params["Ux"][0]) + norm_params["Ux"][0]
        uy_true = np.flipud(np.squeeze(uy_test[i])) * (norm_params["Uy"][1] - norm_params["Uy"][0]) + norm_params["Uy"][0]
        p_true  = np.flipud(np.squeeze(p_test[i]))  * (norm_params["P"][1]  - norm_params["P"][0])  + norm_params["P"][0]

        # Máscaras: dominio fluido (excluye airfoil) y near-wall
        mask_flow = sdf_vis > 1e-6
        # Δ por caso: mínimo SDF fluido de esta geometría → umbral adaptado a su malla
        sdf_flow_vals = sdf_vis[mask_flow]
        delta_case    = sdf_flow_vals.min() if sdf_flow_vals.size > 0 else 1e-4
        nw_threshold  = nw_cells * delta_case
        mask_nw       = mask_flow & (sdf_vis < nw_threshold)

        # MAE por caso — solo sobre el dominio fluido (excluye interior airfoil)
        mae_ux_list.append(np.mean(np.abs((ux_true - fields["Ux"])[mask_flow])))
        mae_uy_list.append(np.mean(np.abs((uy_true - fields["Uy"])[mask_flow])))
        mae_p_list.append( np.mean(np.abs((p_true  - fields["P"])[mask_flow])))

        if mask_nw.any():
            nw_mae_ux_list.append(np.mean(np.abs((ux_true - fields["Ux"])[mask_nw])))
            nw_mae_uy_list.append(np.mean(np.abs((uy_true - fields["Uy"])[mask_nw])))
            nw_mae_p_list.append( np.mean(np.abs((p_true  - fields["P"])[mask_nw])))

        sim_name = f"sim_{idx_test[i]:03d}"

        # Guardar SDF del caso
        fig_sdf = plot_sdf(sdf_vis, case_label=sim_name)
        fig_sdf.savefig(results_dir / f"{sim_name}_SDF.png", dpi=100)
        plt.close(fig_sdf)

        # Guardar gráficas con NMAE por caso como anotación
        nw_maes      = {"Ux": nw_mae_ux_list, "Uy": nw_mae_uy_list, "P": nw_mae_p_list}
        airfoil_mask = sdf_vis == 0

        # Rango de color compartido entre Ux, Uy, P (percentil 99 del mapa NMAE fluido)
        def _nmae_p99(true_arr, pred_arr, fr):
            vals = np.abs(true_arr - pred_arr)[mask_flow] / fr * 100 if fr != 0 else np.zeros(1)
            return float(np.percentile(vals, 99)) if vals.size > 0 else 0.0
        err_vmax = max(
            _nmae_p99(ux_true, fields["Ux"], field_ranges["Ux"]),
            _nmae_p99(uy_true, fields["Uy"], field_ranges["Uy"]),
            _nmae_p99(p_true,  fields["P"],  field_ranges["P"]),
        ) or None  # None → matplotlib autoscale si todos son 0

        for field_key, true_arr, label, unit in [
            ("Ux", ux_true, "X-Velocity", "m/s"),
            ("Uy", uy_true, "Y-Velocity", "m/s"),
            ("P",  p_true,  "Pressure",   "Pa"),
        ]:
            field_range    = field_ranges[field_key]
            mae_case_i     = {"Ux": mae_ux_list, "Uy": mae_uy_list, "P": mae_p_list}[field_key][-1]
            nmae_case      = safe_nmae(mae_case_i, field_range)
            nw_list        = nw_maes[field_key]
            nearwall_nmae  = safe_nmae(nw_list[-1], field_range) if mask_nw.any() and nw_list else None
            fig = plot_comparison(
                fields[field_key], true_arr, label, unit,
                field_range=field_range, nmae_case=nmae_case,
                nearwall_nmae=nearwall_nmae, mask=airfoil_mask, err_vmax=err_vmax,
            )
            fig.savefig(results_dir / f"{sim_name}_{field_key}.png", dpi=100)
            plt.close(fig)

    t_total = time.time() - t_inicio

    # ---- Reporte de métricas  NMAE = MAE / (y_max - y_min) × 100 ----
    mae_ux = np.mean(mae_ux_list)
    mae_uy = np.mean(mae_uy_list)
    mae_p  = np.mean(mae_p_list)
    nw_ux  = np.mean(nw_mae_ux_list) if nw_mae_ux_list else float("nan")
    nw_uy  = np.mean(nw_mae_uy_list) if nw_mae_uy_list else float("nan")
    nw_p   = np.mean(nw_mae_p_list)  if nw_mae_p_list  else float("nan")

    print("=" * 63)
    print(f"RESULTADOS  (near-wall: {nw_cells}×Δ por caso)")
    print("=" * 63)
    print(f"{'Campo':<5}  {'MAE global':>22}  {'MAE near-wall':>22}")
    print(f"{'-'*5}  {'-'*22}  {'-'*22}")
    print(f"{'Ux':<5}  {mae_ux:.4f} m/s ({safe_nmae(mae_ux, field_ranges['Ux']):.2f}% NMAE)  {nw_ux:.4f} m/s ({safe_nmae(nw_ux, field_ranges['Ux']):.2f}% NMAE)")
    print(f"{'Uy':<5}  {mae_uy:.4f} m/s ({safe_nmae(mae_uy, field_ranges['Uy']):.2f}% NMAE)  {nw_uy:.4f} m/s ({safe_nmae(nw_uy, field_ranges['Uy']):.2f}% NMAE)")
    print(f"{'P':<5}  {mae_p:.4f} Pa  ({safe_nmae(mae_p,  field_ranges['P']):.2f}% NMAE)   {nw_p:.4f} Pa  ({safe_nmae(nw_p,  field_ranges['P']):.2f}% NMAE)")
    t_pred_total = sum(t_pred_list)
    t_pred_mean  = np.mean(t_pred_list)
    print(f"\nTiempo predicción total : {t_pred_total:.3f} s  ({len(X_test)} casos)")
    print(f"Tiempo predicción/caso  : {t_pred_mean*1000:.1f} ms")
    print(f"Tiempo total inferencia : {t_total:.2f} s  (incl. métricas y gráficas)")
    print(f"Gráficas guardadas en   : {results_dir.resolve()}")

    # ---- Distribución near-wall NMAE (un valor por caso) ----
    if nw_mae_ux_list:
        fig_dist = plot_nearwall_distribution(
            np.array([safe_nmae(v, field_ranges["Ux"]) for v in nw_mae_ux_list]),
            np.array([safe_nmae(v, field_ranges["Uy"]) for v in nw_mae_uy_list]),
            np.array([safe_nmae(v, field_ranges["P"])  for v in nw_mae_p_list]),
        )
        fig_dist.savefig(results_dir / "nearwall_error_distribution.png", dpi=120)
        fig_dist.savefig(results_dir / "nearwall_error_distribution.pdf")
        plt.close(fig_dist)
        print(f"Distribución near-wall  : {results_dir / 'nearwall_error_distribution.png/.pdf'}")


if __name__ == "__main__":
    evaluate()

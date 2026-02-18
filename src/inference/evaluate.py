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
from sklearn.model_selection import train_test_split

from src.models.unet_hybrid import unet_model_multi_output
from src.inference.predict import predict_fields
from src.visualization.plot_fields import plot_comparison, plot_training_history


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate(config_path="configs/config.yaml"):
    cfg       = load_config(config_path)
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]

    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Cargar y normalizar datos (igual que train.py) ----
    df = pd.read_pickle(paths_cfg["dataset"])
    sdf_array = np.array(df["SDF"].tolist())
    ux_array  = np.array(df["Ux_discretized"].tolist())
    uy_array  = np.array(df["Uy_discretized"].tolist())
    p_array   = np.array(df["P_discretized"].tolist())

    norm_params = {
        "Ux": (ux_array.min(), ux_array.max()),
        "Uy": (uy_array.min(), uy_array.max()),
        "P":  (p_array.min(),  p_array.max()),
    }

    def normalize(arr):
        return (arr - arr.min()) / (arr.max() - arr.min())

    sdf_norm = normalize(sdf_array)
    ux_norm  = normalize(ux_array)
    uy_norm  = normalize(uy_array)
    p_norm   = normalize(p_array)

    # ---- Mismo split que entrenamiento ----
    rs, ts = train_cfg["random_state"], train_cfg["test_size"]
    X_train, X_test, ux_train, ux_test = train_test_split(sdf_norm, ux_norm, test_size=ts, random_state=rs)
    _, _,            uy_train, uy_test  = train_test_split(sdf_norm, uy_norm,  test_size=ts, random_state=rs)
    _, _,            p_train,  p_test   = train_test_split(sdf_norm, p_norm,   test_size=ts, random_state=rs)

    # ---- Cargar modelo ----
    model = unet_model_multi_output(input_shape=tuple(cfg["model"]["input_shape"]))
    model.load_weights(paths_cfg["saved_model"])
    print(f"Modelo cargado desde: {paths_cfg['saved_model']}")
    print(f"Casos de test: {len(X_test)}\n")

    # ---- Predicción y métricas ----
    mae_ux_list, mae_uy_list, mae_p_list = [], [], []
    t_inicio = time.time()

    for i in range(len(X_test)):
        fields, sdf_vis = predict_fields(model, X_test[i], norm_params)

        ux_true = np.flipud(np.squeeze(ux_test[i])) * (norm_params["Ux"][1] - norm_params["Ux"][0]) + norm_params["Ux"][0]
        uy_true = np.flipud(np.squeeze(uy_test[i])) * (norm_params["Uy"][1] - norm_params["Uy"][0]) + norm_params["Uy"][0]
        p_true  = np.flipud(np.squeeze(p_test[i]))  * (norm_params["P"][1]  - norm_params["P"][0])  + norm_params["P"][0]

        mae_ux_list.append(np.mean(np.abs(ux_true - fields["Ux"])))
        mae_uy_list.append(np.mean(np.abs(uy_true - fields["Uy"])))
        mae_p_list.append(np.mean(np.abs(p_true  - fields["P"])))

        # Guardar gráfica de comparación para cada caso
        for field_key, true_arr, label, unit in [
            ("Ux", ux_true, "X-Velocity", "m/s"),
            ("Uy", uy_true, "Y-Velocity", "m/s"),
            ("P",  p_true,  "Pressure",   "Pa"),
        ]:
            fig = plot_comparison(fields[field_key], true_arr, label, unit)
            fig.savefig(results_dir / f"case_{i:03d}_{field_key}.png", dpi=100)
            plt.close(fig)

    t_total = time.time() - t_inicio

    # ---- Reporte de métricas ----
    mae_ux = np.mean(mae_ux_list)
    mae_uy = np.mean(mae_uy_list)
    mae_p  = np.mean(mae_p_list)

    rng_ux = norm_params["Ux"][1] - norm_params["Ux"][0]
    rng_uy = norm_params["Uy"][1] - norm_params["Uy"][0]
    rng_p  = norm_params["P"][1]  - norm_params["P"][0]

    print("=" * 45)
    print("RESULTADOS FINALES")
    print("=" * 45)
    print(f"MAE  Ux : {mae_ux:.4f} m/s  ({mae_ux/rng_ux*100:.2f}%)")
    print(f"MAE  Uy : {mae_uy:.4f} m/s  ({mae_uy/rng_uy*100:.2f}%)")
    print(f"MAE  P  : {mae_p:.4f} Pa   ({mae_p/rng_p*100:.2f}%)")
    print(f"\nTiempo total inferencia : {t_total:.2f} s  ({len(X_test)} casos)")
    print(f"Gráficas guardadas en   : {results_dir.resolve()}")


if __name__ == "__main__":
    evaluate()

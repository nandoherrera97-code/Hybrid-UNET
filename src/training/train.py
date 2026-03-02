"""
Entrenamiento del modelo original (sin weighted loss).

Uso:
    cd original
    python -m src.training.train
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.models.unet_hybrid import unet_model_multi_output


def load_config(config_path="configs/config.yaml"):
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config_path="configs/config.yaml"):
    cfg       = load_config(config_path)
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]

    # --------------------- CARGAR DATAFRAME -------------------
    CNN_df = pd.read_pickle(paths_cfg["dataset"])

    # --------------------- CARGAR DATOS -----------------------------------
    sdf_data = CNN_df["SDF"]
    U_data_x = CNN_df["Ux_discretized"]
    U_data_y = CNN_df["Uy_discretized"]
    P_data   = CNN_df["P_discretized"]

    # ------------------------------- NORMALIZACIÓN DE DATOS ------------------------------

    # Convertir los datos de pandas Series a listas de NumPy arrays
    sdf_array = np.array(sdf_data.tolist())
    sdf_array = np.maximum(sdf_array, 0)   # SDF < 0 (interior airfoil) → 0 exacto
    U_x_array = np.array(U_data_x.tolist())
    U_y_array = np.array(U_data_y.tolist())
    P_array   = np.array(P_data.tolist())

    # Obtener los valores originales de min y max usados en la normalización
    min_Ux, max_Ux = np.min(U_x_array), np.max(U_x_array)
    min_Uy, max_Uy = np.min(U_y_array), np.max(U_y_array)
    min_P,  max_P  = np.min(P_array),   np.max(P_array)

    # Normalización de los datos entre 0 y 1
    sdf_array = (sdf_array - np.min(sdf_array)) / (np.max(sdf_array) - np.min(sdf_array))
    U_x_array = (U_x_array - np.min(U_x_array)) / (np.max(U_x_array) - np.min(U_x_array))
    U_y_array = (U_y_array - np.min(U_y_array)) / (np.max(U_y_array) - np.min(U_y_array))
    P_array   = (P_array   - np.min(P_array))   / (np.max(P_array)   - np.min(P_array))

    # ------------------- Split 70 / 15 / 15 (train / val / test) ----------------------------
    # Se usan índices para garantizar que los tres campos (Ux, Uy, P) compartan
    # exactamente los mismos casos en cada partición.

    n = len(sdf_array)
    indices = np.arange(n)

    # 1) Separar test hold-out (15 % del total)
    idx_trainval, idx_test = train_test_split(
        indices,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"]
    )

    # 2) Separar validación del resto  →  val_size/( 1 - test_size ) del trainval
    val_frac = train_cfg["val_size"] / (1.0 - train_cfg["test_size"])
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_frac,
        random_state=train_cfg["random_state"]
    )

    X_train = np.expand_dims(sdf_array[idx_train], axis=-1)
    X_val   = np.expand_dims(sdf_array[idx_val],   axis=-1)
    X_test  = np.expand_dims(sdf_array[idx_test],  axis=-1)

    y_train_x        = np.expand_dims(U_x_array[idx_train], axis=-1)
    y_val_x          = np.expand_dims(U_x_array[idx_val],   axis=-1)
    y_test_x         = np.expand_dims(U_x_array[idx_test],  axis=-1)

    y_train_y        = np.expand_dims(U_y_array[idx_train], axis=-1)
    y_val_y          = np.expand_dims(U_y_array[idx_val],   axis=-1)
    y_test_y         = np.expand_dims(U_y_array[idx_test],  axis=-1)

    y_train_pressure = np.expand_dims(P_array[idx_train], axis=-1)
    y_val_pressure   = np.expand_dims(P_array[idx_val],   axis=-1)
    y_test_pressure  = np.expand_dims(P_array[idx_test],  axis=-1)

    print(f"Split  →  train: {len(idx_train)}  val: {len(idx_val)}  test: {len(idx_test)}")

    # ---------------- DEFINIR EL MODELO -----------------------------
    modelo_unet_multi = unet_model_multi_output(
        input_shape=tuple(cfg["model"]["input_shape"]),
        dropout_spatial=cfg["model"].get("dropout_spatial", 0.1),
        dropout_dense=cfg["model"].get("dropout_dense",   0.3),
    )

    # ---------------- COMPILAR EL MODELO -----------------------------
    modelo_unet_multi.compile(
        optimizer='adam',
        loss=['mse', 'mse', 'mse'],
        metrics=[['mae'], ['mae'], ['mae']]
    )

    # Callback de EarlyStopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=train_cfg["patience"], restore_best_weights=True
    )

    # --------------------  ENTRENAR MODELO ------------------------------------

    # Entrenamiento del modelo con una entrada (SDF) y tres salidas (campos de velocidad y presión)
    t_inicio = time.time()
    history = modelo_unet_multi.fit(
        X_train,
        [y_train_x, y_train_y, y_train_pressure],
        validation_data=(X_val, [y_val_x, y_val_y, y_val_pressure]),
        epochs=train_cfg["epochs"],
        callbacks=[early_stopping]
    )
    t_total = time.time() - t_inicio
    print(f"Tiempo total entrenamiento: {t_total:.1f} s  ({t_total/60:.1f} min)")

    # Guardar modelo
    saved_model_path = Path(paths_cfg["saved_model"])
    saved_model_path.parent.mkdir(parents=True, exist_ok=True)
    modelo_unet_multi.save_weights(str(saved_model_path))
    print(f"Modelo guardado en: {saved_model_path}")

    # Obtener los valores finales de las métricas de validación
    final_val_loss  = history.history['val_loss'][-1]
    final_val_mae_x = history.history['val_output_1_mae'][-1]
    final_val_mae_y = history.history['val_output_2_mae'][-1]
    final_val_mae_p = history.history['val_output_3_mae'][-1]

    mae_porcentaje_u_x      = (final_val_mae_x / y_val_x.max())        * 100
    mae_porcentaje_u_y      = (final_val_mae_y / y_val_y.max())        * 100
    mae_porcentaje_pressure = (final_val_mae_p / y_val_pressure.max()) * 100

    # Imprimir resultados finales
    print("Resultados Finales del Entrenamiento:")
    print(f"Val Loss Final: {final_val_loss:.4f}")
    print(f"Val MAE para la salida X (Velocidad horizontal): {final_val_mae_x:.4f}")
    print(f"Val MAE para la salida Y (Velocidad vertical): {final_val_mae_y:.4f}")
    print(f"Val MAE para la salida P (Presión): {final_val_mae_p:.4f}")
    print(f"Val MAE(%) para la salida ux (Velocidad horizontal): {mae_porcentaje_u_x:.4f}")
    print(f"Val MAE(%) para la salida UY (Velocidad vertical): {mae_porcentaje_u_y:.4f}")
    print(f"Val MAE(%) para la salida P (Presión): {mae_porcentaje_pressure:.4f}")

    # Guardar curva de entrenamiento
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"],     label="Train loss")
    ax.plot(history.history["val_loss"], label="Val loss")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida (MSE)")
    ax.legend()
    ax.set_title("Curva de entrenamiento")
    plt.tight_layout()
    fig.savefig(results_dir / "training_history.png", dpi=100)
    plt.close(fig)
    print(f"Curva de entrenamiento guardada en: {results_dir / 'training_history.png'}")


if __name__ == "__main__":
    train()

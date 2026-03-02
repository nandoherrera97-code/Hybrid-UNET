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

    # ------------------- Dividir el dataset en entrenamiento y prueba ----------------------------

    # Dividir los datos en entrenamiento y prueba para las tres salidas
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

    X_test            = np.expand_dims(X_test,            axis=-1)
    X_train           = np.expand_dims(X_train,           axis=-1)
    y_train_x         = np.expand_dims(y_train_x,         axis=-1)
    y_train_y         = np.expand_dims(y_train_y,         axis=-1)
    y_train_pressure  = np.expand_dims(y_train_pressure,  axis=-1)
    y_test_pressure   = np.expand_dims(y_test_pressure,   axis=-1)
    y_test_x          = np.expand_dims(y_test_x,          axis=-1)
    y_test_y          = np.expand_dims(y_test_y,          axis=-1)

    # Verificar tipo y forma
    print(type(X_train), X_train.shape)
    print(type(y_train_x), y_train_x.shape)
    print(type(y_train_y), y_train_y.shape)
    print(type(y_train_pressure), y_train_pressure.shape)
    print(type(y_test_pressure), y_test_pressure.shape)

    # ---------------- DEFINIR EL MODELO -----------------------------
    modelo_unet_multi = unet_model_multi_output(input_shape=tuple(cfg["model"]["input_shape"]))

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
        validation_data=(X_test, [y_test_x, y_test_y, y_test_pressure]),
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

    mae_porcentaje_u_x      = (final_val_mae_x / y_test_y.max())        * 100
    mae_porcentaje_u_y      = (final_val_mae_y / y_test_y.max())        * 100
    mae_porcentaje_pressure = (final_val_mae_p / y_test_pressure.max()) * 100

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

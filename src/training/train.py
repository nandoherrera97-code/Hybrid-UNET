import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"   # Silencia logs INFO y WARNING de TF

import numpy as np
import tensorflow as tf
import yaml
from pathlib import Path

from src.models.unet_hybrid import unet_model_multi_output
from src.training.losses import weighted_mse, compute_sdf_weights


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]

    # ---- Cargar datos ----
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_pickle(paths_cfg["dataset"])
    sdf_array = np.array(df["SDF"].tolist())
    ux_array  = np.array(df["Ux_discretized"].tolist())
    uy_array  = np.array(df["Uy_discretized"].tolist())
    p_array   = np.array(df["P_discretized"].tolist())

    # Split 3-vías usando índices — ANTES de normalizar para evitar leakage
    rs       = train_cfg["random_state"]
    ts_test  = train_cfg["test_size"]
    ts_val   = train_cfg["val_size"]
    val_frac = ts_val / (1.0 - ts_test)   # fracción de val relativa al conjunto trainval

    idx = np.arange(len(sdf_array))

    # 1. Separar test hold-out (nunca usado durante el entrenamiento)
    idx_trainval, idx_test = train_test_split(idx, test_size=ts_test, random_state=rs, shuffle=True)

    # 2. Separar validación del resto (trainval → train + val)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=val_frac, random_state=rs, shuffle=True)

    print(f"Split: {len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test hold-out")

    # Persistir índices para reproducibilidad exacta en evaluate.py
    save_path_split = Path(paths_cfg["split_indices"])
    save_path_split.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path_split, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    # Normalización: fit solo sobre entrenamiento → sin leakage de val/test
    def fit_norm(arr, idx):
        mn, mx = arr[idx].min(), arr[idx].max()
        return mn, mx

    def apply_norm(arr, mn, mx):
        return (arr - mn) / (mx - mn) if mx != mn else np.zeros_like(arr)

    sdf_mn, sdf_mx = fit_norm(sdf_array, idx_train)
    ux_mn,  ux_mx  = fit_norm(ux_array,  idx_train)
    uy_mn,  uy_mx  = fit_norm(uy_array,  idx_train)
    p_mn,   p_mx   = fit_norm(p_array,   idx_train)

    # Persistir parámetros de normalización para evaluate.py y predict.py
    np.savez(paths_cfg["norm_params"],
             sdf_mn=sdf_mn, sdf_mx=sdf_mx,
             ux_mn=ux_mn,   ux_mx=ux_mx,
             uy_mn=uy_mn,   uy_mx=uy_mx,
             p_mn=p_mn,     p_mx=p_mx)

    sdf_norm = apply_norm(sdf_array, sdf_mn, sdf_mx)
    ux_norm  = apply_norm(ux_array,  ux_mn,  ux_mx)
    uy_norm  = apply_norm(uy_array,  uy_mn,  uy_mx)
    p_norm   = apply_norm(p_array,   p_mn,   p_mx)

    # Aplicar índices a todos los arrays de una vez
    X_train  = sdf_norm[idx_train];  X_val  = sdf_norm[idx_val]
    ux_train = ux_norm[idx_train];   ux_val = ux_norm[idx_val]
    uy_train = uy_norm[idx_train];   uy_val = uy_norm[idx_val]
    p_train  = p_norm[idx_train];    p_val  = p_norm[idx_val]

    X_train  = np.expand_dims(X_train,  -1).astype(np.float32)
    X_val    = np.expand_dims(X_val,    -1).astype(np.float32)
    ux_train = np.expand_dims(ux_train, -1).astype(np.float32)
    ux_val   = np.expand_dims(ux_val,   -1).astype(np.float32)
    uy_train = np.expand_dims(uy_train, -1).astype(np.float32)
    uy_val   = np.expand_dims(uy_val,   -1).astype(np.float32)
    p_train  = np.expand_dims(p_train,  -1).astype(np.float32)
    p_val    = np.expand_dims(p_val,    -1).astype(np.float32)

    # ---- Modelo ----
    model = unet_model_multi_output(input_shape=tuple(cfg["model"]["input_shape"]))
    optimizer = tf.keras.optimizers.Adam(learning_rate=train_cfg["learning_rate"])

    # ---- Datasets ----
    batch_size = train_cfg["batch_size"]
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, ux_train, uy_train, p_train))
        .shuffle(len(X_train))
        .batch(batch_size)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, ux_val, uy_val, p_val)).batch(batch_size)

    # ---- Bucle de entrenamiento ----
    alpha   = train_cfg["loss_alpha"]
    patience = train_cfg["patience"]
    best_val = float("inf")
    wait = 0
    history = {"loss": [], "val_loss": []}
    save_path = Path(paths_cfg["saved_model"])

    for epoch in range(train_cfg["epochs"]):
        # Entrenamiento
        train_losses = []
        for x_b, ux_b, uy_b, p_b in train_ds:
            w = compute_sdf_weights(x_b, alpha=alpha)
            with tf.GradientTape() as tape:
                ux_p, uy_p, p_p = model(x_b, training=True)
                loss = (weighted_mse(ux_b, ux_p, w) +
                        weighted_mse(uy_b, uy_p, w) +
                        weighted_mse(p_b,  p_p,  w))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_losses.append(loss.numpy())

        # Validación
        val_losses = []
        for x_b, ux_b, uy_b, p_b in val_ds:
            w = compute_sdf_weights(x_b, alpha=alpha)
            ux_p, uy_p, p_p = model(x_b, training=False)
            val_loss = (weighted_mse(ux_b, ux_p, w) +
                        weighted_mse(uy_b, uy_p, w) +
                        weighted_mse(p_b,  p_p,  w))
            val_losses.append(val_loss.numpy())

        mean_train = np.mean(train_losses)
        mean_val   = np.mean(val_losses)
        history["loss"].append(mean_train)
        history["val_loss"].append(mean_val)

        total_epochs = train_cfg["epochs"]
        marker = " *" if mean_val < best_val else ""
        print(f"Epoch {epoch+1:4d}/{total_epochs} — loss: {mean_train:.5f}  val_loss: {mean_val:.5f}{marker}")

        # Early stopping
        if mean_val < best_val:
            best_val = mean_val
            model.save_weights(str(save_path))
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\nEarly stopping en epoch {epoch + 1}")
                break

    model.load_weights(str(save_path))
    print(f"\nMejor val_loss: {best_val:.5f}")
    return model, history


if __name__ == "__main__":
    train()

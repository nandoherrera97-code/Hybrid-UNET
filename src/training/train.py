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

    # Normalización
    def normalize(arr):
        return (arr - arr.min()) / (arr.max() - arr.min())

    sdf_array = normalize(sdf_array)
    ux_array  = normalize(ux_array)
    uy_array  = normalize(uy_array)
    p_array   = normalize(p_array)

    # Split
    rs = train_cfg["random_state"]
    ts = train_cfg["test_size"]
    X_train, X_test, ux_train, ux_test = train_test_split(sdf_array, ux_array, test_size=ts, random_state=rs)
    _, _,           uy_train, uy_test  = train_test_split(sdf_array, uy_array,  test_size=ts, random_state=rs)
    _, _,           p_train,  p_test   = train_test_split(sdf_array, p_array,   test_size=ts, random_state=rs)

    X_train  = np.expand_dims(X_train,  -1).astype(np.float32)
    X_test   = np.expand_dims(X_test,   -1).astype(np.float32)
    ux_train = np.expand_dims(ux_train, -1).astype(np.float32)
    ux_test  = np.expand_dims(ux_test,  -1).astype(np.float32)
    uy_train = np.expand_dims(uy_train, -1).astype(np.float32)
    uy_test  = np.expand_dims(uy_test,  -1).astype(np.float32)
    p_train  = np.expand_dims(p_train,  -1).astype(np.float32)
    p_test   = np.expand_dims(p_test,   -1).astype(np.float32)

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
    val_ds = tf.data.Dataset.from_tensor_slices((X_test, ux_test, uy_test, p_test)).batch(batch_size)

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

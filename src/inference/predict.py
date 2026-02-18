import numpy as np
import tensorflow as tf

from src.models.unet_hybrid import unet_model_multi_output


def load_model(weights_path, input_shape=(80, 200, 1)):
    model = unet_model_multi_output(input_shape=input_shape)
    model.load_weights(weights_path)
    return model


def predict_fields(model, sdf_norm, norm_params):
    """
    Realiza la predicción y desnormaliza los campos.

    Args:
        model      : modelo Keras cargado
        sdf_norm   : array (80, 200) SDF normalizado
        norm_params: dict con min/max de cada campo
                     {"Ux": (min, max), "Uy": (min, max), "P": (min, max)}

    Returns:
        dict con campos desnormalizados: {"Ux", "Uy", "P"}
        y el SDF invertido para visualización
    """
    sdf_input = sdf_norm.reshape(1, *sdf_norm.shape, 1)
    ux_pred, uy_pred, p_pred = model.predict(sdf_input, verbose=0)

    def denorm(arr, key):
        mn, mx = norm_params[key]
        return arr * (mx - mn) + mn

    sdf_vis  = np.flipud(sdf_norm)
    ux_field = np.flipud(np.squeeze(denorm(ux_pred[0], "Ux")))
    uy_field = np.flipud(np.squeeze(denorm(uy_pred[0], "Uy")))
    p_field  = np.flipud(np.squeeze(denorm(p_pred[0],  "P")))

    # Imponer condición de contorno: velocidad y presión = 0 dentro del perfil
    mask = sdf_vis == 0
    ux_field[mask] = 0
    uy_field[mask] = 0
    p_field[mask]  = 0

    return {"Ux": ux_field, "Uy": uy_field, "P": p_field}, sdf_vis

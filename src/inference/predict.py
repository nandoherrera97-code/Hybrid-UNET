import numpy as np


def predict_fields(model, sdf_test, norm_params):
    """
    Realiza la predicción y desnormaliza los campos.

    Args:
        model      : modelo Keras cargado
        sdf_test   : array (80, 200, 1) SDF normalizado (una muestra del test set)
        norm_params: dict con min/max de cada campo
                     {"Ux": (min, max), "Uy": (min, max), "P": (min, max)}

    Returns:
        dict con campos desnormalizados: {"Ux", "Uy", "P"}
        y el SDF invertido para visualización
    """
    min_Ux, max_Ux = norm_params["Ux"]
    min_Uy, max_Uy = norm_params["Uy"]
    min_P,  max_P  = norm_params["P"]

    predicted_velocity_x, predicted_velocity_y, predicted_pressure = model.predict(
        sdf_test.reshape(1, 80, 200, 1), verbose=0
    )

    U_x_pred = np.flipud(predicted_velocity_x[0])
    U_y_pred = np.flipud(predicted_velocity_y[0])
    P_pred   = np.flipud(predicted_pressure[0])

    sdf_vis = np.flipud(np.squeeze(sdf_test))

    # Desnormalizar las predicciones
    U_x_pred = U_x_pred * (max_Ux - min_Ux) + min_Ux
    U_y_pred = U_y_pred * (max_Uy - min_Uy) + min_Uy
    P_pred   = P_pred   * (max_P  - min_P)  + min_P

    # Imponer condición de contorno: velocidad y presión = 0 dentro del perfil
    U_x_pred[sdf_vis == 0] = 0
    U_y_pred[sdf_vis == 0] = 0
    P_pred[sdf_vis == 0]   = 0

    U_x_pred = np.squeeze(U_x_pred)
    U_y_pred = np.squeeze(U_y_pred)
    P_pred   = np.squeeze(P_pred)

    return {"Ux": U_x_pred, "Uy": U_y_pred, "P": P_pred}, sdf_vis

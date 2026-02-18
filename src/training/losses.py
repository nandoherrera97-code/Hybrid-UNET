import tensorflow as tf


def weighted_mse(y_true, y_pred, weights):
    """MSE ponderado píxel a píxel por un mapa de pesos."""
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))


def weighted_mae(y_true, y_pred, weights):
    """MAE ponderado píxel a píxel por un mapa de pesos."""
    return tf.reduce_mean(weights * tf.abs(y_true - y_pred))


def compute_sdf_weights(sdf_batch, alpha=3.0, epsilon=1e-6):
    """
    Calcula el mapa de pesos a partir del campo SDF normalizado.

    Las zonas near-wall (SDF ≈ 0⁺) reciben peso alto.
    El interior del perfil (SDF = 0) queda enmascarado.

    Args:
        sdf_batch : tensor (batch, H, W, 1), SDF normalizado en [0, 1]
        alpha     : controla la caída del peso con la distancia (1–10)
        epsilon   : estabilidad numérica

    Returns:
        weights   : tensor de misma forma que sdf_batch
    """
    alpha   = tf.cast(alpha,   sdf_batch.dtype)
    epsilon = tf.cast(epsilon, sdf_batch.dtype)
    mask    = tf.cast(sdf_batch > epsilon, sdf_batch.dtype)
    weights = tf.exp(-alpha * sdf_batch) * mask
    weights = weights / (tf.reduce_mean(weights) + epsilon)
    return weights

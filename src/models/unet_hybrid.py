import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Conv2D, Input, Conv2DTranspose, concatenate,
    Flatten, Dense, Reshape
)


def unet_model_multi_output(input_shape=(80, 200, 1)):
    """
    Hybrid CNN + U-Net con tres salidas independientes.

    Entrada:
        sdf: campo Signed Distance Function normalizado (80x200x1)

    Salidas:
        output_1: campo de velocidad Ux
        output_2: campo de velocidad Uy
        output_3: campo de presión P
    """
    inputs = Input(shape=input_shape)

    # ---- Encoder (compartido) ----
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    # ---- Bottleneck + transición CNN → U-Net ----
    b = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    b = Conv2D(128, (3, 3), activation='relu', padding='same')(b)
    flat = Flatten()(b)
    dense = Dense(10 * 25 * 32, activation='relu')(flat)
    reshaped = Reshape((10, 25, 32))(dense)

    # ---- Decoder Ux ----
    ux3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(reshaped)
    ux3 = concatenate([ux3, c3])
    cx4 = Conv2D(64, (3, 3), activation='relu', padding='same')(ux3)
    cx4 = Conv2D(64, (3, 3), activation='relu', padding='same')(cx4)

    ux2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(cx4)
    ux2 = concatenate([ux2, c2])
    cx5 = Conv2D(32, (3, 3), activation='relu', padding='same')(ux2)
    cx5 = Conv2D(32, (3, 3), activation='relu', padding='same')(cx5)

    ux1 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(cx5)
    ux1 = concatenate([ux1, c1])
    cx6 = Conv2D(16, (3, 3), activation='relu', padding='same')(ux1)
    cx6 = Conv2D(16, (3, 3), activation='relu', padding='same')(cx6)

    # ---- Decoder Uy ----
    uy3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(reshaped)
    uy3 = concatenate([uy3, c3])
    cy4 = Conv2D(64, (3, 3), activation='relu', padding='same')(uy3)
    cy4 = Conv2D(64, (3, 3), activation='relu', padding='same')(cy4)

    uy2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(cy4)
    uy2 = concatenate([uy2, c2])
    cy5 = Conv2D(32, (3, 3), activation='relu', padding='same')(uy2)
    cy5 = Conv2D(32, (3, 3), activation='relu', padding='same')(cy5)

    uy1 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(cy5)
    uy1 = concatenate([uy1, c1])
    cy6 = Conv2D(16, (3, 3), activation='relu', padding='same')(uy1)
    cy6 = Conv2D(16, (3, 3), activation='relu', padding='same')(cy6)

    # ---- Decoder P ----
    up3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(reshaped)
    up3 = concatenate([up3, c3])
    cp4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up3)
    cp4 = Conv2D(64, (3, 3), activation='relu', padding='same')(cp4)

    up2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(cp4)
    up2 = concatenate([up2, c2])
    cp5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    cp5 = Conv2D(32, (3, 3), activation='relu', padding='same')(cp5)

    up1 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(cp5)
    up1 = concatenate([up1, c1])
    cp6 = Conv2D(16, (3, 3), activation='relu', padding='same')(up1)
    cp6 = Conv2D(16, (3, 3), activation='relu', padding='same')(cp6)

    # ---- Salidas ----
    output_ux = Conv2D(1, (1, 1), activation='linear', padding='same', name='output_1')(cx6)
    output_uy = Conv2D(1, (1, 1), activation='linear', padding='same', name='output_2')(cy6)
    output_p  = Conv2D(1, (1, 1), activation='linear', padding='same', name='output_3')(cp6)

    return Model(inputs=inputs, outputs=[output_ux, output_uy, output_p])

#!/usr/bin/env python3
"""Module that creates a convolutional encoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """creates convolutional autoencoder
    Args:
        input_dims:is a tuple of integers containing the dimension od the model
        filters: list containing number of filters for each convolutional layer
        latent_dims: tuple of integerrs containing the dimension of the latent
            space representation

    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder eis the decoder model
        auto is the full autoencoder model
    """
    input = keras.layers.Input(shape=input_dims)
    x = input
    for f in filters:
        x = keras.layers.Conv2D(f, kernel_size=(3, 3), padding='same',
                                activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    output = x

    encoder = keras.Model(input, output)

    # Decoder model

    input_decod = keras.layers.Input(shape=latent_dims)
    x = input_decod
    for f in reversed(filters[1:]):
        x = keras.layers.Conv2D(f, kernel_size=(3, 3), padding='same',
                                activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(filters[0], kernel_size=(3, 3), padding='valid',
                            activation='relu')(x)

    x = keras.layers.UpSampling2D((2, 2))(x)

    output_decod = keras.layers.Conv2D(input_dims[2], kernel_size=(3, 3),
                                       activation='sigmoid', padding='same')(x)

    decoder = keras.Model(inputs=input_decod, outputs=output_decod)

    auto = keras.Model(input, decoder(encoder(input)))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

#!/usr/bin/env python3
"""Module that creates an autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates an autoencoder
    Args:
        input_dims is an integer containing the dimensions of the model input
        hidden_layers is a list containing the number of nodes for each hidden
            layer in the encoder, respectively
            the hidden layers should be reversed for the decoder
        latent_dims is an integer containing the dimensions of the latent
            space representation
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    """
    # This is our input image
    input_img = keras.Input(shape=(input_dims,))
    layer_enc = keras.layers.Dense(hidden_layers[0],
                                   activation='relu')(input_img)
    for hl in range(1, len(hidden_layers)):
        layer_enc = keras.layers.Dense(hidden_layers[hl],
                                       activation='relu')(layer_enc)
    latent_enc = keras.layers.Dense(latent_dims, activation='relu')(layer_enc)

    # "encoded" is the encoded representation of the input
    encoded = keras.layers.Input(shape=(latent_dims,))
    # "decoded" is the lossy reconstruction of the input
    latent = keras.layers.Dense(hidden_layers[hl],
                                activation='relu')(encoded)
    c = 1
    for hl in range(len(hidden_layers) - 2, -1, -1):
        decod = keras.layers.Dense(hidden_layers[hl],
                                   activation='relu')(latent if c else decod)
        c = 0
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(encoded)

    encoder = keras.Model(input_img, latent_enc)

    # Create the decoder model
    decoder = keras.Model(encoded, decoded)

    outputs = decoder(encoder(input_img))
    auto = keras.Model(input_img, outputs)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto

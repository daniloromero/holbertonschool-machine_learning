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
    x = input_img
    for hl in hidden_layers:
        x = keras.layers.Dense(hl, activation='relu')(x)
    latent_enc = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(input_img, latent_enc)

    # "decoder_input" is the encoded representation of img as input for decoder
    decoder_input = keras.layers.Input(shape=(latent_dims,))

    x = decoder_input
    for hl in reversed(hidden_layers):
        x = keras.layers.Dense(hl, activation='relu')(x)
    # "decoded" is the lossy reconstruction of the input
    decoded = keras.layers.Dense(input_dims,
                                 activation='sigmoid')(decoder_input)

    # Create the decoder model
    decoder = keras.Model(decoder_input, decoded)

    outputs = decoder(encoder(input_img))
    auto = keras.Model(input_img, outputs)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto

#!/usr/bin/env python3
"""Module that creates a convolutional encoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates convolutional autoencoder
    Args:
        input_dims:is a tuple of integers containing the dimension od the model
        hidden_layers: list containing  number of nodes for each hidden layer
        latent_dims: tuple of integerrs containing the dimension of the latent
            space representation

    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder eis the decoder model
        auto is the full autoencoder model
    """
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for l in hidden_layers:
        x = keras.layers.Dense(l, activation='relu')(x)
    mu = keras.layers.Dense(latent_dims)(x)
    log_var = keras.layers.Dense(latent_dims)(x)

    def sample(args):
        mu, log_var = args
        s1 = keras.backend.shape(mu)[0]
        epsilon = keras.backend.random_normal(
            shape=(s1, latent_dims),
            mean=mu,
            stddev=1.
        )
        return mu + keras.backend.exp(log_var / 2) * epsilon

    encoder_output = keras.layers.Lambda(sample)([mu, log_var])

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=[encoder_output, mu, log_var]
    )

    # deocder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for l in reversed(hidden_layers):
        x = keras.layers.Dense(l, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=decoder_input, outputs=decoder_output)
    auto = keras.Model(
        inputs=encoder_input,
        outputs=decoder(encoder(encoder_input)[2])
    )

    # loss
    def total_loss(x, x_decoded):
        reconstruction_loss = keras.losses.binary_crossentropy(x, x_decoded)
        reconstruction_loss *= input_dims
        kl_loss = (1 + log_var - keras.backend.square(mu) -
                   keras.backend.exp(log_var))
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return keras.backend.mean(reconstruction_loss + kl_loss)

    auto.compile(optimizer='adam', loss=total_loss)
    return encoder, decoder, auto

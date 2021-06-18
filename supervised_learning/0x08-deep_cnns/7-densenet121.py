#!/usr/bin/env python3
"""Module that builds the DenseNet-121 architecture"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """builds the DenseNet-121 architecture
    Args:
        growth_rate: is the growth rate
        compression: is the compression factor
    Return: keras model
    """
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)

    batch_norm = K.layers.BatchNormalization(axis=3)(X)

    activation = K.layers.Activation('relu')(batch_norm)

    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        padding='same',
        strides=2,
        kernel_initializer=init
    )(activation)

    maxpool1 = K.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(conv1)

    dense1, nb_filters = dense_block(maxpool1, 64, growth_rate, 6)
    transition1, nb_filters = transition_layer(dense1, nb_filters, compression)

    dense2, nb_filters = dense_block(transition1, nb_filters, growth_rate, 12)
    transition2, nb_filters = transition_layer(dense2, nb_filters, compression)

    dense3, nb_filters = dense_block(transition2, nb_filters, growth_rate, 24)
    transition3, nb_filters = transition_layer(dense3, nb_filters, compression)

    dense4, nb_filters = dense_block(transition3, nb_filters, growth_rate, 16)

    l_avg_pool = K.layers.AveragePooling2D(pool_size=[1, 1],
                                           strides=7,
                                           padding='same')(dense4)

    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=init)(l_avg_pool)

    model = K.models.Model(inputs=X, outputs=Y)
    return model

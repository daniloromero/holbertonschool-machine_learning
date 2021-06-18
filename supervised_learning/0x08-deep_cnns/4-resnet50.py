#!/usr/bin/env python3
"""Module that builds the ResNet-50 architecture"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """builds the ResNet-50 architecture
    Ars:
         input data will have shape (224, 224, 3)
    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)
    layer_1 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    batch_norm1 = K.layers.BatchNormalization()(layer_1)
    activation_1 = K.layers.Activation('relu')(batch_norm1)
    max_pool1 = K.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(activation_1)

    pro_block1 = projection_block(max_pool1, [64, 64, 256], 1)

    id_block1 = identity_block(pro_block1, [64, 64, 256])

    id_block2 = identity_block(id_block1, [64, 64, 256])

    pro_block2 = projection_block(id_block2, [128, 128, 512])

    id_block3 = identity_block(pro_block2, [128, 128, 512])

    id_block4 = identity_block(id_block3, [128, 128, 512])

    id_block5 = identity_block(id_block4, [128, 128, 512])

    pro_block3 = projection_block(id_block5, [256, 256, 1024])

    id_block6 = identity_block(pro_block3, [256, 256, 1024])

    id_block7 = identity_block(id_block6, [256, 256, 1024])

    id_block8 = identity_block(id_block7, [256, 256, 1024])

    id_block9 = identity_block(id_block8, [256, 256, 1024])

    id_block10 = identity_block(id_block9, [256, 256, 1024])

    pro_block4 = projection_block(id_block10, [512, 512, 2048])

    id_block11 = identity_block(pro_block4, [512, 512, 2048])

    id_block12 = identity_block(id_block11, [512, 512, 2048])

    l_avg_pool = K.layers.AveragePooling2D(pool_size=[7, 7],
                                           strides=7,
                                           padding='same')(id_block12)

    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=init)(l_avg_pool)

    model = K.models.Model(inputs=X, outputs=Y)
    return model

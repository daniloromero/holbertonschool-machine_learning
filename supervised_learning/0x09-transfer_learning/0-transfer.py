#!/usr/bin/env python3
"""Transfer learning"""

import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    This method pre-processes the data
    for the model
    """
    X_p = K.applications.inception_v3.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return (X_p, Y_p)


if __name__ == '__main__':

    """
    Transfer learning of the model VGG19
    and save it in a file cifar10.h5
    """
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    base_model = K.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(299, 299, 3)
    )
    base_model.summary()

    inputs = K.Input(shape=(32, 32, 3))

    input = K.layers.Lambda(
        lambda image: tf.image.resize(image,
                                      (299, 299)))(inputs)
    x = base_model(input, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(500, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs, outputs)
    base_model.trainable = False
    optimizer = K.optimizers.Adam()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['acc']
    )
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=200,
        epochs=5,
        verbose=1
    )

    model.save('cifar10.h5')

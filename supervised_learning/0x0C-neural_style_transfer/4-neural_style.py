#!/usr/bin/env python3
"""Neural Style Tranfer NST Class"""

import numpy as np
import tensorflow as tf


class NST:
    """performs tasks for neural style transfer
    Class atributes:
        - Content layer where will pull our feature maps
        - Style layer we are interested in
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = ['block5_conv2']

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """ Class constructor
        Arg:
            - style_image: img used as a style reference, numpy.ndarray
            - content_image: image used as a content reference, numpy.ndarray
            - alpha: the weight for content cost
            - beta: the weight for style cost
        Environment:
            Eager execution: TensorFlow’s imperative programming
                             environment, evaluates operations immediately
        """
        sty_error = 'style_image must be a numpy.ndarray with shape (h, w, 3)'
        c_error = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(style_image, np.ndarray):
            raise TypeError(sty_error)
        if len(style_image.shape) != 3:
            raise TypeError(sty_error)
        if style_image.shape[2] != 3:
            raise TypeError(sty_error)

        if not isinstance(content_image, np.ndarray):
            raise TypeError(c_error)
        if len(content_image.shape) != 3:
            raise TypeError(c_error)
        if content_image.shape[2] != 3:
            raise TypeError(c_error)

        if isinstance(alpha, str):
            raise TypeError("alpha must be a non-negative number")
        if alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if isinstance(beta, str):
            raise TypeError("beta must be a non-negative number")
        if beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """ rescales an image such that its pixels values are between 0
            and 1 and its largest side is 512 pixels
        Arg:
           - image: np.ndarray (h, w, 3) containing the image to be scaled
        Returns:
           - A scaled image Tensor
        """
        img_error = 'image must be a numpy.ndarray with shape (h, w, 3)'
        if not isinstance(image, np.ndarray):
            raise TypeError(img_error)
        if len(image.shape) != 3:
            raise TypeError(img_error)
        if image.shape[2] != 3:
            raise TypeError(img_error)

        h, w, _ = image.shape
        max_dim = 512
        maximum = max(h, w)
        scale = max_dim / maximum
        new_shape = (int(h * scale), int(w * scale))
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_bicubic(image, new_shape)
        image = tf.cast(image, tf.float32)
        image /= 255
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
        return image

    def load_model(self):
        """Loads model VGG19 Keras as base"""
        vgg_base = tf.keras.applications.VGG19(include_top=False,
                                               weights='imagenet')
        x = vgg_base.input
        outputs = []
        layer_names = self.style_layers + self.content_layer
        for layer in vgg_base.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )
                x = layer(x)
            else:
                x = layer(x)
                if layer.name in layer_names:
                    outputs.append(x)
                layer.trainable = False

        model = tf.keras.models.Model(vgg_base.input, outputs)
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """ calculate the gram matrices
        Arg:
            input_layer: instance of tf.Tensor or tf.Variable of shape
                    (1, h, w, c) with layer.output to calculate gram matrix
        Returns: tf.Tensor shape (1, c, c) with gram matrix of input_layer
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError('input_layer must be a tensor of rank 4')
        if len(input_layer.shape) != 4:
            raise TypeError('input_layer must be a tensor of rank 4')
        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

    def generate_features(self):
        """extracts the features used to calculate neural style cost"""
        prepro_style = tf.keras.applications.vgg19.preprocess_input(
                                            self.style_image * 255)
        prepro_content = tf.keras.applications.vgg19.preprocess_input(
                                            self.content_image * 255)
        style_features = self.model(prepro_style)[:-1]
        content_feature = self.model(prepro_content)[-1]
        self.gram_style_features = [self.gram_matrix(output)
                                    for output in style_features]
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer"""
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError('input_layer must be a tensor of rank 4')
        if len(style_output.shape) != 4:
            raise TypeError('input_layer must be a tensor of rank 4')
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError('input_layer must be a tensor of rank 4')
        _, h, w, c = style_output.shape
        if gram_target.shape.dims != [1, c, c]:
            raise TypeError('input_layer must be a tensor of rank 4')

        gram_style = self.gram_matrix(style_output)
        E = tf.reduce_mean(tf.square(gram_style - gram_target))
        return E

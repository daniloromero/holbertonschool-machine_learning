#!/usr/bin/env python3
"""Module that performs PCA color augmentation s in AlexNet"""
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """performs PCA color augmentation
    Args:
        image is a 3D tf.tensor containing the image to change
        alphas id a tuple of length 3 containing the amount for each channel
    Return the augmented image
    """
    print(image.shape)
    print(alphas)
    orig_img = image.numpy().astype(float).copy()

    img = image.numpy().astype(float) / 255.0  # rescale to 0 to 1 range

    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)
    # img_rs shape (640000, 3)

    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)

    # paper says 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

    #     eig_vals [0.00154689 0.00448816 0.18438678]

    #     eig_vecs [[ 0.35799106 -0.74045435 -0.56883192]
    #      [-0.81323938  0.05207541 -0.57959456]
    #      [ 0.45878547  0.67008619 -0.58352411]]

    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))

    # 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw
    # once per augmentation (not once per channel)

    # broad cast to speed things up
    m2[:, 0] = alphas * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):  # RGB
        orig_img[..., idx] += add_vect[idx]

    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # orig_img /= 255.0
    orig_img = np.clip(orig_img, 0.0, 255.0)

    # orig_img *= 255
    orig_img = orig_img.astype(np.uint8)

    return orig_img

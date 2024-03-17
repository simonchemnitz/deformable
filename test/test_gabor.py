# test of numpy and tensorflow related functions
# Import necessary modules and functions
import pytest
from typing import Callable
import numpy as np
import tensorflow as tf
from ..deformable.misc import gabor_filter, gabor
from skimage.filters import gabor_kernel as sk_gabor_kernel


def test_gabor_1():
    """
    Verify tensorflow and skimage gabor
    produce the same filter/kernel
    """
    theta = 45.0
    sigma = 3
    frequency = 0.25

    tf_result = gabor_filter(
        frequency=frequency, theta=theta, sigma_x=sigma, sigma_y=sigma
    )
    sk_result = sk_gabor_kernel(
        frequency=frequency, theta=theta, sigma_x=sigma, sigma_y=sigma
    )

    assert tf.debugging.assert_near(tf_result, sk_result)


# def test_tf_2_np():
#     """
#     Verify that tf->np->tf is the identity map
#     """
#     tf_image = tf.random.uniform(
#         shape=(1, 256, 256, 1), minval=0, maxval=1, dtype=tf.float32
#     )
#     tf.debugging.assert_near(np_img2_tf(tf_img2_np(tf_image)), tf_image)


# # VERIFY SHAPE
# def test_np_2_tf_shape():
#     """
#     Verify that np->tf produces correct shape
#     """
#     np_image = np.random.random(size=(256, 256))
#     assert np_img2_tf(np_image).shape == (1, 256, 256, 1)

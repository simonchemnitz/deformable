# test of numpy and tensorflow related functions
# Import necessary modules and functions
import pytest
from ..deformable.imstat import add
from typing import Callable
import numpy as np
import tensorflow as tf
from ..deformable.misc import np_img2_tf, tf_img2_np


# define general function to test numpy and tf output
def np_tf_equal(
    np_func: Callable[[np.ndarray], np.ndarray],
    tf_func: Callable[[tf.Tensor], tf.Tensor],
    input_shape: tuple[int, int],
):
    """
    Test if a tensorflow and numpy function
    produces the same output
    """
    # Generate input
    input_array = np.random.rand(*input_shape)

    # Convert input array to tensor
    input_tensor = tf.convert_to_tensor(input_array)

    # call both functions
    np_output = np_func(input_array)
    tf_output = tf_func(input_tensor).numpy()

    # assert out is equal
    np.testing.assert_allclose(np_output, tf_output)


# VERIFY IDENTIDY MAPS
def test_np_2_tf_id():
    """
    Verify that np->tf->np is the identity map
    """
    np_image = np.random.random(size=(256, 256))
    np.testing.assert_allclose(tf_img2_np(np_img2_tf(np_image)), np_image)


def test_tf_2_np():
    """
    Verify that tf->np->tf is the identity map
    """
    tf_image = tf.random.uniform(
        shape=(1, 256, 256, 1), minval=0, maxval=1, dtype=tf.float32
    )
    tf.debugging.assert_near(np_img2_tf(tf_img2_np(tf_image)), tf_image)


# VERIFY SHAPE
def test_np_2_tf_shape():
    """
    Verify that np->tf produces correct shape
    """
    np_image = np.random.random(size=(256, 256))
    assert np_img2_tf(np_image).shape == (1, 256, 256, 1)


def test_tf_2_np_shape():
    """
    Verify that tf->np produces correct shape
    """
    tf_image = tf.random.uniform(
        shape=(1, 256, 256, 1), minval=0, maxval=1, dtype=tf.float32
    )
    assert tf_img2_np(tf_image).shape == (256, 256)

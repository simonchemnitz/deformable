# Import necessary modules and functions
import pytest
from ..deformable.imstat import add
from typing import Callable
import numpy as np
import tensorflow as tf


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

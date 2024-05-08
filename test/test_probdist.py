# test of numpy and tensorflow related functions
# Import necessary modules and functions
import pytest
from typing import Callable
import numpy as np
import tensorflow as tf
from ..deformable.misc import np_img2_tf, tf_img2_np
from ..deformable.probmetrics import region_pdf
from ..deformable.imstat import region_intensity_pdf

# Test probdistribution calculations


def test_tf_broadcast():
    """
    Verify broadcast is done correctly
    """
    img = np.random.random(size=(1, 200, 300, 1))

    newshape = (256,) + img.shape[1:]
    broadcasted_img = tf.broadcast_to(img, shape=newshape)
    intensities = np.expand_dims(tf.range(256), axis=(1, 2, 3))

    diff_img = broadcasted_img - intensities

    for i in range(256):
        int_img = diff_img[i, :, :, 0]
        original_diff_img = img[0, :, :, 0] - i
        assert np.allclose(int_img, original_diff_img), "notsame"

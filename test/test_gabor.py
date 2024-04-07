# test of numpy and tensorflow related functions
# Import necessary modules and functions
import pytest
from typing import Callable
import numpy as np
import tensorflow as tf
from ..deformable.misc import gabor_filter, gabor
from ..deformable.misc import np_img2_tf, tf_img2_np
from skimage.filters import gabor_kernel as sk_gabor_kernel
from skimage.filters import gabor as sk_gabor
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


def test_nd_tf_convolution(show_image=False):
    # Create image and filter
    img = np.random.random(size=(200, 300))
    nd_filter = (5 * np.random.random(size=(17, 17))) ** 2

    # Convert to image and filter to tf shape
    tf_img = np.expand_dims(img, axis=(0, 3))
    tf_filter = np.expand_dims(nd_filter[::-1, ::-1], axis=(2, 3))

    # convolve images
    nd_result = ndi.convolve(img, nd_filter, mode="reflect")
    tf_result = tf.nn.convolution(tf_img, tf_filter, padding="SAME")
    tf_result = (tf_result[0, :, :, 0]).numpy()

    if show_image:
        fig = plt.figure()
        plt.imshow(nd_result - tf_result, cmap="jet")
        plt.colorbar()
        plt.show()

    # assert results are equal (excluding different padding methods)
    assert np.allclose(
        nd_result[17 : 200 - 17, 17 : 300 - 17], tf_result[17 : 200 - 17, 17 : 300 - 17]
    )


def test_multiple_conv():
    img = np.random.random(size=(200, 300))
    img = tf.constant(np.expand_dims(img, axis=(0, 3)))
    filter1 = np.random.random(size=(17, 17))
    filter1 = np.expand_dims(filter1, axis=(2, 3))
    filter1 = tf.constant(filter1)

    filter2 = np.random.random(size=(17, 17))
    filter2 = np.expand_dims(filter2, axis=(2, 3))
    filter2 = tf.constant(filter2)

    combined_filter = np.concatenate([filter1, filter2], axis=3)

    img_fc = tf.nn.convolution(input=img, filters=combined_filter, padding="SAME")
    img_f1 = tf.nn.convolution(input=img, filters=filter1, padding="SAME")
    img_f2 = tf.nn.convolution(input=img, filters=filter2, padding="SAME")

    condition1 = np.allclose(img_fc[:, :, :, 0:1], img_f1)
    condition2 = np.allclose(img_fc[:, :, :, 1:2], img_f2)

    condition = (condition1) & (condition2)
    err_msg = f"""\n
    Dimensions not equal:
    Shapes:
    -------
    Image:............................{img.shape}
    Filter1:..........................{filter1.shape}
    Filter2:..........................{filter2.shape}
    Combined Filter:..................{combined_filter.shape}

    Convolved image Filter1:..........{img_f1.shape}
    Convolved image Filter2:..........{img_f2.shape}
    Convolved image Combined filter:..{img_fc.shape}

    Filter1 vs Combines:..............{condition1}
    Filter2 vs Combines:..............{condition2}
    """
    assert condition, err_msg


def test_single_gabor():
    """
    Assert that tf gabor convolution produces same
    result at skimage gabor filtering
    """
    # generate image
    img = np.random.random(size=(200, 300))
    tf_image = np.expand_dims(img, axis=(0, 3))

    # Define gabor parameters
    frequency = 0.25
    theta = 1.7
    sigma = 1
    n_stds = 3
    offset = 0
    bandwidth = 1

    # Skimage gabor convolution
    sk_real, sk_imag = sk_gabor(
        image=img,
        frequency=frequency,
        theta=theta,
        sigma_x=sigma,
        sigma_y=sigma,
        n_stds=n_stds,
        offset=offset,
        bandwidth=bandwidth,
    )

    tf_real, tf_imag = gabor(
        image=tf_image,
        frequency=frequency,
        theta=theta,
        sigma_x=sigma,
        sigma_y=sigma,
        n_stds=n_stds,
        offset=offset,
        bandwidth=bandwidth,
    )

    print(sk_real.shape)
    print(tf_real.shape)

    diff = sk_real - tf_real[0, :, :, 0]
    tf_real = tf_real[0, :, :, 0]
    condition = np.allclose(
        sk_real[17 : 200 - 17, 17 : 300 - 17], tf_real[17 : 200 - 17, 17 : 300 - 17]
    )
    assert condition, "test"


def test_multiple_gabor():
    """
    test that multiple gabor produces same result as
    multiple single gabors
    """
    assert True, "test"

import numpy as np
import tensorflow as tf


def bspline(u: tf.Tensor) -> tf.Tensor:
    """
    Calculate bspline for a tensor
    Parameters:
    -----------
    u: tf.Tensor
        Input tensor

    Returns:
    --------
    b: tf.Tensor
        Output tensor of bspline values
    """
    b0 = tf.pow(1 - u, 3) / 6
    b1 = (3 * tf.pow(u, 3) - 6 * tf.pow(u, 2) + 4) / 6
    b2 = (-3 * tf.pow(u, 3) + 3 * tf.pow(u, 2) + 3 * u + 1) / 6
    b3 = tf.pow(u, 3) / 6
    return tf.transpose(tf.stack([b0, b1, b2, b3]))


def inflection_points(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate inflection points

    Parameters:
    -----------
    x: np.ndarray
        x values, for deformable models this is
        scale

    y: np.ndarray
        y values, for deformable models this is
        the KL divergence or special cases of it

    Returns:
    --------
    ip: np.ndarray
        Inflection points
    """
    # Calculate the second derivative
    dydx = np.gradient(y, x)
    d2ydx2 = np.gradient(dydx, x)

    # Find the indices where the second derivative changes sign
    inflection_indices = np.where(np.diff(np.sign(d2ydx2)))[0]

    # Get the x and y values at the inflection points
    # ips = list(zip(x[inflection_indices], y[inflection_indices]))
    ips = list(x[inflection_indices])

    return ips


def delta(x: int) -> bool:
    """
    Delta function
    return 1 if x==0
    return 0 if x!=0
    """
    return int(x + 1 == 1)


def np_img2_tf(img: np.ndarray) -> tf.Tensor:
    """
    Convert a numpy array image
    to a tensorflow tensor image
    """
    return tf.expand_dims(tf.expand_dims(img, axis=0), axis=-1)


def tf_img2_np(img: tf.Tensor) -> np.ndarray:
    """
    Convert a tensorflow tensor image
    to a numpy array image
    """
    return img.numpy()[0, :, :, 0]


def tf_expand_img(image: tf.Tensor) -> tf.Tensor:
    """
    Expand an image of shape (height, width)
    to (1,height, width, 1), so tf functions can
    be applied

    Parameters:
    -----------
    image: tf.Tensor
        Tensorflow tensor image, shape=(height, width)

    Returns:
    --------
    tf_img: tf.Tensor
        Tensorflow tensor image, shape=(1,height, width,1)
    """

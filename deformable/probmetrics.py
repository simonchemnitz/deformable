import numpy as np
import tensorflow as tf


def chernoff_information(p1: np.ndarray, p2: np.ndarray, n_points=500):
    """
    Calculate the chernoff information given by:

    max_{0<=t<=1} -log( µ(t) )
    with:
    µ(t) = $integral p1(i)^(1-t) * p_2(i)^t di


    This is a measure of distance between two probability densities

    Parameters:
    -----------
    p1: np.ndarray
        Image pdf

    p2: np.ndarray
        Image pdf

    n_points: int
        Number of points between 0 and 1 to evaluate the µ
        function.

    Returns:
    --------
    chernoff: float
        The Chernoff information
    """
    # t values
    ts = tf.linspace(start=0.00001, stop=1, num=n_points)

    # Calculate µ(t) for each t
    mu_result = tf.map_fn(lambda t: mu_t(p1, p2, t), ts)

    # apply -log()
    result = tf.map_fn(lambda mu: -tf.math.log(mu), mu_result)

    return tf.math.reduce_min(result)


def mu_t(p1: tf.Tensor, p2: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    """
    Calculate mu  for a tensor
    """
    pow1 = tf.pow(p1, t)
    pow2 = tf.pow(p2, 1 - t)
    integral = tf.reduce_sum(tf.multiply(pow1, pow2))

    return integral


def rho(p1: tf.Tensor, p2: tf.Tensor) -> tf.Tensor:
    """
    Calculate rho
    """
    return tf.reduce_sum(tf.math.sqrt(tf.multiply(p1, p2)))


def bhattachayya_distance(p1: tf.Tensor, p2: tf.Tensor) -> tf.Tensor:
    """
    Calculate the bhattachayya distance between two
    distributions, p1 & p2, B(p1|p2)
    """
    return -tf.math.log(rho(p1, p2))


def region_pdf(image: tf.Tensor, region: tf.Tensor, sigma: float) -> tf.Tensor:
    """
    Calculate the intensity pdf for a given region
    """
    # Normalisation constant
    volume = 1 / tf.reduce_sum(region)
    c = tf.math.sqrt(1 / (2 * np.pi * (sigma**2)))
    norm_constant = volume * c
    # Apply region
    masked_image = image * region

    # Broadcast image
    new_shape = (256,) + image.shape[1:]
    tiled_image = tf.broadcast_to(masked_image, new_shape)

    # Intensities to subtract from the image
    intensities = np.expand_dims(tf.range(256), axis=(1, 2, 3))

    # Calculate intensity difference
    diff_map = tiled_image - intensities

    # calculate numerator
    numerator = tf.pow(diff_map, 2)
    numerator = -numerator / (2 * sigma**2)

    # Exponential map
    exp_map = tf.exp(numerator)

    # Calculate integral
    integral = tf.reduce_sum(exp_map, axis=(1, 2, 3))

    # normalise
    probdist = integral * norm_constant

    return probdist

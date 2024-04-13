import numpy as np
import tensorflow as tf


def mu_t_np(p1: np.ndarray, p2: np.ndarray, t: float) -> float:
    """
    Calculate the mu value in chernoff information
    µ(t) = \integral p1(i)^(1-t) * p_2(i)^t di

    Parameters:
    -----------
    p1: np.ndarray
        Image pdf

    p2: np.ndarray
        Image pdf
    t: float
    """
    # Verify input
    assert_msg = f"""
    Length of p1 or p2 is not equal to 256:
    len p1: {len(p1)}
    len p2: {len(p2)} 
    """
    assert len(p1) == 256 and len(p2) == 256, assert_msg
    # Calculate powers
    p1_values = p1 ** (1 - t)
    p2_values = p2 ** (t)

    # Calculate product
    prod = p1_values * p2_values

    # Calculate integral (sum)
    integral = np.sum(prod)

    return integral


def rho_np(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Special case of µ with t=0.5
    Parameters:
    -----------
    p1: np.ndarray
        Image pdf

    p2: np.ndarray
        Image pdf

    Returns:
    --------
    µ(0.5)
    """
    return mu_t_np(p1, p2, 0.5)


def bhattachayya_distance_np(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Special case of Chernoff information with t=0.5

    Parameters:
    -----------
    p1: np.ndarray
        Image pdf

    p2: np.ndarray
        Image pdf

    Returns:
    --------
    B: float
        The bhattachayya distance
        between p1 and p2
    """
    return -np.log(rho_np(p1, p2))


def chernoff_information_np(p1: np.ndarray, p2: np.ndarray, n_points=500) -> float:
    """
    Calculate the chernoff information given by:

    max_{0<=t<=1} -log( µ(t) )
    with:
    µ(t) = \integral p1(i)^(1-t) * p_2(i)^t di


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
    # List og t values to evaluate µ(t)
    ts = np.linspace(start=0, stop=1, num=n_points)

    # µ(t) for each t in ts
    vals = np.array(list(map(lambda t: mu_t_np(p1, p2, t), ts)))
    log_vals = -np.log(vals)

    # Calculate chernoff information (max)
    chernoff = np.max(log_vals)

    return chernoff


def mu_t(p1: tf.Tensor, p2: tf.Tensor, t: float) -> tf.Tensor:
    """
    Calculate mu  for a tensor
    """
    return tf.reduce_sum(tf.pow(tf.multiply(p1, p2), t))


def rho(p1: tf.Tensor, p2: tf.Tensor) -> tf.Tensor:
    """
    Calculate rho
    """
    return mu_t(p1=p1, p2=p2, t=0.5)


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
    c = 1 / (2 * np.pi * sigma * tf.reduce_sum(region))

    # Apply region
    masked_image = image * region

    # Tile image
    new_shape = (256,) + image.shape[1:]
    tiled_image = tf.broadcast_to(masked_image, new_shape)

    intensities = np.expand_dims(tf.range(256), axis=(1, 2, 3))

    # Calculate intensity difference
    diff_map = tiled_image - intensities

    # calculate numerator
    numerator = tf.pow(diff_map, 2)
    numerator = -numerator / (2 * sigma**2)

    # Exponential map
    exp_map = tf.exp(numerator)

    # Calculate integral
    integral = tf.reduce_sum(exp_map, axis=0)

    # normalise
    probdist = integral * c

    return probdist

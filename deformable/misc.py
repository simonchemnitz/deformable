import numpy as np
import tensorflow as tf
import math


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
    ###TODO


def _sigma_prefactor(bandwidth):
    b = bandwidth
    # See http://www.cs.rug.nl/~imaging/simplecell.html
    return 1.0 / math.pi * math.sqrt(math.log(2) / 2.0) * (2.0**b + 1) / (2.0**b - 1)


def gabor_filter(
    frequency: float,
    theta=0,
    bandwidth=1,
    sigma_x=None,
    sigma_y=None,
    n_stds=3,
    offset=0,
) -> tf.Tensor:
    """Return real and imaginary responses to Gabor filter.
    See numpy.gabor_kernel for more information.

    Parameters
    ----------
    image : 2-D array
        Input image.
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.
    theta : float, optional
        Orientation in radians. If 0, the harmonic is in the x-direction.
    bandwidth : float, optional
        The bandwidth captured by the filter. For fixed bandwidth, ``sigma_x``
        and ``sigma_y`` will decrease with increasing frequency. This value is
        ignored if ``sigma_x`` and ``sigma_y`` are set by the user.
    sigma_x, sigma_y : float, optional
        Standard deviation in x- and y-directions. These directions apply to
        the kernel *before* rotation. If `theta = pi/2`, then the kernel is
        rotated 90 degrees so that ``sigma_x`` controls the *vertical*
        direction.
    n_stds : scalar, optional
        The linear size of the kernel is n_stds (3 by default) standard
        deviations.
    offset : float, optional
        Phase offset of harmonic function in radians.
    mode : {'constant', 'nearest', 'reflect', 'mirror', 'wrap'}, optional
        Mode used to convolve image with a kernel, passed to `ndi.convolve`
    cval : scalar, optional
        Value to fill past edges of input if ``mode`` of convolution is
        'constant'. The parameter is passed to `ndi.convolve`.

    Returns
    -------
    real, imag : arrays
        Filtered images using the real and imaginary parts of the Gabor filter
        kernel. Images are of the same dimensions as the input one.
    """
    if sigma_x is None:
        sigma_x = _sigma_prefactor(bandwidth) / frequency
    if sigma_y is None:
        sigma_y = _sigma_prefactor(bandwidth) / frequency

    ct = tf.cos(theta)
    st = tf.sin(theta)
    x0 = tf.cast(
        tf.math.ceil(
            tf.math.maximum(
                tf.math.abs(n_stds * sigma_x * ct),
                tf.math.abs(n_stds * sigma_y * st),
                1,
            )
        ),
        dtype=tf.int32,
    )
    y0 = tf.cast(
        tf.math.ceil(
            tf.math.maximum(
                tf.math.abs(n_stds * sigma_y * ct),
                tf.math.abs(n_stds * sigma_x * st),
                1,
            )
        ),
        dtype=tf.int32,
    )

    y_range = tf.range(-y0, y0 + 1, dtype=tf.float64)
    x_range = tf.range(-x0, x0 + 1, dtype=tf.float64)
    y, x = tf.meshgrid(y_range, x_range, indexing="ij", sparse=True)
    rotx = x * ct + y * st
    roty = -x * st + y * ct

    g = tf.exp(
        -0.5 * (rotx**2 / sigma_x**2 + roty**2 / sigma_y**2)
        + 1j * (2 * tf.constant(np.pi, dtype=tf.float64) * frequency * rotx + offset)
    )
    g *= 1 / (2 * tf.constant(np.pi, dtype=tf.float64) * sigma_x * sigma_y)

    return g


def gabor(
    image: tf.Tensor,
    frequency: float,
    theta=0,
    bandwidth=1,
    sigma_x=None,
    sigma_y=None,
    n_stds=3,
    offset=0,
    mode="reflect",
    cval=0,
) -> tf.Tensor:
    """
    Convolve an image with a gabor filter
    For more information see numpy.gabor

    Parameters
    ----------
    image : 2-D array
        Input image.
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.
    theta : float, optional
        Orientation in radians. If 0, the harmonic is in the x-direction.
    bandwidth : float, optional
        The bandwidth captured by the filter. For fixed bandwidth, ``sigma_x``
        and ``sigma_y`` will decrease with increasing frequency. This value is
        ignored if ``sigma_x`` and ``sigma_y`` are set by the user.
    sigma_x, sigma_y : float, optional
        Standard deviation in x- and y-directions. These directions apply to
        the kernel *before* rotation. If `theta = pi/2`, then the kernel is
        rotated 90 degrees so that ``sigma_x`` controls the *vertical*
        direction.
    n_stds : scalar, optional
        The linear size of the kernel is n_stds (3 by default) standard
        deviations.
    offset : float, optional
        Phase offset of harmonic function in radians.
    mode : {'constant', 'nearest', 'reflect', 'mirror', 'wrap'}, optional
        Mode used to convolve image with a kernel, passed to `ndi.convolve`
    cval : scalar, optional
        Value to fill past edges of input if ``mode`` of convolution is
        'constant'. The parameter is passed to `ndi.convolve`.

    Returns
    -------
    real, imag : arrays
        Filtered images using the real and imaginary parts of the Gabor filter
        kernel. Images are of the same dimensions as the input one.
    """
    g = gabor_filter(frequency, theta, bandwidth, sigma_x, sigma_y, n_stds, offset)

    g_real = tf.math.real(g)
    g_imag = tf.math.imag(g)

    filtered_real = tf.nn.convolution(
        tf.expand_dims(image, axis=0),
        tf.expand_dims(tf.expand_dims(g_real, axis=-1), axis=-1),
        padding=mode.upper(),
    )[0]

    filtered_imag = tf.nn.convolution(
        tf.expand_dims(image, axis=0),
        tf.expand_dims(tf.expand_dims(g_imag, axis=-1), axis=-1),
        padding=mode.upper(),
    )[0]

    return filtered_real, filtered_imag


def gauss(x: tf.Tensor, sigma: float) -> tf.Tensor:
    """
    Calculate gaus exponential
    Exp(-x²/(2*sigma²))
    """
    # calculate the fraction -x²/(2*sigma²)
    fraction = tf.divide(-(x**2), 2 * sigma**2)
    return tf.exp(fraction)

import numpy as np
import tensorflow as tf
import skimage.filters as sf
import math

print("+---------------- +")
print("| Packages Loaded |")
print("+---------------- +")
print()
print()
print()


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
    See skimage.filters._gabor.gabor_kernel for more information.

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
    # Extract skimage filter
    g = sf.gabor_kernel(
        frequency=frequency,
        theta=theta,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        n_stds=n_stds,
        offset=offset,
        bandwidth=bandwidth,
    )
    # Convert to tensorflow
    g_real = np.real(np.expand_dims(g, axis=(2, 3)))  # [::-1, ::-1],
    g_imag = np.imag(np.expand_dims(g, axis=(2, 3)))  # [::-1, ::-1],

    return (g_real, g_imag)


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
    For more information see skimage.filters._gabor.gabor_kernel

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
    g_real, g_imag = gabor_filter(
        frequency, theta, bandwidth, sigma_x, sigma_y, n_stds, offset
    )

    filtered_real = tf.nn.convolution(image, g_real, padding="SAME")

    filtered_imag = tf.nn.convolution(image, g_imag, padding="SAME")

    return (filtered_real, filtered_imag)


def multiple_gabor(image: tf.Tensor, gabor_list: list[tuple[tf.Tensor]]) -> tf.Tensor:
    """TODO
    Convolve an image with multiple gabor filters
    """
    # Extract the real and imaginary gabor filters

    g_real = np.concatenate([g[0] for g in gabor_list], axis=3)
    g_imag = np.concatenate([g[1] for g in gabor_list], axis=3)
    filtered_real = tf.nn.convolution(input=image, filters=g_real, padding="SAME")
    filtered_imag = tf.nn.convolution(input=image, filters=g_imag, padding="SAME")
    return (filtered_real, filtered_imag)


def gauss(x: tf.Tensor, sigma: float) -> tf.Tensor:
    """
    Calculate gaus exponential
    Exp(-x²/(2*sigma²))
    """
    # calculate the fraction -x²/(2*sigma²)
    fraction = tf.divide(-(x**2), 2 * sigma**2)
    return tf.exp(fraction)


if __name__ == "__main__":
    theta = 45.0
    sigma = 3
    frequency = 0.25
    image = np.random.rand(200, 300)
    image = np_img2_tf(image)

    filted = gabor(
        image, frequency=frequency, theta=theta, sigma_x=sigma, sigma_y=sigma
    )

import numpy as np
from scipy.spatial.distance import cdist
from skimage import io, util
from skimage.filters import gabor
import skimage.morphology as morph
import tensorflow as tf


def E_d_map(
    image: np.ndarray,
    boundary_indices: np.ndarray,
    normalise=False,
) -> np.ndarray:
    """
    Calculate the depth map to the region

    Parameters:
    -----------
    im: np.ndarray
        Array of shape (N,K)

    boundary_indices: np.ndarray
        array containing indices of the boundary region

    Returns:
    --------
    distance_map: np.ndarray
        Array of shape(N,K)
    """
    # Indices of the image
    img_indices = np.indices(image.shape).reshape(2, -1).T

    # Calculate the euclidean distance for each index to
    # each boundary point
    boundary_distances = cdist(img_indices, boundary_indices, metric="euclidean")

    # Minimum distance
    min_distances = np.min(boundary_distances, axis=1)

    # Distance map
    distance_map = min_distances.T.reshape(image.shape)

    # Normalise:
    if normalise:
        distance_map = distance_map / np.max(distance_map)

    return distance_map


def distance_map(model: np.ndarray) -> np.ndarray:
    """
    Calculate distance map for a given model

    Parameters:
    -----------
    model: np.ndarray
        Binary image map of the model

    Returns:
    --------
    d_map: np.ndarray
        Distance mapå image, with each value the distance to
        the model boundary
    """
    model_boundary = boundary(model)
    model_boundary_indices = boundary_indices(model_boundary=model_boundary)
    Ed = E_d_map(image_shape=image.shape, boundary_indices=model_boundary_indices)

    d_map = (2 * model - 1) * Ed
    return d_map


def boundary(model: np.ndarray) -> np.ndarray:
    """
    Return the boundary of a model
    """
    return morph.binary_dilation(model, footprint=np.ones(shape=(3, 3))) - model


def boundary_indices(model_boundary: np.ndarray) -> list[int]:
    """
    Given a model boundary map, return the indices
    """
    idx = np.where(model_boundary == 1)
    idx = np.stack(idx).reshape(2, -1).T

    return idx


def volume(model: np.ndarray) -> int:
    """
    Calculate volume of a model
    """
    return np.sum(model, dtype=int)


def np_bspline(u: np.ndarray) -> np.ndarray[float]:
    """
    Caclulate b spline values
    """
    b0 = (1 - u) ** 3 / 6
    b1 = (3 * u**3 - 6 * u**2 + 4) / 6
    b2 = (-3 * u**3 + 3 * u**2 + 3 * u + 1) / 6
    b3 = (u**3) / 6

    return np.vstack((b0, b1, b2, b3)).T


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


def region_point_pdf(region: np.ndarray, intensity_i: int, sigma: float) -> float:
    """
    Calculate phi point probability for a given intensity i:
    P(i|Phi_M)

    Parameters:
    -----------
    region:n p.ndarray
        Region to calculate P from

    intensity_i: int
        Intensity i

    sigma: float
        Width of gaussian kernel

    Returns:
    --------
    point_prob: float
    """
    # Calulate multiplicative normalisation constant
    area = np.prod(np.shape(region))
    constant = np.sqrt(1 / (2 * np.pi * sigma**2))

    # Calculate exponential
    numerator = -((intensity_i - region) ** 2)
    denominator = 2 * (sigma**2)
    exp = np.exp(numerator / denominator)

    # Calculate intergral (sum)
    point_prob = constant * np.sum(exp) / area

    return point_prob


def region_intensity_pdf(region: np.ndarray, sigma: float) -> np.ndarray[float]:
    """
    Calcuate the phi prob for a region

    Parameters:
    -----------
    region: np.ndarray
    sigma: float


    Returns:
    --------
    pdf: list[float]

    """
    pdf = [
        region_point_pdf(region=region, sigma=sigma, intensity_i=i) for i in range(255)
    ]
    return pdf


def initial_model(
    image_shape: tuple[int], centerpoint: list[int], width: int
) -> np.ndarray:
    """
    Create a square region as initial model

    Parameters:
    -----------


    Returns:
    --------
    model: np.ndarray
        Binary model
    """
    assert width % 2 == 1, "Width needs to be odd"

    x, y = centerpoint

    model = np.zeros(shape=image_shape)

    model[y - width : y + width + 1, x - width : x + width + 1] = 1

    return model


# TODO tensorflow gabor filter


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Image
    image = np.random.random(size=(1024, 1024))

    # Model
    model = initial_model(image_shape=image.shape, centerpoint=[700, 250], width=101)
    # Model boundary
    model_boundary = boundary(model)

    # distance map
    d_map = distance_map(image=image, model=model)

    # Plot results
    fig, axis = plt.subplots(nrows=2, ncols=2)
    im0 = axis[0, 0].imshow(image, cmap="gray")
    im1 = axis[0, 1].imshow(model, cmap="jet")
    im2 = axis[1, 0].imshow(model_boundary, cmap="jet")
    im3 = axis[1, 1].imshow(d_map, cmap="jet")

    # Create colorbars for each subplot
    fig.colorbar(im0, ax=axis[0, 0])
    fig.colorbar(im1, ax=axis[0, 1])
    fig.colorbar(im2, ax=axis[1, 0])
    fig.colorbar(im3, ax=axis[1, 1])

    # Titles
    axis[0, 0].set_title("Original image")
    axis[0, 1].set_title("Model")
    axis[1, 0].set_title("Model boundary")
    axis[1, 1].set_title("Distance map")

    # Remove axis
    for ax in axis.ravel():
        ax.axis("off")

    plt.show()

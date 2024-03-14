import numpy as np
import tensorflow as tf
from math import pi
from misc import gauss


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


def dilate(model: tf.Tensor, n_dilations=1) -> tf.Tensor:
    """
    Dilate a tensorflow image model
    """
    dilated_model = tf.nn.dilation2d(
        input=model,
        filters=tf.ones((3, 3, 1), dtype=tf.float32),
        strides=[1, 1, 1, 1],
        padding="SAME",
        dilations=[1, n_dilations, n_dilations, 1],
        data_format="NHWC",
    )

    return dilated_model


def boundary(model: tf.Tensor, n_dilations=1) -> tf.Tensor:
    """
    Extract the boundary of a model
    """
    # Perform binary dilation using TensorFlow's dilation2d function
    dilated_model = dilate(model=model, n_dilations=n_dilations)

    # Subtract original image from dilated image
    result = dilated_model - model
    return result


def volume(model: tf.Tensor) -> tf.Tensor:
    """
    Calculate volume of a model
    """
    return tf.reduce_sum(model)


def prob_density(model: tf.Tensor, image: tf.Tensor, sigma: float) -> tf.Tensor:
    """
    Calculate the probability density for
    a given model,

    Point probability for intensity, i, is given by:
    P(i|model) = exp[ -( (i-model)²  ) / (2*sigma²) ]

    Parameters:
    -----------
    model: tf.Tensor,
        Intial binary model

    image: tf.Tensor,
        Image of interest

    sigma: float
        sigma

    Returns:
    --------
    pdf: tf.Tensor
        Intensity pdf, shape(256)
    """
    # Normalisation constants
    area = tf.reduce_prod(tf.shape(model))
    int_constant = tf.pow(tf.constant(2 * pi * tf.pow(sigma, 2)), -0.5)
    norm_constant = tf.math.divide(area, int_constant)

    # Calculate images
    new_shape = (256, model.shape[1], model.shape[2], 1)
    broadcasted_model = tf.broadcast_to(model, shape=new_shape)

    # Tensor of intensities
    # Generate intensity tensor
    intensities = tf.range(256, dtype=float)

    # Expand dimensions to match image_tensor shape
    intensities = tf.expand_dims(intensities, axis=(1, 2, 3))

    # substract intensity i from image
    tilde_image = tf.subtract(intensities, broadcasted_model)

    # expnential images
    exp_img = gauss(tilde_image, sigma=sigma)
    point_prob = norm_constant * exp_img

    pdf = tf.reduce_sum(point_prob, axis=0)

    return pdf


if __name__ == "__main__":
    print()
    print("Main")
    import matplotlib.pyplot as plt
    import numpy as np

    img = np.random.random(size=(1, 200, 200, 1))
    model = np.zeros(shape=img.shape, dtype=int)
    model[0, 55 - 5 : 55 + 6, 75 - 5 : 75 + 6, 0] = 1

    # Convert to tensorflow:
    tf_img = tf.convert_to_tensor(img, dtype=tf.float32)
    tf_model = tf.convert_to_tensor(model, dtype=tf.float32)
    model_boundary = boundary(tf_model).numpy()[0, :, :, 0]
    print(model_boundary.shape)
    fig, axs = plt.subplots(2, 2)

    # Plot each image in a subplot
    axs[0, 0].imshow(img[0, :, :, 0], cmap="gray")
    axs[0, 0].set_title("Image 1")

    axs[0, 1].imshow(model[0, :, :, 0], cmap="gray")
    axs[0, 1].set_title("Image 2")

    axs[1, 0].imshow(model_boundary, cmap="gray")
    axs[1, 0].set_title("Image 3")

    # axs[1, 1].imshow(image4, cmap="gray")
    # axs[1, 1].set_title("Image 4")

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()

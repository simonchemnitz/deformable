import numpy as np
import tensorflow as tf


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


def boundary(model: tf.Tensor, n_dilations=1) -> tf.Tensor:
    """
    Extract the boundary of a model
    """
    # Perform binary dilation using TensorFlow's dilation2d function
    dilated_img = tf.nn.dilation2d(
        input=model,
        filters=tf.ones((3, 3, 1), dtype=tf.float32),
        strides=[1, 1, 1, 1],
        padding="SAME",
        dilations=[1, n_dilations, n_dilations, 1],
        data_format="NHWC",
    )

    # Subtract original image from dilated image
    result = dilated_img - model
    return result


def volume(model: tf.Tensor) -> tf.Tensor:
    """
    Calculate volume of a model
    """
    return tf.reduce_sum(model)


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

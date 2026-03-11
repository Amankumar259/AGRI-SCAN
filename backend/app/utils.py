import numpy as np
import tensorflow as tf
import io

def load_image(file_bytes: bytes) -> np.ndarray:
    """
    EXACT SAME preprocessing as training (CRITICAL)
    """

    # Load image using TensorFlow (same as training)
    image = tf.keras.utils.load_img(
        io.BytesIO(file_bytes),
        target_size=(160, 160)
    )

    # Convert to array (NO normalization)
    img_array = tf.keras.utils.img_to_array(image)

    return img_array

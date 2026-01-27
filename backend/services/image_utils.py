import io
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = (224, 224)


def preprocess_image(image_bytes: bytes):
    """
    Converts raw image bytes into a preprocessed tensor
    suitable for EfficientNet.
    """

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    
    image = image.resize(IMG_SIZE)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return tf.convert_to_tensor(img_array, dtype=tf.float32)

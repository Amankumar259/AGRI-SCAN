import numpy as np
from PIL import Image
import io

CLASS_NAMES = [
  "Tomato Early Blight",
  "Tomato Late Blight",
  "Potato Early Blight",
  "Potato Late Blight",
  "Healthy"
]

def load_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    image = image.resize((224,224))
    img_array = np.array(image)/255.0
    img_array = img_array.reshape(1,224,224,3)
    return img_array


def format_result(pred):
    idx = int(np.argmax(pred))
    return CLASS_NAMES[idx]

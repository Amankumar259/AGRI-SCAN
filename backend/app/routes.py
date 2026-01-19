from fastapi import APIRouter, UploadFile
from PIL import Image
import numpy as np
from .utils import load_image, format_result
from .treatment_data import TREATMENTS
import tensorflow as tf

router = APIRouter()

MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = tf.keras.models.load_model("ml/plant_disease_model.h5")
    return MODEL


@router.post("/predict")
async def predict(file: UploadFile):

    bytes_img = await file.read()

    img_array = load_image(bytes_img)

    model = get_model()

    pred = model.predict(img_array)

    disease_name = format_result(pred)

    return {
      "disease": disease_name,
      "confidence": float(np.max(pred)),
      "treatments": TREATMENTS.get(disease_name, TREATMENTS["Healthy"])
    }

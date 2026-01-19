import numpy as np
from PIL import Image
from io import BytesIO
import os
import logging

logger = logging.getLogger(__name__)

class PlantDiseaseModel:
    def __init__(self):
        """Initialize the plant disease detection model with simple classification"""
        self.disease_classes = [
            "Healthy",
            "Tomato Early Blight",
            "Tomato Late Blight",
            "Potato Early Blight",
            "Potato Late Blight",
            "Corn Common Rust",
            "Pepper Bell Bacterial Spot",
            "Apple Scab",
            "Grape Black Rot",
            "Powdery Mildew"
        ]
        self.input_size = (224, 224)
        logger.info(f"Model initialized with {len(self.disease_classes)} disease classes")
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image for model inference.
        Handles image loading, resizing, and normalization.
        """
        try:
            image = Image.open(BytesIO(image_bytes))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = image.resize(self.input_size, Image.Resampling.LANCZOS)
            img_array = np.array(image, dtype=np.float32)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {e}")
    
    def predict(self, image_bytes: bytes) -> dict:
        """
        Make prediction on uploaded image using simulated ML model.
        In production, replace this with actual TensorFlow model inference.
        """
        try:
            processed_image = self.preprocess_image(image_bytes)
            
            # Simulated predictions - Replace with actual model.predict()
            # In production: predictions = self.model.predict(processed_image, verbose=0)
            np.random.seed(hash(image_bytes) % 2**32)
            class_probabilities = np.random.dirichlet(np.ones(len(self.disease_classes)) * 2)
            
            predicted_class_idx = np.argmax(class_probabilities)
            predicted_class = self.disease_classes[predicted_class_idx]
            confidence = float(class_probabilities[predicted_class_idx])
            
            all_predictions = [
                {
                    "class": self.disease_classes[i],
                    "confidence": round(float(class_probabilities[i]), 4)
                }
                for i in range(len(self.disease_classes))
            ]
            all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
            
            return {
                "predicted_disease": predicted_class,
                "confidence": confidence,
                "all_predictions": all_predictions[:5],
                "success": True
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "error": str(e),
                "success": False
            }

plant_disease_model = None

def get_model() -> PlantDiseaseModel:
    """Dependency for obtaining model instance"""
    global plant_disease_model
    if plant_disease_model is None:
        plant_disease_model = PlantDiseaseModel()
    return plant_disease_model
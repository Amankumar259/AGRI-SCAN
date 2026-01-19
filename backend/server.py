from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from ml.model import get_model

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
try:
    client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=2000)
    db = client[os.environ.get('DB_NAME', 'test_database')]
except Exception as e:
    logging.warning(f"MongoDB connection warning: {e}")
    client = None
    db = None

app = FastAPI(title="AgriScan AI - Plant Disease Detection API")
api_router = APIRouter(prefix="/api")

# Models
class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    prediction_id: str
    filename: str
    predicted_disease: str
    confidence: float
    all_predictions: List[dict]
    timestamp: str

class PredictionHistory(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    filename: str
    predicted_disease: str
    confidence: float
    timestamp: str

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
ALLOWED_MIMETYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}
MAX_FILE_SIZE = 25 * 1024 * 1024

async def validate_upload_file(file: UploadFile) -> bool:
    """Validates uploaded file for security and compatibility"""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file selected"
        )
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type {file_ext} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    if file.content_type not in ALLOWED_MIMETYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"MIME type {file.content_type} not allowed"
        )
    
    return True

@api_router.get("/")
async def root():
    return {"message": "AgriScan AI - Plant Disease Detection API", "version": "1.0"}

@api_router.post("/predictions/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Accept image upload and return disease prediction.
    Validates file, processes image, runs model inference, and stores result in MongoDB.
    """
    try:
        await validate_upload_file(file)
        
        file_content = await file.read()
        
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed size of 25MB"
            )
        
        model = get_model()
        prediction_result = model.predict(file_content)
        
        if not prediction_result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=prediction_result.get("error", "Prediction failed")
            )
        
        prediction_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        prediction_doc = {
            "prediction_id": prediction_id,
            "filename": file.filename,
            "timestamp": timestamp.isoformat(),
            "predicted_disease": prediction_result["predicted_disease"],
            "confidence": prediction_result["confidence"],
            "all_predictions": prediction_result["all_predictions"],
            "model_version": "1.0"
        }
        
        await db.predictions.insert_one(prediction_doc)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "prediction_id": prediction_id,
                "filename": file.filename,
                "predicted_disease": prediction_result["predicted_disease"],
                "confidence": round(prediction_result["confidence"], 4),
                "all_predictions": prediction_result["all_predictions"],
                "timestamp": timestamp.isoformat()
            }
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@api_router.get("/predictions/history")
async def get_prediction_history(limit: int = 20, skip: int = 0):
    """Retrieve prediction history from database"""
    try:
        predictions = await db.predictions.find(
            {},
            {"_id": 0}
        ).sort("timestamp", -1).skip(skip).limit(limit).to_list(length=limit)
        
        return {
            "total": await db.predictions.count_documents({}),
            "results": predictions
        }
    except Exception as e:
        logging.error(f"History retrieval error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@api_router.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "AgriScan AI",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
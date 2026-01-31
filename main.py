from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from schemas import VoiceDetectionRequest, VoiceDetectionResponse, ErrorResponse
from auth import verify_api_key
from audio_processor import decode_base64_audio, extract_features
from model_manager import get_model_manager
from config import SUPPORTED_LANGUAGES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Voice Detection API",
    description="Detects whether a voice sample is AI-generated or Human",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    try:
        get_model_manager()
        logger.info("Model and scaler loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "operational",
        "api": "AI Voice Detection",
        "version": "1.0.0"
    }


@app.post(
    "/api/voice-detection",
    response_model=VoiceDetectionResponse,
    tags=["Voice Detection"],
    summary="Detect AI vs Human Voice",
    description="Analyzes an MP3 audio file (Base64 encoded) and classifies whether it's AI-generated or Human"
)
async def voice_detection(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
) -> VoiceDetectionResponse:
    try:
        if request.language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported language. Supported: {', '.join(SUPPORTED_LANGUAGES)}"
            )
        
        if request.audioFormat != "mp3":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only MP3 format is supported"
            )

        if not request.audioBase64 or len(request.audioBase64.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio Base64 cannot be empty"
            )
        
        logger.info(f"Processing: language={request.language}")
        
        audio_bytes = decode_base64_audio(request.audioBase64)
        logger.info(f"Audio decoded: {len(audio_bytes)} bytes")
        
        features = extract_features(audio_bytes)
        logger.info(f"Features extracted: shape {features.shape}")

        model_mgr = get_model_manager()
        classification, confidence, explanation = model_mgr.predict(features)
        logger.info(f"Prediction: {classification} (confidence: {confidence:.4f})")

        return VoiceDetectionResponse(
            status="success",
            language=request.language,
            classification=classification,
            confidenceScore=round(confidence, 4),
            explanation=explanation
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

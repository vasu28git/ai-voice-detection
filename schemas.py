from pydantic import BaseModel, Field


class VoiceDetectionRequest(BaseModel):
    language: str = Field(..., description="Tamil, English, Hindi, Malayalam, or Telugu")
    audioFormat: str = Field(..., description="Always 'mp3'")
    audioBase64: str = Field(..., description="Base64 encoded MP3 audio")


class VoiceDetectionResponse(BaseModel):

    status: str = Field(default="success", description="success or error")
    language: str = Field(..., description="Language of the audio")
    classification: str = Field(..., description="AI_GENERATED or HUMAN")
    confidenceScore: float = Field(..., description="Confidence between 0.0 and 1.0")
    explanation: str = Field(..., description="Reasoning for the classification")


class ErrorResponse(BaseModel):

    status: str = Field(default="error")
    message: str = Field(...)

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from typing import Optional
import os
from pydantic import BaseModel

from app.models.whisper_model import whisper_model
from app.utils.audio_utils import is_valid_audio_file, save_upload_file, normalize_audio
from app.core.config import settings

router = APIRouter()

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    segments: list

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """
    Transcribe an audio file to text.
    
    Parameters:
    - file: Audio file to transcribe
    - language: Optional ISO language code (e.g., 'en', 'fr', 'es')
    
    Returns:
    - Transcription result with text and metadata
    """
    # Validate file size
    if file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size allowed is {settings.MAX_UPLOAD_SIZE/(1024*1024)}MB"
        )
    
    # Validate file type
    content_type = file.content_type
    if not content_type or not is_valid_audio_file(content_type):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Allowed types: {', '.join(settings.ALLOWED_AUDIO_TYPES)}"
        )
    
    try:
        # Process the audio file
        file_data = await file.read()
        await file.seek(0)  # Reset file pointer
        
        # Transcribe the audio
        result = whisper_model.transcribe_audio(file, language)
        
        # Create response
        response = TranscriptionResponse(
            text=result["text"],
            language=result["language"],
            segments=result["segments"]
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )

@router.get("/info")
async def get_info():
    """Get information about the API and its configuration."""
    return {
        "model": settings.MODEL_SIZE,
        "device": settings.DEVICE,
        "max_upload_size": f"{settings.MAX_UPLOAD_SIZE/(1024*1024)}MB",
        "allowed_formats": settings.ALLOWED_AUDIO_TYPES
    }
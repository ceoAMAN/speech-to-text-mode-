import os
import uuid
import torch
import tempfile
from pathlib import Path
from typing import BinaryIO, Optional, List, Dict, Any

import whisper

from app.core.config import settings

class WhisperModel:
    """Wrapper for the OpenAI Whisper model."""
    
    def __init__(self):
        self.model = None
        self.model_name = settings.MODEL_SIZE
        self.device = settings.DEVICE
        print(f"Initializing Whisper model '{self.model_name}' on {self.device}")
    
    def load_model(self):
        """Load the Whisper model if not already loaded."""
        if self.model is None:
            self.model = whisper.load_model(self.model_name, device=self.device)
        return self.model
    
    def transcribe_file(self, file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe an audio file.
        
        Args:
            file_path: Path to the audio file
            language: Optional language code (ISO 639-1)
            
        Returns:
            Dictionary containing transcription results
        """
        model = self.load_model()
        
        # Set transcription options
        options = {}
        if language:
            options["language"] = language
        
        # Perform transcription
        result = model.transcribe(file_path, **options)
        
        return {
            "text": result["text"],
            "segments": result["segments"],
            "language": result.get("language", language or "auto-detected")
        }
    
    def transcribe_audio(self, audio_file: BinaryIO, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an uploaded audio file and transcribe it.
        
        Args:
            audio_file: File-like object containing audio data
            language: Optional language code
            
        Returns:
            Dictionary containing transcription results
        """
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
        temp_file_path = temp_file.name
        
        try:
            # Write uploaded file to temporary file
            temp_file.write(audio_file.read())
            temp_file.close()
            
            # Perform transcription
            result = self.transcribe_file(temp_file_path, language)
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

# Create a singleton instance
whisper_model = WhisperModel()
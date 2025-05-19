import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Tuple, Optional

import torchaudio
import numpy as np
import ffmpeg

from app.core.config import settings

def is_valid_audio_file(content_type: str) -> bool:
    """
    Check if the content type is a valid audio format.
    
    Args:
        content_type: MIME type of the uploaded file
        
    Returns:
        Boolean indicating whether the file type is supported
    """
    return content_type in settings.ALLOWED_AUDIO_TYPES

def get_audio_info(file_path: str) -> Tuple[int, int, float]:
    """
    Get audio file information.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Tuple of (sample_rate, num_channels, duration_seconds)
    """
    try:
        metadata = ffmpeg.probe(file_path)
        audio_info = next(s for s in metadata['streams'] if s['codec_type'] == 'audio')
        
        sample_rate = int(audio_info.get('sample_rate', 0))
        num_channels = int(audio_info.get('channels', 0))
        
        # Calculate duration
        duration = float(metadata.get('format', {}).get('duration', 0))
        
        return sample_rate, num_channels, duration
        
    except ffmpeg.Error:
        # Fall back to torchaudio if ffmpeg probe fails
        try:
            info = torchaudio.info(file_path)
            return info.sample_rate, info.num_channels, info.num_frames / info.sample_rate
        except Exception:
            return 0, 0, 0

def save_upload_file(file: BinaryIO, filename: Optional[str] = None) -> str:
    """
    Save an uploaded file to the upload directory.
    
    Args:
        file: File-like object
        filename: Optional filename (will be generated if not provided)
        
    Returns:
        Path to the saved file
    """
    if filename is None:
        # Generate a random filename
        ext = Path(getattr(file, 'filename', '')).suffix
        filename = f"{os.urandom(8).hex()}{ext}"
    
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write the file
    with open(file_path, "wb") as f:
        f.write(file.read())
    
    return file_path

def normalize_audio(file_path: str) -> str:
    """
    Normalize audio to prepare for transcription.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Path to the normalized audio file
    """
    try:
        # Create a temporary file for the normalized audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Use ffmpeg to normalize the audio to 16kHz mono
        (
            ffmpeg
            .input(file_path)
            .output(temp_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
        
        return temp_path
        
    except ffmpeg.Error as e:
        print(f"Error normalizing audio: {e.stderr.decode() if e.stderr else str(e)}")
        # Return the original file path if normalization fails
        return file_path
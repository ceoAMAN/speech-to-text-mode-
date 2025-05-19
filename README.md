# Speech-to-Text API with Whisper

A FastAPI-based API that uses OpenAI's Whisper model to transcribe speech to text. The application supports file uploads and can utilize GPU acceleration when available.

## Features

- Audio file upload endpoint for transcription
- Multiple language support
- Automatic device selection (CPU/GPU)
- Configurable model size (tiny, base, small, medium, large)
- Proper error handling and validation

## Folder Structure

```
speech_to_text_api/
├── app/
│   ├── main.py                # FastAPI application entry point
│   ├── api/
│   │   └── routes.py          # API route definitions
│   ├── core/
│   │   └── config.py          # Configuration settings
│   ├── models/
│   │   └── whisper_model.py   # Whisper model wrapper
│   ├── utils/
│   │   └── audio_utils.py     # Audio processing utilities
├── tests/                     # Test files
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── .env                       # Environment variables
└── .gitignore                 # Git ignore file
```

## Installation

1. Clone the repository
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install ffmpeg (required for audio processing):
   - **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

4. Create a `.env` file with your configuration

## Usage

Start the API server:

```
cd speech_to_text_api
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

- `POST /api/transcribe/`: Upload an audio file for transcription

## Requirements

- Python 3.8+
- ffmpeg
- PyTorch
- FastAPI
- openai-whisper
- torchaudio

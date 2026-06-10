"""
Configuration Module

Contains all configuration settings for the Audio Language Model system.
Updated for OpenRouter API integration.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# OPENROUTER API CONFIGURATION
# ============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model to use via OpenRouter
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4-turbo")

# ============================================================================
# YAMNET MODEL CONFIGURATION
# ============================================================================

YAMNET_MODEL_PATH = 'https://tfhub.dev/google/yamnet/1'

# ============================================================================
# WHISPER MODEL CONFIGURATION
# ============================================================================

WHISPER_MODEL = 'base'
WHISPER_LANGUAGE = None
WHISPER_TEMPERATURE = 0.0

# ============================================================================
# AUDIO PROCESSING CONFIGURATION
# ============================================================================

AUDIO_SAMPLE_RATE = 16000
AUDIO_MONO = True
AUDIO_NORMALIZATION = True

SUPPORTED_FORMATS = ['wav', 'mp3', 'm4a', 'flac', 'ogg']

AUDIO_CHUNK_DURATION = 10
AUDIO_FRAME_LENGTH = 0.01

# ============================================================================
# YAMNET SOUND DETECTION
# ============================================================================

YAMNET_CONFIDENCE_THRESHOLD = 0.3
YAMNET_TOP_EVENTS = 5

# ============================================================================
# EMOTION DETECTION
# ============================================================================

EMOTION_CLASSES = [
    'neutral', 'calm', 'happy', 'sad',
    'angry', 'fearful', 'disgust', 'surprised'
]

EMOTION_CONFIDENCE_THRESHOLD = 0.3

# ============================================================================
# OPENROUTER SYSTEM PROMPT
# ============================================================================

OPENROUTER_SYSTEM_PROMPT = """
You are an advanced AI audio scene understanding system.

You analyze:
1. Speech transcripts
2. Environmental sound events
3. Emotional tone
4. Acoustic context
5. Emergency indicators

YOUR TASK:
Given integrated audio analysis data, generate a structured and professional report.

ANALYSIS REQUIREMENTS:
- Detect environment and location clues
- Identify activities/events occurring
- Assess emotional state
- Evaluate danger or emergency risk
- Estimate confidence score
- Base conclusions only on observable evidence
- Avoid hallucinations or unsupported assumptions

RISK DETECTION:
- Medical emergency indicators
- Conflict or violence indicators
- Vulnerability signals
- Dispatch/guidance phrases
- Distress language

OUTPUT FORMAT:
Return ONLY valid JSON with no markdown fences.

{
  "location": "...",
  "people": "...",
  "activity": "...",
  "emotion": "...",
  "risk_level": "low|moderate|high",
  "confidence": 0.0,
  "explanation": "..."
}
"""

# ============================================================================
# STREAMLIT UI CONFIGURATION
# ============================================================================

APP_TITLE = "Audio Language Model (ALM)"
APP_DESCRIPTION = "AI-Powered Audio Scene Understanding System"
APP_ICON = "🎵"

PAGE_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# ============================================================================
# LOGGING AND DEBUG
# ============================================================================

DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
LOG_LEVEL = 'INFO'

# ============================================================================
# FEATURE FLAGS
# ============================================================================

ENABLE_EMOTION_ANALYSIS = True
ENABLE_REASONING = True
ENABLE_ADVANCED_PREPROCESSING = True
ENABLE_SEMANTIC_ANALYSIS = True

# ============================================================================
# FILE UPLOAD SETTINGS
# ============================================================================

MAX_UPLOAD_SIZE = 50 * 1024 * 1024
UPLOAD_TEMP_DIR = './temp_uploads'

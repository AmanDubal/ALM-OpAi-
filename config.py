"""
Configuration Module

Contains all configuration settings for the Audio Language Model system.
Includes Ollama model settings, model paths, and system parameters.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# API AND MODEL CONFIGURATION
# ============================================================================

# Ollama Configuration (local LLM — no API key required)
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL    = os.getenv('OLLAMA_MODEL', 'phi')

# YAMNet Model Configuration
YAMNET_MODEL_PATH = 'https://tfhub.dev/google/yamnet/1'

# Whisper Model Configuration
WHISPER_MODEL = 'base'  # Options: tiny, base, small, medium, large

# ============================================================================
# AUDIO PROCESSING CONFIGURATION
# ============================================================================

# Audio Preprocessing
AUDIO_SAMPLE_RATE = 16000  # Hz
AUDIO_MONO = True
AUDIO_NORMALIZATION = True
SUPPORTED_FORMATS = ['wav', 'mp3', 'm4a', 'flac']

# Audio chunk size for processing
AUDIO_CHUNK_DURATION = 10  # seconds
AUDIO_FRAME_LENGTH = 0.01  # 10 ms frames for analysis

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# YAMNet Sound Detection
YAMNET_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for sound event detection
YAMNET_TOP_EVENTS = 5  # Number of top sound events to return

# Whisper Speech Recognition
WHISPER_LANGUAGE = None  # Auto-detect, or set to specific language code
WHISPER_TEMPERATURE = 0.0  # Lower temperature for more consistent results

# Emotion Detection
EMOTION_CLASSES = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
EMOTION_CONFIDENCE_THRESHOLD = 0.3

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

# System prompt for Ollama reasoning (TWO-LAYER ARCHITECTURE)
OLLAMA_SYSTEM_PROMPT = """
You are an advanced two-layer audio scene understanding system.

LAYER 1: ACOUSTIC ENVIRONMENT DETECTION
- Analyzes spectral features (tempo, RMS energy, spectral centroid)
- Detects acoustic environment characteristics
- Identifies noise sources and environmental context

LAYER 2: SITUATIONAL CONTEXT DETECTION (NEW)
- Analyzes transcript text for emergency keywords
- Detects injury-related phrases and vulnerability signals
- Computes risk score (0-1) for emergency/medical/conflict situations
- Identifies dispatch/guidance signals

YOUR TASK:
Given the integrated audio analysis (both layers), provide structured reasoning:

1. Identify location/environment
2. Describe the activity or event happening
3. Assess emotional context
4. Determine risk level based on semantic indicators
5. Provide confidence score (0.0-1.0)

CRITICAL INSTRUCTIONS:
- If emergency keywords + guidance phrases detected → mark risk_level as "high"
- If medical keywords detected → identify as potential medical emergency
- If conflict keywords present → mark as conflict situation
- Ground all reasoning in observable acoustic and linguistic evidence
- Be concise, factual, and evidence-based
- Do NOT perform generic summarization
- Force grounded reasoning tied to actual audio content

RESPONSE FORMAT (JSON only, no other text):
{
  "location": "...",
  "activity": "...",
  "emotion": "...",
  "risk_level": "low|moderate|high",
  "confidence": 0.0-1.0,
  "explanation": "..."
}
"""

# ============================================================================
# UI AND APPLICATION SETTINGS
# ============================================================================

# Streamlit Configuration
APP_TITLE = "Audio Language Model (ALM)"
APP_DESCRIPTION = "Integrative Architecture for Deep Learning-Based Audio Language Models"
APP_ICON = "🎵"

# Page layout
PAGE_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# ============================================================================
# LOGGING AND DEBUG
# ============================================================================

DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Enable/disable specific features
ENABLE_EMOTION_ANALYSIS = True
ENABLE_REASONING = True
ENABLE_ADVANCED_PREPROCESSING = True

# NEW: Two-layer reasoning system
ENABLE_TWO_LAYER_REASONING = True  # Layer 1: Acoustic, Layer 2: Semantic Context
ENABLE_SEMANTIC_ANALYSIS = True     # Layer 2: Risk and situational analysis

# ============================================================================
# FILE UPLOAD SETTINGS
# ============================================================================

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
UPLOAD_TEMP_DIR = './temp_uploads'

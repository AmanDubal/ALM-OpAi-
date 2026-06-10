"""
Flask Backend Server for ALM
Handles API requests from the frontend and processes audio using ALM modules
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

# Import ALM modules
from modules.preprocessing.audio_preprocessor import AudioPreprocessor
from modules.speech.speech_recognizer import SpeechRecognizer
from modules.sound_detection.sound_detector import SoundDetector
from modules.emotion.emotion_analyzer import EmotionAnalyzer
from modules.context.context_integrator import ContextIntegrator
from modules.context.semantic_analyzer import SemanticAnalyzer
from modules.context.adapter import DataAdapter
from modules.reasoning.inference_engine import InferenceEngine

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    WHISPER_MODEL,
    YAMNET_CONFIDENCE_THRESHOLD,
    ENABLE_EMOTION_ANALYSIS,
    ENABLE_REASONING,
)

# Initialize Flask
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# File storage for uploads
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}

# Cache for uploaded files
file_cache = {}

# Initialize modules
def initialize_modules():
    """Initialize all ALM processing modules."""
    engine = InferenceEngine()
    engine.setup_openrouter(
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_MODEL,
        base_url=OPENROUTER_BASE_URL,
    )
    return {
        'preprocessor': AudioPreprocessor(),
        'speech_recognizer': SpeechRecognizer(model_size=WHISPER_MODEL),
        'sound_detector': SoundDetector(confidence_threshold=YAMNET_CONFIDENCE_THRESHOLD),
        'emotion_analyzer': EmotionAnalyzer(),
        'context_integrator': ContextIntegrator(),
        'semantic_analyzer': SemanticAnalyzer(),
        'data_adapter': DataAdapter(),
        'inference_engine': engine,
    }

modules = initialize_modules()

# Routes
@app.route('/')
def index():
    """Serve the main HTML file."""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('.', path)

# API Routes
@app.route('/api/upload', methods=['POST'])
def upload_audio():
    """Upload and preprocess audio file."""
    try:
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'}), 400

        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f'{file_id}.wav')
        file.save(file_path)

        # Preprocess audio
        audio, sr = modules['preprocessor'].load_audio(file_path)
        duration = modules['preprocessor'].get_audio_duration(audio)
        audio_info = modules['preprocessor'].get_audio_info(audio)

        # Cache file info
        file_cache[file_id] = {
            'path': file_path,
            'audio': audio,
            'sr': sr,
        }

        return jsonify({
            'file_id': file_id,
            'duration': duration,
            'sample_rate': audio_info['sample_rate'],
            'peak_amplitude': audio_info['peak_amplitude'],
            'filename': file.filename,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/speech-recognition', methods=['POST'])
def speech_recognition():
    """Perform speech recognition on uploaded audio."""
    try:
        data = request.get_json()
        file_id = data.get('file_id')

        if file_id not in file_cache:
            return jsonify({'error': 'File not found'}), 404

        audio = file_cache[file_id]['audio']
        sr = file_cache[file_id]['sr']

        # Transcribe
        result = modules['speech_recognizer'].transcribe(audio, sr=sr)

        return jsonify({
            'transcript': result.get('text', ''),
            'language': result.get('language', 'unknown'),
            'confidence': 0.95,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sound-detection', methods=['POST'])
def sound_detection():
    """Detect sound events in audio."""
    try:
        data = request.get_json()
        file_id = data.get('file_id')

        if file_id not in file_cache:
            return jsonify({'error': 'File not found'}), 404

        audio = file_cache[file_id]['audio']
        sr = file_cache[file_id]['sr']

        # Detect sounds
        result = modules['sound_detector'].detect(audio, sr=sr)

        return jsonify({
            'events': result.get('events', []),
            'count': len(result.get('events', [])),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emotion-analysis', methods=['POST'])
def emotion_analysis():
    """Analyze emotion in audio."""
    try:
        data = request.get_json()
        file_id = data.get('file_id')

        if file_id not in file_cache:
            return jsonify({'error': 'File not found'}), 404

        if not ENABLE_EMOTION_ANALYSIS:
            return jsonify({
                'emotion': 'neutral',
                'confidence': 0.0,
            })

        audio = file_cache[file_id]['audio']
        sr = file_cache[file_id]['sr']

        # Analyze emotion
        result = modules['emotion_analyzer'].analyze(audio, sr=sr)

        return jsonify({
            'emotion': result.get('emotion', 'neutral'),
            'confidence': result.get('confidence', 0.0),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/context-integration', methods=['POST'])
def context_integration():
    """Integrate audio context."""
    try:
        data = request.get_json()
        file_id = data.get('file_id')

        if file_id not in file_cache:
            return jsonify({'error': 'File not found'}), 404

        transcript = data.get('transcript', '')
        sounds = data.get('sounds', [])
        emotion = data.get('emotion', 'neutral')

        # Integrate context
        context, formatted_context = modules['context_integrator'].integrate(
            transcript=transcript,
            sound_events=sounds,
            emotion=emotion,
        )

        return jsonify({
            'context': context,
            'formatted_context': formatted_context,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/inference', methods=['POST'])
def inference():
    """Generate AI inference via OpenRouter."""
    try:
        data = request.get_json()
        file_id = data.get('file_id')

        if file_id not in file_cache:
            return jsonify({'error': 'File not found'}), 404

        if not ENABLE_REASONING:
            return jsonify({
                'inference': 'Reasoning is disabled in settings.',
            })

        transcript = data.get('transcript', '')
        context = data.get('context', '')

        # Semantic analysis
        risk_analysis = modules['semantic_analyzer'].analyze(transcript)

        # Merge all analysis
        merged_context = modules['data_adapter'].merge_all_analysis(
            audio_analysis=None,
            speech_data={
                'transcript': transcript,
                'language': 'unknown',
                'confidence': 0.95,
            },
            sound_events=[],
            emotion_data={
                'emotional_state': 'neutral',
                'confidence': 0.7,
                'vocal_tension': 'unknown',
            },
            risk_analysis=risk_analysis,
        )

        # Generate inference
        inference_text = modules['inference_engine'].generate_inference_v2(
            context=merged_context,
            risk_analysis=risk_analysis,
        )

        return jsonify({
            'inference': inference_text,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'modules_loaded': True,
    })

# Utility functions
def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎵 ALM - Audio Language Model Server")
    print("="*60)
    print("✓ Frontend: http://localhost:5000")
    print("✓ API Health: http://localhost:5000/api/health")
    print("✓ OpenRouter Model:", OPENROUTER_MODEL)
    print("✓ Emotion Analysis:", "Enabled" if ENABLE_EMOTION_ANALYSIS else "Disabled")
    print("✓ AI Reasoning:", "Enabled" if ENABLE_REASONING else "Disabled")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

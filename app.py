"""
Audio Language Model (ALM) - Main Application

A Streamlit-based application that integrates multiple audio understanding modules
to perform contextual reasoning over audio scenes.

Project: Integrative Architectures for Deep Learning-Based Audio Language Models

Architecture:
    Listen → Think → Understand
    
    1. Audio Input & Preprocessing
    2. Speech Recognition (Whisper)
    3. Sound Event Detection (YAMNet)
    4. Emotion Analysis
    5. Context Integration
    6. Reasoning & Inference (Ollama)
"""

import streamlit as st
import numpy as np
import tempfile
import os
from datetime import datetime
from pathlib import Path

# Import modules
from modules.preprocessing.audio_preprocessor import AudioPreprocessor
from modules.speech.speech_recognizer import SpeechRecognizer
from modules.sound_detection.sound_detector import SoundDetector
from modules.emotion.emotion_analyzer import EmotionAnalyzer
from modules.context.context_integrator import ContextIntegrator
from modules.context.semantic_analyzer import SemanticAnalyzer
from modules.context.adapter import DataAdapter
from modules.reasoning.inference_engine import InferenceEngine

from config import (
    APP_TITLE, APP_DESCRIPTION, PAGE_LAYOUT, INITIAL_SIDEBAR_STATE,
    OLLAMA_BASE_URL, OLLAMA_MODEL, WHISPER_MODEL, YAMNET_CONFIDENCE_THRESHOLD,
    ENABLE_EMOTION_ANALYSIS, ENABLE_REASONING
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)
os.system("ollama run phi3")


# ============================================================================
# CUSTOM CSS FOR AI THEME & ANIMATIONS
# ============================================================================

def inject_custom_css():
    """Inject custom CSS for modern AI theme with animations."""
    custom_css = """
    <style>
        * {
            margin: 0;
            padding: 0;
        }
        
        /* Main background with gradient */
        .main {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3f 50%, #0f0f23 100%);
            color: #e0e0e0;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }
        
        /* Smooth header styling */
        h1, h2, h3 {
            background: linear-gradient(120deg, #00d4ff, #7c3aed, #00d4ff);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: gradientShift 3s ease infinite;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Animated container */
        .animated-container {
            background: rgba(20, 20, 40, 0.6);
            border: 2px solid rgba(0, 212, 255, 0.3);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
            animation: slideInUp 0.6s ease-out;
        }
        
        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Pulsing glow effect */
        .pulse-glow {
            animation: pulseGlow 2s ease-in-out infinite;
        }
        
        @keyframes pulseGlow {
            0%, 100% {
                box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
            }
            50% {
                box-shadow: 0 0 20px rgba(0, 212, 255, 0.6);
            }
        }
        
        /* Floating animation */
        .float-up {
            animation: floatUp 0.8s ease-out forwards;
        }
        
        @keyframes floatUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Processing wave animation */
        @keyframes wave {
            0%, 100% { transform: scaleY(1); }
            50% { transform: scaleY(1.5); }
        }
        
        .wave-bar {
            display: inline-block;
            width: 4px;
            height: 20px;
            background: linear-gradient(180deg, #00d4ff, #7c3aed);
            margin: 0 2px;
            border-radius: 2px;
            animation: wave 0.6s ease-in-out infinite;
        }
        
        /* Status badges */
        .status-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-success {
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.5);
        }
        
        .status-processing {
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
            border: 1px solid rgba(0, 212, 255, 0.5);
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: rgba(30, 30, 50, 0.5);
            padding: 10px;
            border-radius: 10px;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: 2px solid transparent;
            border-radius: 8px;
            padding: 10px 20px;
            color: #a0a0c0;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] [data-baseweb="tab"] {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(124, 58, 237, 0.2));
            border-color: #00d4ff;
            color: #00d4ff;
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #00d4ff, #7c3aed);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 12px 24px !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
        }
        
        /* Input styling */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select,
        .stSlider > div > div > div > input {
            background: rgba(30, 30, 50, 0.8) !important;
            border: 2px solid rgba(0, 212, 255, 0.3) !important;
            color: #e0e0e0 !important;
            border-radius: 8px !important;
        }
        
        /* Alert styling */
        .stAlert {
            background: rgba(30, 30, 50, 0.8) !important;
            border-left: 4px solid;
            border-radius: 8px !important;
            backdrop-filter: blur(10px);
        }
        
        .stAlert > div > div {
            color: #e0e0e0 !important;
        }
        
        /* Metric styling */
        .stMetric {
            background: rgba(20, 20, 40, 0.8) !important;
            border: 2px solid rgba(0, 212, 255, 0.2) !important;
            padding: 20px !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1) !important;
        }
        
        /* Code block styling */
        .stCode {
            background: rgba(10, 10, 20, 0.9) !important;
            border: 2px solid rgba(0, 212, 255, 0.2) !important;
            border-radius: 8px !important;
        }
        
        /* Divider styling */
        .stDivider {
            background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), transparent) !important;
        }
        
        /* Custom header animation */
        .header-glow {
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5), 0 0 20px rgba(124, 58, 237, 0.3);
        }
        
        /* Sidebar header */
        .sidebar-header {
            text-align: center;
            padding: 20px 0;
            border-bottom: 2px solid rgba(0, 212, 255, 0.3);
            margin-bottom: 20px;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Inject CSS on app load
inject_custom_css()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_modules():
    """Initialize all processing modules."""
    inference_engine = InferenceEngine()
    # Connect to local Ollama
    inference_engine.setup_ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    
    return {
        'preprocessor': AudioPreprocessor(),
        'speech_recognizer': SpeechRecognizer(model_size=WHISPER_MODEL),
        'sound_detector': SoundDetector(confidence_threshold=YAMNET_CONFIDENCE_THRESHOLD),
        'emotion_analyzer': EmotionAnalyzer(),
        'context_integrator': ContextIntegrator(),
        'semantic_analyzer': SemanticAnalyzer(),
        'data_adapter': DataAdapter(),
        'inference_engine': inference_engine
    }


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def render_header():
    """Render animated header with AI theme."""
    col1, col2, col3 = st.columns([0.5, 2, 0.5])
    with col2:
        st.markdown("""
            <div style="text-align: center; padding: 30px 0;">
                <h1 style="font-size: 3.5em; margin: 0; letter-spacing: 2px;">🎵 ALM</h1>
                <p style="font-size: 1.3em; color: #00d4ff; margin: 10px 0; letter-spacing: 1px;">
                    AUDIO LANGUAGE MODEL
                </p>
                <p style="color: #a0a0c0; font-size: 1em; margin-top: 10px;">
                    🧠 Understand Audio with AI Intelligence
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border: 2px solid rgba(0, 212, 255, 0.3);'>", unsafe_allow_html=True)


def render_progress_step(step_num, total_steps, title, emoji):
    """Render animated progress indicator."""
    progress = step_num / total_steps
    st.markdown(f"""
        <div class="animated-container pulse-glow">
            <div style="display: flex; align-items: center; gap: 15px;">
                <span style="font-size: 2em;">{emoji}</span>
                <div style="flex: 1;">
                    <p style="color: #00d4ff; font-weight: bold; margin: 0; text-transform: uppercase; letter-spacing: 0.5px;">
                        Step {step_num}/{total_steps} · {title}
                    </p>
                    <div style="background: rgba(0, 212, 255, 0.1); height: 4px; border-radius: 2px; margin-top: 8px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #00d4ff, #7c3aed); height: 100%; width: {progress*100}%; border-radius: 2px;"></div>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_success_badge(title):
    """Render success completion badge."""
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; padding: 15px; background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; border-radius: 8px; margin: 10px 0;">
            <span style="font-size: 1.5em;">✅</span>
            <span style="color: #10b981; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">{title}</span>
        </div>
    """, unsafe_allow_html=True)


def render_metric_grid(metrics):
    """Render metrics in animated grid."""
    cols = st.columns(len(metrics))
    for idx, (col, (label, value)) in enumerate(zip(cols, metrics.items())):
        with col:
            st.metric(label, value)


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Main application function."""
    
    # Render animated header
    render_header()
    # Create tabs with custom styling
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎤 PROCESS AUDIO", "📊 SYSTEM ARCHITECTURE", "⚙️ SETTINGS", "📚 GUIDE"]
    )
    
    # Initialize modules
    modules = initialize_modules()
    
    # ========================================================================
    # TAB 1: PROCESS AUDIO
    # ========================================================================
    
    with tab1:
        st.markdown("<h2 style='text-align: center;'>🎵 Audio Processing Pipeline</h2>", unsafe_allow_html=True)
        st.markdown("")
        
        # Upload section with enhanced styling
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
                <div class="animated-container" style="padding: 25px;">
                    <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px;">
                        📥 Upload Your Audio File
                    </p>
                </div>
            """, unsafe_allow_html=True)
            audio_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("""
                <div style="background: rgba(0, 212, 255, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #00d4ff; text-align: center;">
                    <p style="color: #00d4ff; font-weight: bold; font-size: 0.9em; margin: 0;">MAX SIZE</p>
                    <p style="color: #e0e0e0; font-weight: bold; font-size: 1.2em; margin: 5px 0;">50 MB</p>
                </div>
            """, unsafe_allow_html=True)
        
        if audio_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_path = tmp_file.name
            
            try:
                # Create progress tracking
                progress_placeholder = st.empty()
                results_placeholder = st.empty()
                
                # Step 1: Load and preprocess audio
                with progress_placeholder.container():
                    render_progress_step(1, 6, "Loading & Preprocessing Audio", "📥")
                
                audio, sr = modules['preprocessor'].load_audio(tmp_path)
                audio_duration = modules['preprocessor'].get_audio_duration(audio)
                audio_info = modules['preprocessor'].get_audio_info(audio)
                
                # Display audio info with animations
                with results_placeholder.container():
                    render_success_badge("Audio Loaded Successfully")
                    st.markdown("")
                    metrics = {
                        "⏱️ Duration": f"{audio_duration:.2f}s",
                        "🔊 Sample Rate": f"{audio_info['sample_rate']} Hz",
                        "📈 Peak Amplitude": f"{audio_info['peak_amplitude']:.3f}"
                    }
                    render_metric_grid(metrics)
                
                # Step 2: Speech Recognition
                with progress_placeholder.container():
                    render_progress_step(2, 6, "Performing Speech Recognition", "🗣️")
                
                speech_result = modules['speech_recognizer'].transcribe(audio, sr=sr)
                transcript = speech_result.get('text', '')
                speech_language = speech_result.get('language', 'unknown')
                
                with results_placeholder.container():
                    render_success_badge("Speech Recognition Complete")
                    if transcript:
                        st.markdown("""
                            <div class="animated-container" style="margin-top: 15px;">
                                <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;">
                                    📝 Transcribed Text
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.text_area(
                            "Transcript",
                            value=transcript,
                            height=100,
                            disabled=True,
                            label_visibility="collapsed"
                        )
                        st.caption(f"🌐 Detected Language: **{speech_language.upper()}**")
                    else:
                        st.markdown("""
                            <div style="background: rgba(239, 68, 68, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #ef4444;">
                                <p style="color: #fca5a5; font-weight: bold; margin: 0;">⚠️ No speech detected in audio</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Step 3: Sound Event Detection
                with progress_placeholder.container():
                    render_progress_step(3, 6, "Detecting Sound Events", "🔊")
                
                sound_result = modules['sound_detector'].detect(audio, sr=sr)
                sounds_detected = sound_result.get('events', [])
                
                with results_placeholder.container():
                    render_success_badge("Sound Detection Complete")
                    if sounds_detected:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("""
                                <div class="animated-container">
                                    <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">
                                        🔊 Detected Sound Events
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                            for i, sound in enumerate(sounds_detected, 1):
                                st.markdown(f"• **{sound}**")
                        
                        with col2:
                            st.markdown("""
                                <div class="animated-container">
                                    <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">
                                        📊 Categorized Events
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                            categorized = modules['sound_detector'].categorize_events(sounds_detected)
                            for category, events in categorized.items():
                                if events:
                                    st.markdown(f"**{category.replace('_', ' ').title()}:** {len(events)}")
                    else:
                        st.markdown("""
                            <div style="background: rgba(59, 130, 246, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                                <p style="color: #93c5fd; font-weight: bold; margin: 0;">ℹ️ No significant sound events detected</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Step 4: Emotion Analysis
                emotion_detected = "neutral"
                if ENABLE_EMOTION_ANALYSIS:
                    with progress_placeholder.container():
                        render_progress_step(4, 6, "Analyzing Emotional Tone", "😊")
                    
                    emotion_result = modules['emotion_analyzer'].analyze(audio, sr=sr)
                    emotion_detected = emotion_result.get('emotion', 'unknown')
                    emotion_confidence = emotion_result.get('confidence', 0.0)
                    
                    with results_placeholder.container():
                        render_success_badge("Emotion Analysis Complete")
                        metrics = {
                            "😊 Emotion": emotion_detected.upper(),
                            "🎯 Confidence": f"{emotion_confidence:.1%}"
                        }
                        render_metric_grid(metrics)
                
                # Step 5: Context Integration
                with progress_placeholder.container():
                    render_progress_step(5, 6, "Integrating Audio Context", "🔗")
                
                context, formatted_context = modules['context_integrator'].integrate(
                    transcript=transcript,
                    sound_events=sounds_detected,
                    emotion=emotion_detected
                )
                
                with results_placeholder.container():
                    render_success_badge("Context Integration Complete")
                    st.markdown("""
                        <div class="animated-container" style="margin-top: 15px;">
                            <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;">
                                🔗 Integrated Context
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.text_area(
                        "Context",
                        value=formatted_context,
                        height=250,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                
                # Step 6: Reasoning & Inference (TWO-LAYER SYSTEM)
                inference_text = ""
                if ENABLE_REASONING:
                    with progress_placeholder.container():
                        render_progress_step(6, 6, "Generating AI Inference (Two-Layer System)", "🧠")
                    
                    # Layer 2: Semantic Context Analysis
                    risk_analysis = modules['semantic_analyzer'].analyze(transcript)
                    
                    # Merge all analysis (Layer 1 + Layer 2)
                    merged_context = modules['data_adapter'].merge_all_analysis(
                        audio_analysis=None,  # Would come from InferenceEngine.analyze() in production
                        speech_data={
                            'transcript': transcript,
                            'language': speech_language,
                            'confidence': 1.0
                        },
                        sound_events=[{'event': sound, 'confidence': 0.7} for sound in sounds_detected],
                        emotion_data={
                            'emotional_state': emotion_detected,
                            'confidence': 0.7,
                            'vocal_tension': 'unknown'
                        },
                        risk_analysis=risk_analysis
                    )
                    
                    # Generate inference with merged context
                    inference_text = modules['inference_engine'].generate_inference_v2(
                        context=merged_context,
                        risk_analysis=risk_analysis
                    )
                    
                    with results_placeholder.container():
                        render_success_badge("Two-Layer Reasoning Complete")
                        st.markdown("""
                            <div class="animated-container" style="margin-top: 15px; background: linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(0, 212, 255, 0.1));">
                                <p style="color: #a78bfa; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 15px;">
                                    🧠 Two-Layer AI Reasoning (Acoustic + Semantic)
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.write(inference_text)
                else:
                    with progress_placeholder.container():
                        st.info("ℹ️ Reasoning disabled in settings")
                
                # Clear progress message
                progress_placeholder.empty()
                
                # Download results section
                st.markdown("<hr style='border: 2px solid rgba(0, 212, 255, 0.3); margin: 30px 0;'>", unsafe_allow_html=True)
                
                st.markdown("""
                    <div class="animated-container">
                        <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin: 0;">
                            📥 Download Results
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                results_text = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║               AUDIO LANGUAGE MODEL (ALM) - ANALYSIS RESULTS                    ║
║                                                                                ║
║                  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                          ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝

┌─ 1. AUDIO INFORMATION
│
├─ Duration: {audio_duration:.2f} seconds
├─ Sample Rate: {audio_info['sample_rate']} Hz
└─ Peak Amplitude: {audio_info['peak_amplitude']:.3f}

┌─ 2. SPEECH RECOGNITION
│
├─ Transcript: {transcript if transcript else "No speech detected"}
└─ Detected Language: {speech_language}

┌─ 3. SOUND EVENTS
│
└─ Detected Events: {', '.join(sounds_detected) if sounds_detected else "None"}

┌─ 4. EMOTION ANALYSIS
│
└─ Detected Emotion: {emotion_detected}

┌─ 5. INTEGRATED CONTEXT
│
└─ {formatted_context}

┌─ 6. AI INFERENCE
│
└─ {inference_text if ENABLE_REASONING else "Reasoning disabled"}

════════════════════════════════════════════════════════════════════════════════
"""
                
                st.download_button(
                    label="📄 Download Analysis Results (TXT)",
                    data=results_text,
                    file_name=f"alm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    
    # ========================================================================
    # TAB 2: ABOUT SYSTEM
    # ========================================================================
    
    with tab2:
        st.markdown("<h2 style='text-align: center;'>🏗️ System Architecture</h2>", unsafe_allow_html=True)
        st.markdown("")
        
        st.markdown("""
            <div class="animated-container">
                <h3 style="color: #00d4ff; margin-top: 0;">🎯 Listen → Think → Understand</h3>
                <p style="color: #e0e0e0; line-height: 1.8;">
                    The ALM system processes audio through an intelligent pipeline of specialized modules 
                    that work together to extract meaning and context from sound.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Architecture modules
        modules_info = [
            {
                "title": "Audio Input & Preprocessing",
                "emoji": "📥",
                "description": "Standardizes audio format (mono, 16 kHz) and normalizes amplitude",
                "tools": "Librosa, Pydub"
            },
            {
                "title": "Speech Recognition",
                "emoji": "🗣️",
                "description": "Extracts spoken language content with multilingual support (Hindi & English)",
                "tools": "OpenAI Whisper"
            },
            {
                "title": "Sound Event Detection",
                "emoji": "🔊",
                "description": "Identifies environmental sounds (traffic, crowd, aircraft, alarms, machinery)",
                "tools": "YAMNet CNN-based model"
            },
            {
                "title": "Emotion Analysis",
                "emoji": "😊",
                "description": "Captures emotional and paralinguistic cues from audio features",
                "tools": "MFCC Feature Extraction"
            },
            {
                "title": "Context Integration ⭐",
                "emoji": "🔗",
                "description": "Merges all outputs into unified context representation (Core Innovation)",
                "tools": "Custom Integration Engine"
            },
            {
                "title": "Reasoning & Inference",
                "emoji": "🧠",
                "description": "Performs logical reasoning and generates scene understanding",
                "tools": "Ollama (Local LLM)"
            }
        ]
        
        cols = st.columns(2)
        for idx, module in enumerate(modules_info):
            with cols[idx % 2]:
                st.markdown(f"""
                    <div class="animated-container" style="margin-bottom: 15px;">
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                            <span style="font-size: 2em;">{module['emoji']}</span>
                            <div>
                                <p style="color: #00d4ff; font-weight: bold; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; font-size: 0.95em;">
                                    {module['title']}
                                </p>
                            </div>
                        </div>
                        <p style="color: #e0e0e0; margin: 10px 0; font-size: 0.95em; line-height: 1.6;">
                            {module['description']}
                        </p>
                        <p style="color: #a0a0c0; margin: 0; font-size: 0.85em;">
                            🛠️ <strong>{module['tools']}</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Key advantages
        st.markdown("""
            <div class="animated-container">
                <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px;">
                    🚀 Key Advantages
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        advantages = [
            ("Integrated Understanding", "Combines speech, sound, and emotion modalities"),
            ("Context-Aware Reasoning", "Reasons over joint context instead of isolated predictions"),
            ("Modular Architecture", "Each component is independent and easily testable"),
            ("Scalable Design", "Easy to add new modules or features"),
            ("Human-Like Perception", "Mimics how humans understand audio scenes")
        ]
        
        cols = st.columns(1)
        for emoji, (title, desc) in enumerate(advantages):
            st.markdown(f"""
                <div style="display: flex; gap: 15px; margin-bottom: 12px; padding: 12px; background: rgba(0, 212, 255, 0.05); border-radius: 8px; border-left: 3px solid #00d4ff;">
                    <span style="font-size: 1.5em;">✅</span>
                    <div>
                        <p style="color: #00d4ff; font-weight: bold; margin: 0; font-size: 0.95em;">{title}</p>
                        <p style="color: #a0a0c0; margin: 5px 0; font-size: 0.9em;">{desc}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 3: SETTINGS
    # ========================================================================
    
    with tab3:
        st.markdown("<h2 style='text-align: center;'>⚙️ System Settings</h2>", unsafe_allow_html=True)
        st.markdown("")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="animated-container">
                    <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px;">
                        🎛️ Preprocessing
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("""
                <div style="background: rgba(16, 185, 129, 0.1); padding: 12px; border-radius: 8px; border-left: 4px solid #10b981;">
                    <p style="color: #10b981; font-weight: bold; margin: 0;">✅ Advanced preprocessing enabled</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            st.markdown("""
                <div class="animated-container">
                    <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px;">
                        🤖 Models
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div style="background: rgba(59, 130, 246, 0.1); padding: 12px; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 10px;">
                    <p style="color: #93c5fd; font-weight: bold; margin: 0;">🗣️ Speech Model</p>
                    <p style="color: #e0e0e0; margin: 5px 0; font-size: 0.95em;">{WHISPER_MODEL}</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div style="background: rgba(59, 130, 246, 0.1); padding: 12px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                    <p style="color: #93c5fd; font-weight: bold; margin: 0;">🔊 Detection Threshold</p>
                    <p style="color: #e0e0e0; margin: 5px 0; font-size: 0.95em;">{YAMNET_CONFIDENCE_THRESHOLD}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="animated-container">
                    <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px;">
                        ✨ Features
                    </p>
                </div>
            """, unsafe_allow_html=True)
            emotion_status = "Enabled" if ENABLE_EMOTION_ANALYSIS else "Disabled"
            emotion_color = "#10b981" if ENABLE_EMOTION_ANALYSIS else "#ef4444"
            st.markdown(f"""
                <div style="background: rgba({('16, 185, 129' if ENABLE_EMOTION_ANALYSIS else '239, 68, 68')}, 0.1); padding: 12px; border-radius: 8px; border-left: 4px solid {emotion_color}; margin-bottom: 10px;">
                    <p style="color: {emotion_color}; font-weight: bold; margin: 0;">😊 Emotion Analysis</p>
                    <p style="color: #e0e0e0; margin: 5px 0; font-size: 0.95em;">{emotion_status}</p>
                </div>
            """, unsafe_allow_html=True)
            
            reasoning_status = "Enabled" if ENABLE_REASONING else "Disabled"
            reasoning_color = "#10b981" if ENABLE_REASONING else "#ef4444"
            st.markdown(f"""
                <div style="background: rgba({('16, 185, 129' if ENABLE_REASONING else '239, 68, 68')}, 0.1); padding: 12px; border-radius: 8px; border-left: 4px solid {reasoning_color}; margin-bottom: 10px;">
                    <p style="color: {reasoning_color}; font-weight: bold; margin: 0;">🧠 Reasoning Engine</p>
                    <p style="color: #e0e0e0; margin: 5px 0; font-size: 0.95em;">{reasoning_status}</p>
                </div>
            """, unsafe_allow_html=True)
            
            ollama_ok = True  # Ollama is local; no key to validate
            api_color = "#10b981"
            api_status = "Local (No Key Required)"
            st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.1); padding: 12px; border-radius: 8px; border-left: 4px solid {api_color};">
                    <p style="color: {api_color}; font-weight: bold; margin: 0;">🦙 Ollama LLM</p>
                    <p style="color: #e0e0e0; margin: 5px 0; font-size: 0.95em;">{api_status}</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border: 2px solid rgba(0, 212, 255, 0.3); margin: 30px 0;'>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="animated-container">
                <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin: 0;">
                    📋 Module Information
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        modules_info = {
            'Speech Recognizer': modules['speech_recognizer'].get_model_info(),
            'Sound Detector': modules['sound_detector'].get_model_info(),
            'Emotion Analyzer': modules['emotion_analyzer'].get_model_info(),
            'Inference Engine': modules['inference_engine'].get_model_info()
        }
        
        selected_module = st.selectbox("Select a module to view details:", list(modules_info.keys()), label_visibility="collapsed")
        
        if selected_module:
            st.markdown("""
                <div class="animated-container">
            """, unsafe_allow_html=True)
            st.json(modules_info[selected_module])
            st.markdown("""
                </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 4: DOCUMENTATION
    # ========================================================================
    
    with tab4:
        st.markdown("<h2 style='text-align: center;'>📚 Documentation & Guide</h2>", unsafe_allow_html=True)
        st.markdown("")
        
        # Quick Start
        st.markdown("""
            <div class="animated-container">
                <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin: 0; margin-bottom: 15px;">
                    🚀 Quick Start
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        steps = [
            ("Upload Audio", "Click the upload button on the Process Audio tab"),
            ("Choose Format", "Supports WAV, MP3, M4A, FLAC, OGG (max 50 MB)"),
            ("Review Results", "The system analyzes: speech, sounds, emotion, context, and reasoning")
        ]
        
        for idx, (step, desc) in enumerate(steps, 1):
            st.markdown(f"""
                <div style="display: flex; gap: 15px; margin-bottom: 12px; padding: 12px; background: rgba(124, 58, 237, 0.1); border-radius: 8px; border-left: 3px solid #a78bfa;">
                    <span style="font-size: 1.5em; font-weight: bold; color: #a78bfa; min-width: 30px;">{idx}</span>
                    <div>
                        <p style="color: #a78bfa; font-weight: bold; margin: 0; font-size: 0.95em;">{step}</p>
                        <p style="color: #e0e0e0; margin: 5px 0; font-size: 0.9em;">{desc}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Understanding Output
        st.markdown("""
            <div class="animated-container">
                <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin: 0; margin-bottom: 15px;">
                    📖 Understanding Output
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        outputs = [
            ("Speech Transcript", "Exact text of what was spoken with detected language"),
            ("Sound Events", "Environmental sounds detected and categorized"),
            ("Emotional Tone", "Speaker's emotional state inferred from audio features"),
            ("Integrated Context", "Unified representation combining all audio modalities"),
            ("Scene Reasoning", "AI-generated inference about what's happening")
        ]
        
        for title, desc in outputs:
            st.markdown(f"""
                <div style="padding: 12px; background: rgba(0, 212, 255, 0.05); border-radius: 8px; margin-bottom: 10px; border-left: 3px solid #00d4ff;">
                    <p style="color: #00d4ff; font-weight: bold; margin: 0; font-size: 0.95em;">🔹 {title}</p>
                    <p style="color: #e0e0e0; margin: 5px 0; font-size: 0.9em;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Tips
        st.markdown("""
            <div class="animated-container">
                <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin: 0; margin-bottom: 15px;">
                    💡 Pro Tips
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        tips = [
            ("Clear Audio", "Better results with less background noise"),
            ("Complete Sentences", "Speech recognition works best with full sentences"),
            ("Natural Speech", "Normal speech quality produces better results"),
            ("Multilingual Support", "Supports 20+ languages including English and Hindi")
        ]
        
        cols = st.columns(2)
        for idx, (tip, desc) in enumerate(tips):
            with cols[idx % 2]:
                st.markdown(f"""
                    <div style="padding: 12px; background: rgba(16, 185, 129, 0.1); border-radius: 8px; border-left: 3px solid #10b981;">
                        <p style="color: #10b981; font-weight: bold; margin: 0; font-size: 0.95em;">⚡ {tip}</p>
                        <p style="color: #e0e0e0; margin: 5px 0; font-size: 0.9em;">{desc}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border: 2px solid rgba(0, 212, 255, 0.3); margin: 30px 0;'>", unsafe_allow_html=True)
        
        # Technical Details
        st.markdown("""
            <div class="animated-container">
                <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin: 0; margin-bottom: 15px;">
                    🔧 Technical Details
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        tech_details = [
            ("Sample Rate", "All audio normalized to 16 kHz"),
            ("Maximum Duration", "30 seconds for sound detection"),
            ("Inference Engine", "Uses Ollama (local LLM) for reasoning"),
            ("Processing Time", "Typically 30-60 seconds depending on audio length")
        ]
        
        for detail, value in tech_details:
            st.markdown(f"""
                <div style="padding: 10px; display: flex; justify-content: space-between; align-items: center; background: rgba(30, 30, 50, 0.8); border-radius: 8px; margin-bottom: 8px;">
                    <span style="color: #e0e0e0; font-weight: bold;">{detail}</span>
                    <span style="color: #00d4ff; font-family: monospace;">{value}</span>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Languages
        st.markdown("""
            <div class="animated-container">
                <p style="color: #00d4ff; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin: 0; margin-bottom: 15px;">
                    🌐 Supported Languages
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <p style="color: #e0e0e0; line-height: 1.8;">
                English · Hindi · Spanish · French · German · Chinese · Japanese · Korean · Arabic · Portuguese 
                and 15+ others
            </p>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

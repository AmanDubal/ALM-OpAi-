"""
Audio Language Model (ALM) - Main Application

A Streamlit-based application that integrates multiple audio understanding modules
to perform contextual reasoning over audio scenes.

Architecture:
    Listen → Think → Understand

    1. Audio Input & Preprocessing
    2. Speech Recognition (Whisper)
    3. Sound Event Detection (YAMNet)
    4. Emotion Analysis
    5. Context Integration
    6. Reasoning & Inference (OpenRouter AI API)
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
    APP_TITLE,
    APP_DESCRIPTION,
    PAGE_LAYOUT,
    INITIAL_SIDEBAR_STATE,
    OPENROUTER_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    WHISPER_MODEL,
    YAMNET_CONFIDENCE_THRESHOLD,
    ENABLE_EMOTION_ANALYSIS,
    ENABLE_REASONING,
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

def inject_custom_css():
    custom_css = """
    <style>
        * { margin: 0; padding: 0; }

        .main {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3f 50%, #0f0f23 100%);
            color: #e0e0e0;
        }

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
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

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
            from { opacity: 0; transform: translateY(30px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        .pulse-glow { animation: pulseGlow 2s ease-in-out infinite; }

        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 10px rgba(0, 212, 255, 0.3); }
            50%       { box-shadow: 0 0 20px rgba(0, 212, 255, 0.6); }
        }

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

        .stMetric {
            background: rgba(20, 20, 40, 0.8) !important;
            border: 2px solid rgba(0, 212, 255, 0.2) !important;
            padding: 20px !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: rgba(30, 30, 50, 0.5);
            padding: 10px;
            border-radius: 10px;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


inject_custom_css()

# ============================================================================
# MODULE INITIALISATION
# ============================================================================

@st.cache_resource
def initialize_modules():
    """Initialise all processing modules (cached across reruns)."""
    engine = InferenceEngine()
    # Set up OpenRouter — reads OPENROUTER_API_KEY from env / config
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


# ============================================================================
# UI HELPERS
# ============================================================================

def render_header():
    col1, col2, col3 = st.columns([0.5, 2, 0.5])
    with col2:
        st.markdown("""
            <div style="text-align:center;padding:30px 0;">
                <h1 style="font-size:3.5em;margin:0;letter-spacing:2px;">🎵 ALM</h1>
                <p style="font-size:1.3em;color:#00d4ff;margin:10px 0;letter-spacing:1px;">
                    AUDIO LANGUAGE MODEL
                </p>
                <p style="color:#a0a0c0;font-size:1em;margin-top:10px;">
                    🧠 Powered by OpenRouter AI — Understand Audio with AI Intelligence
                </p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("<hr style='border:2px solid rgba(0,212,255,0.3);'>", unsafe_allow_html=True)


def render_progress_step(step_num, total_steps, title, emoji):
    progress = step_num / total_steps
    st.markdown(f"""
        <div class="animated-container pulse-glow">
            <div style="display:flex;align-items:center;gap:15px;">
                <span style="font-size:2em;">{emoji}</span>
                <div style="flex:1;">
                    <p style="color:#00d4ff;font-weight:bold;margin:0;text-transform:uppercase;letter-spacing:0.5px;">
                        Step {step_num}/{total_steps} · {title}
                    </p>
                    <div style="background:rgba(0,212,255,0.1);height:4px;border-radius:2px;margin-top:8px;overflow:hidden;">
                        <div style="background:linear-gradient(90deg,#00d4ff,#7c3aed);height:100%;width:{progress*100:.0f}%;border-radius:2px;"></div>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_success_badge(title):
    st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:15px;
             background:rgba(16,185,129,0.1);border-left:4px solid #10b981;
             border-radius:8px;margin:10px 0;">
            <span style="font-size:1.5em;">✅</span>
            <span style="color:#10b981;font-weight:bold;text-transform:uppercase;letter-spacing:0.5px;">{title}</span>
        </div>
    """, unsafe_allow_html=True)


def render_metric_grid(metrics):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label, value)


# ============================================================================
# MAIN
# ============================================================================

def main():
    render_header()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎤 PROCESS AUDIO", "📊 SYSTEM ARCHITECTURE", "⚙️ SETTINGS", "📚 GUIDE"]
    )

    modules = initialize_modules()

    # ========================================================================
    # TAB 1: PROCESS AUDIO
    # ========================================================================
    with tab1:
        st.markdown("<h2 style='text-align:center;'>🎵 Audio Processing Pipeline</h2>", unsafe_allow_html=True)
        st.markdown("")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
                <div class="animated-container" style="padding:25px;">
                    <p style="color:#00d4ff;font-weight:bold;text-transform:uppercase;letter-spacing:1px;margin-bottom:15px;">
                        📥 Upload Your Audio File
                    </p>
                </div>
            """, unsafe_allow_html=True)
            audio_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
                label_visibility="collapsed",
            )
        with col2:
            st.markdown("""
                <div style="background:rgba(0,212,255,0.1);padding:15px;border-radius:8px;
                     border-left:4px solid #00d4ff;text-align:center;">
                    <p style="color:#00d4ff;font-weight:bold;font-size:0.9em;margin:0;">MAX SIZE</p>
                    <p style="color:#e0e0e0;font-weight:bold;font-size:1.2em;margin:5px 0;">50 MB</p>
                </div>
            """, unsafe_allow_html=True)

        if audio_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_path = tmp_file.name

            try:
                progress_ph = st.empty()
                results_ph = st.empty()

                # ── Step 1: Preprocess ───────────────────────────────────────
                with progress_ph.container():
                    render_progress_step(1, 6, "Loading & Preprocessing Audio", "📥")

                audio, sr = modules['preprocessor'].load_audio(tmp_path)
                audio_duration = modules['preprocessor'].get_audio_duration(audio)
                audio_info = modules['preprocessor'].get_audio_info(audio)

                with results_ph.container():
                    render_success_badge("Audio Loaded Successfully")
                    render_metric_grid({
                        "⏱️ Duration": f"{audio_duration:.2f}s",
                        "🔊 Sample Rate": f"{audio_info['sample_rate']} Hz",
                        "📈 Peak Amplitude": f"{audio_info['peak_amplitude']:.3f}",
                    })

                # ── Step 2: Speech Recognition ───────────────────────────────
                with progress_ph.container():
                    render_progress_step(2, 6, "Performing Speech Recognition", "🗣️")

                speech_result = modules['speech_recognizer'].transcribe(audio, sr=sr)
                transcript = speech_result.get('text', '')
                speech_language = speech_result.get('language', 'unknown')

                with results_ph.container():
                    render_success_badge("Speech Recognition Complete")
                    if transcript:
                        st.markdown("""
                            <div class="animated-container" style="margin-top:15px;">
                                <p style="color:#00d4ff;font-weight:bold;text-transform:uppercase;
                                   letter-spacing:0.5px;margin-bottom:10px;">📝 Transcribed Text</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.text_area("Transcript", value=transcript, height=100,
                                     disabled=True, label_visibility="collapsed")
                        st.caption(f"🌐 Detected Language: **{speech_language.upper()}**")
                    else:
                        st.warning("⚠️ No speech detected in audio")

                # ── Step 3: Sound Detection ──────────────────────────────────
                with progress_ph.container():
                    render_progress_step(3, 6, "Detecting Sound Events", "🔊")

                sound_result = modules['sound_detector'].detect(audio, sr=sr)
                sounds_detected = sound_result.get('events', [])

                with results_ph.container():
                    render_success_badge("Sound Detection Complete")
                    if sounds_detected:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("""
                                <div class="animated-container">
                                    <p style="color:#00d4ff;font-weight:bold;text-transform:uppercase;
                                       letter-spacing:0.5px;">🔊 Detected Sound Events</p>
                                </div>
                            """, unsafe_allow_html=True)
                            for sound in sounds_detected:
                                st.markdown(f"• **{sound}**")
                        with col_b:
                            st.markdown("""
                                <div class="animated-container">
                                    <p style="color:#00d4ff;font-weight:bold;text-transform:uppercase;
                                       letter-spacing:0.5px;">📊 Categorized Events</p>
                                </div>
                            """, unsafe_allow_html=True)
                            categorized = modules['sound_detector'].categorize_events(sounds_detected)
                            for category, events in categorized.items():
                                if events:
                                    st.markdown(f"**{category.replace('_', ' ').title()}:** {len(events)}")
                    else:
                        st.info("ℹ️ No significant sound events detected")

                # ── Step 4: Emotion Analysis ─────────────────────────────────
                emotion_detected = "neutral"
                if ENABLE_EMOTION_ANALYSIS:
                    with progress_ph.container():
                        render_progress_step(4, 6, "Analyzing Emotional Tone", "😊")

                    emotion_result = modules['emotion_analyzer'].analyze(audio, sr=sr)
                    emotion_detected = emotion_result.get('emotion', 'unknown')
                    emotion_confidence = emotion_result.get('confidence', 0.0)

                    with results_ph.container():
                        render_success_badge("Emotion Analysis Complete")
                        render_metric_grid({
                            "😊 Emotion": emotion_detected.upper(),
                            "🎯 Confidence": f"{emotion_confidence:.1%}",
                        })

                # ── Step 5: Context Integration ──────────────────────────────
                with progress_ph.container():
                    render_progress_step(5, 6, "Integrating Audio Context", "🔗")

                context, formatted_context = modules['context_integrator'].integrate(
                    transcript=transcript,
                    sound_events=sounds_detected,
                    emotion=emotion_detected,
                )

                with results_ph.container():
                    render_success_badge("Context Integration Complete")
                    st.markdown("""
                        <div class="animated-container" style="margin-top:15px;">
                            <p style="color:#00d4ff;font-weight:bold;text-transform:uppercase;
                               letter-spacing:0.5px;margin-bottom:10px;">🔗 Integrated Context</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.text_area("Context", value=formatted_context, height=250,
                                 disabled=True, label_visibility="collapsed")

                # ── Step 6: OpenRouter Reasoning ─────────────────────────────
                inference_text = ""
                if ENABLE_REASONING:
                    with progress_ph.container():
                        render_progress_step(6, 6, "Generating AI Inference via OpenRouter", "🧠")

                    # Layer 2: Semantic risk analysis
                    risk_analysis = modules['semantic_analyzer'].analyze(transcript)

                    # Merge all layers
                    merged_context = modules['data_adapter'].merge_all_analysis(
                        audio_analysis=None,
                        speech_data={
                            'transcript': transcript,
                            'language': speech_language,
                            'confidence': 1.0,
                        },
                        sound_events=[
                            {'event': s, 'confidence': 0.7} for s in sounds_detected
                        ],
                        emotion_data={
                            'emotional_state': emotion_detected,
                            'confidence': 0.7,
                            'vocal_tension': 'unknown',
                        },
                        risk_analysis=risk_analysis,
                    )

                    # OpenRouter inference
                    inference_text = modules['inference_engine'].generate_inference_v2(
                        context=merged_context,
                        risk_analysis=risk_analysis,
                    )

                    with results_ph.container():
                        render_success_badge("OpenRouter AI Reasoning Complete")
                        st.markdown("""
                            <div class="animated-container" style="margin-top:15px;
                                 background:linear-gradient(135deg,rgba(124,58,237,0.1),rgba(0,212,255,0.1));">
                                <p style="color:#a78bfa;font-weight:bold;text-transform:uppercase;
                                   letter-spacing:0.5px;margin-bottom:15px;">
                                    🧠 Two-Layer AI Reasoning (Acoustic + Semantic) via OpenRouter
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.text(inference_text)
                else:
                    with progress_ph.container():
                        st.info("ℹ️ Reasoning disabled in settings.")

                progress_ph.empty()

                # ── Download ─────────────────────────────────────────────────
                st.markdown("<hr style='border:2px solid rgba(0,212,255,0.3);margin:30px 0;'>",
                            unsafe_allow_html=True)

                results_text = f"""
╔═══════════════════════════════════════════════════════════════╗
║        AUDIO LANGUAGE MODEL (ALM) — ANALYSIS RESULTS         ║
║           Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                    ║
╚═══════════════════════════════════════════════════════════════╝

1. AUDIO INFORMATION
   Duration        : {audio_duration:.2f} seconds
   Sample Rate     : {audio_info['sample_rate']} Hz
   Peak Amplitude  : {audio_info['peak_amplitude']:.3f}

2. SPEECH RECOGNITION
   Transcript      : {transcript if transcript else 'No speech detected'}
   Language        : {speech_language}

3. SOUND EVENTS
   Detected        : {', '.join(sounds_detected) if sounds_detected else 'None'}

4. EMOTION ANALYSIS
   Emotion         : {emotion_detected}

5. INTEGRATED CONTEXT
{formatted_context}

6. AI INFERENCE (OpenRouter)
{inference_text if ENABLE_REASONING else 'Reasoning disabled'}

════════════════════════════════════════════════════════════════
"""
                st.download_button(
                    label="📄 Download Analysis Results (TXT)",
                    data=results_text,
                    file_name=f"alm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # ========================================================================
    # TAB 2: ARCHITECTURE
    # ========================================================================
    with tab2:
        st.markdown("<h2 style='text-align:center;'>🏗️ System Architecture</h2>", unsafe_allow_html=True)
        st.markdown("")

        st.markdown("""
            <div class="animated-container">
                <h3 style="color:#00d4ff;margin-top:0;">🎯 Listen → Think → Understand</h3>
                <p style="color:#e0e0e0;line-height:1.8;">
                    ALM processes audio through an intelligent pipeline of specialised modules.
                    Reasoning is powered by <strong>OpenRouter AI API</strong>, giving access to
                    top-tier LLMs (GPT-4 Turbo, Claude, Mistral, etc.) through a single endpoint.
                </p>
            </div>
        """, unsafe_allow_html=True)

        modules_info = [
            ("📥", "Audio Input & Preprocessing", "Standardises audio (mono, 16 kHz) and normalises amplitude", "Librosa, Pydub"),
            ("🗣️", "Speech Recognition", "Extracts spoken language with multilingual support", "OpenAI Whisper"),
            ("🔊", "Sound Event Detection", "Identifies environmental sounds (traffic, crowd, alarms, machinery)", "YAMNet CNN"),
            ("😊", "Emotion Analysis", "Captures emotional cues from MFCC audio features", "MFCC Feature Extraction"),
            ("🔗", "Context Integration ⭐", "Merges all outputs into unified context representation (Core Innovation)", "Custom Integration Engine"),
            ("🧠", "Reasoning & Inference", "Structured LLM reasoning via OpenRouter AI API", "OpenRouter (GPT-4 Turbo / any model)"),
        ]

        cols = st.columns(2)
        for idx, (emoji, title, desc, tools) in enumerate(modules_info):
            with cols[idx % 2]:
                st.markdown(f"""
                    <div class="animated-container" style="margin-bottom:15px;">
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                            <span style="font-size:2em;">{emoji}</span>
                            <p style="color:#00d4ff;font-weight:bold;margin:0;text-transform:uppercase;
                               letter-spacing:0.5px;font-size:0.95em;">{title}</p>
                        </div>
                        <p style="color:#e0e0e0;margin:10px 0;font-size:0.95em;line-height:1.6;">{desc}</p>
                        <p style="color:#a0a0c0;margin:0;font-size:0.85em;">🛠️ <strong>{tools}</strong></p>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("""
            <div class="animated-container" style="margin-top:20px;">
                <p style="color:#00d4ff;font-weight:bold;text-transform:uppercase;letter-spacing:1px;margin-bottom:15px;">
                    🔌 OpenRouter Integration
                </p>
                <p style="color:#e0e0e0;line-height:1.8;">
                    OpenRouter provides a single <strong>OpenAI-compatible API endpoint</strong> that routes
                    requests to dozens of LLMs including GPT-4 Turbo, Claude 3, Mistral, LLaMA 3, and more.
                    Set <code>OPENROUTER_MODEL</code> in your <code>.env</code> file to switch models instantly.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # ========================================================================
    # TAB 3: SETTINGS
    # ========================================================================
    with tab3:
        st.markdown("<h2 style='text-align:center;'>⚙️ System Settings</h2>", unsafe_allow_html=True)
        st.markdown("")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
                <div class="animated-container">
                    <p style="color:#00d4ff;font-weight:bold;text-transform:uppercase;margin-bottom:15px;">
                        🤖 Model Configuration
                    </p>
                </div>
            """, unsafe_allow_html=True)

            for label, value in [
                ("🗣️ Whisper Model", WHISPER_MODEL),
                ("🔊 YAMNet Threshold", str(YAMNET_CONFIDENCE_THRESHOLD)),
                ("🧠 OpenRouter Model", OPENROUTER_MODEL),
                ("🌐 OpenRouter Base URL", OPENROUTER_BASE_URL),
            ]:
                st.markdown(f"""
                    <div style="background:rgba(59,130,246,0.1);padding:12px;border-radius:8px;
                         border-left:4px solid #3b82f6;margin-bottom:10px;">
                        <p style="color:#93c5fd;font-weight:bold;margin:0;">{label}</p>
                        <p style="color:#e0e0e0;margin:5px 0;font-size:0.95em;font-family:monospace;">{value}</p>
                    </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div class="animated-container">
                    <p style="color:#00d4ff;font-weight:bold;text-transform:uppercase;margin-bottom:15px;">
                        ✨ Feature Flags
                    </p>
                </div>
            """, unsafe_allow_html=True)

            for label, enabled in [
                ("😊 Emotion Analysis", ENABLE_EMOTION_ANALYSIS),
                ("🧠 Reasoning Engine", ENABLE_REASONING),
            ]:
                color = "#10b981" if enabled else "#ef4444"
                status = "Enabled" if enabled else "Disabled"
                rgb = "16,185,129" if enabled else "239,68,68"
                st.markdown(f"""
                    <div style="background:rgba({rgb},0.1);padding:12px;border-radius:8px;
                         border-left:4px solid {color};margin-bottom:10px;">
                        <p style="color:{color};font-weight:bold;margin:0;">{label}</p>
                        <p style="color:#e0e0e0;margin:5px 0;">{status}</p>
                    </div>
                """, unsafe_allow_html=True)

            # API key status
            key_set = bool(OPENROUTER_API_KEY)
            key_color = "#10b981" if key_set else "#ef4444"
            key_status = "✅ API Key Configured" if key_set else "❌ API Key Not Set (set OPENROUTER_API_KEY)"
            st.markdown(f"""
                <div style="background:rgba({'16,185,129' if key_set else '239,68,68'},0.1);padding:12px;
                     border-radius:8px;border-left:4px solid {key_color};margin-top:10px;">
                    <p style="color:{key_color};font-weight:bold;margin:0;">🔑 OpenRouter API Key</p>
                    <p style="color:#e0e0e0;margin:5px 0;font-size:0.9em;">{key_status}</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border:2px solid rgba(0,212,255,0.3);margin:30px 0;'>", unsafe_allow_html=True)

        st.markdown("""
            <div class="animated-container">
                <p style="color:#00d4ff;font-weight:bold;text-transform:uppercase;letter-spacing:1px;">
                    📋 Module Information
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        module_info_map = {
            'Speech Recognizer': modules['speech_recognizer'].get_model_info(),
            'Sound Detector': modules['sound_detector'].get_model_info(),
            'Emotion Analyzer': modules['emotion_analyzer'].get_model_info(),
            'Inference Engine': modules['inference_engine'].get_model_info(),
        }
        selected = st.selectbox("Select a module:", list(module_info_map.keys()),
                                label_visibility="collapsed")
        if selected:
            st.json(module_info_map[selected])

    # ========================================================================
    # TAB 4: GUIDE
    # ========================================================================
    with tab4:
        st.markdown("<h2 style='text-align:center;'>📚 Documentation & Guide</h2>", unsafe_allow_html=True)
        st.markdown("")

        # OpenRouter setup
        st.markdown("""
            <div class="animated-container">
                <p style="color:#00d4ff;font-weight:bold;text-transform:uppercase;letter-spacing:1px;margin-bottom:15px;">
                    🔌 OpenRouter Setup
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.code("""# .env file
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-4-turbo   # or any model on openrouter.ai

# Popular model options:
# openai/gpt-4-turbo
# openai/gpt-4o
# anthropic/claude-3-opus
# mistralai/mistral-7b-instruct
# meta-llama/llama-3-70b-instruct
""", language="bash")

        st.markdown("""
            <div class="animated-container" style="margin-top:15px;">
                <p style="color:#00d4ff;font-weight:bold;text-transform:uppercase;letter-spacing:1px;margin-bottom:15px;">
                    🚀 Quick Start
                </p>
            </div>
        """, unsafe_allow_html=True)

        for idx, (step, desc) in enumerate([
            ("Set API Key", "Add OPENROUTER_API_KEY to your .env file"),
            ("Upload Audio", "Click the upload button on the Process Audio tab"),
            ("Review Results", "System analyzes speech, sounds, emotion, context, and generates LLM reasoning"),
        ], 1):
            st.markdown(f"""
                <div style="display:flex;gap:15px;margin-bottom:12px;padding:12px;
                     background:rgba(124,58,237,0.1);border-radius:8px;border-left:3px solid #a78bfa;">
                    <span style="font-size:1.5em;font-weight:bold;color:#a78bfa;min-width:30px;">{idx}</span>
                    <div>
                        <p style="color:#a78bfa;font-weight:bold;margin:0;">{step}</p>
                        <p style="color:#e0e0e0;margin:5px 0;font-size:0.9em;">{desc}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <div class="animated-container" style="margin-top:20px;">
                <p style="color:#00d4ff;font-weight:bold;text-transform:uppercase;letter-spacing:1px;margin-bottom:15px;">
                    💡 Pro Tips
                </p>
            </div>
        """, unsafe_allow_html=True)

        tips = [
            ("Clear Audio", "Less background noise = better transcription"),
            ("Model Choice", "GPT-4 Turbo gives best reasoning; Mistral-7B is fastest/cheapest"),
            ("Multilingual", "Whisper supports 20+ languages automatically"),
            ("Fallback Mode", "If OpenRouter is unavailable, deterministic fallback activates automatically"),
        ]

        tip_cols = st.columns(2)
        for idx, (tip, desc) in enumerate(tips):
            with tip_cols[idx % 2]:
                st.markdown(f"""
                    <div style="padding:12px;background:rgba(16,185,129,0.1);border-radius:8px;
                         border-left:3px solid #10b981;margin-bottom:10px;">
                        <p style="color:#10b981;font-weight:bold;margin:0;">⚡ {tip}</p>
                        <p style="color:#e0e0e0;margin:5px 0;font-size:0.9em;">{desc}</p>
                    </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

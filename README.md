# 🎵 ALM 4.0 — Audio Language Model (Ollama Edition)

> **Understand audio with AI intelligence** — a fully local, privacy-first audio analysis system powered by OpenAI Whisper, YAMNet, and a local Ollama LLM.

---

## Overview

ALM 4.0 is a Streamlit-based application that processes audio files through a **six-stage pipeline**, combining acoustic analysis, speech recognition, emotion detection, and local LLM reasoning to produce structured, human-readable scene reports — all without sending data to any external API.

---

## ✨ Features

- 🗣️ **Speech Recognition** — Multilingual transcription via OpenAI Whisper (supports Hindi, English, and more)
- 🔊 **Sound Event Detection** — Environmental sound identification using Google's YAMNet CNN
- 😊 **Emotion Analysis** — Vocal emotion classification using MFCC feature extraction
- 🧠 **Two-Layer AI Reasoning**
  - **Layer 1 — Acoustic Analysis:** Spectral features, noise profiling, reverberation estimation
  - **Layer 2 — Semantic Context:** Emergency keyword detection, risk scoring, dispatch signal identification
- 🦙 **Local LLM via Ollama** — Structured reasoning using `phi` (or any Ollama model), no internet required
- 📥 **Downloadable Reports** — Full analysis exported as a `.txt` file

---

## 🏗️ Architecture

```
Audio File
    │
    ▼
┌─────────────────────┐
│ 1. Preprocessing    │  Librosa / Pydub — mono 16 kHz normalisation
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 2. Speech Recognit. │  OpenAI Whisper (base model)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 3. Sound Detection  │  YAMNet — top-5 environmental events
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 4. Emotion Analysis │  MFCC-based vocal emotion classifier
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 5. Context Integr.  │  Merge transcript + sounds + emotion
└────────┬────────────┘
         ▼
┌──────────────────────────────────────────────┐
│ 6. Two-Layer Reasoning (Ollama LLM)          │
│                                              │
│  Layer 1: Acoustic scene understanding       │
│  Layer 2: Semantic risk & emergency scoring  │
│                                              │
│  Output format:                              │
│   📍 Location                                │
│   👥 Number of People Talking                │
│   🎯 Activity                                │
│   😊 Emotional Tone                          │
│   ⚠  Risk Level                              │
│   📊 Confidence                              │
│   📝 Summary                                 │
└──────────────────────────────────────────────┘
```

---

## 📦 Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally
- A pulled Ollama model (default: `phi`)

### Python Dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `librosa`, `pydub`, `scipy` | Audio loading & preprocessing |
| `openai-whisper` | Speech recognition |
| `tensorflow`, `tensorflow-hub` | YAMNet sound detection |
| `ollama` | Local LLM client |
| `streamlit` | Web UI |
| `python-dotenv` | Environment variable loading |
| `numpy`, `matplotlib`, `plotly` | Numerics & visualization |

---

## 🚀 Quick Start

### 1. Install Ollama and pull a model

```bash
# Install Ollama from https://ollama.com
ollama pull phi
```

### 2. Clone and set up the project

```bash
git clone <repo-url>
cd ALM4.0
pip install -r requirements.txt
```

### 3. Configure environment (optional)

Copy `.env.example` to `.env` and edit as needed:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi
DEBUG_MODE=False
```

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚙️ Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `phi` | Local Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `WHISPER_MODEL` | `base` | Whisper model size (`tiny`/`base`/`small`/`medium`/`large`) |
| `YAMNET_CONFIDENCE_THRESHOLD` | `0.3` | Minimum confidence for sound event detection |
| `ENABLE_EMOTION_ANALYSIS` | `True` | Toggle emotion analysis |
| `ENABLE_REASONING` | `True` | Toggle LLM reasoning |
| `MAX_UPLOAD_SIZE` | `50 MB` | Maximum audio file size |

### LLM Parameters (in `inference_engine.py`)

```python
options={'temperature': 0.2, 'num_predict': 400}
```

Low temperature is intentional — keeps reasoning deterministic and evidence-based.

---

## 📁 Project Structure

```
ALM4.0(Ollama)/
├── app.py                  # Main Streamlit application
├── config.py               # All configuration settings
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (local, not committed)
└── modules/
    ├── preprocessing/      # Audio loading & normalisation
    ├── speech/             # Whisper speech recognizer
    ├── sound_detection/    # YAMNet sound event detector
    ├── emotion/            # MFCC-based emotion analyser
    ├── context/            # Context integrator + semantic analyser
    └── reasoning/          # InferenceEngine (Ollama + fallback)
```

---

## 🛡️ Emergency Audio Mode

The LLM system prompt is configured specifically for **emergency audio reasoning**:

- Does **not** guess without evidence
- Marks location as *"Location unclear"* when insufficient evidence exists
- Distinguishes emotional tones of different speakers
- Uses detected keywords as explicit evidence
- Provides structured, professional output in every response

The **Layer 2 semantic analyser** independently scores risk (0–1) using keyword banks for:
- Medical emergencies (`breathing`, `hurt`, `ambulance`, …)
- Conflict situations (`fight`, `gun`, `weapon`, …)
- Vulnerability signals (`child`, `baby`, `elderly`, …)
- Dispatch/guidance phrases (`call 911`, `stay on the phone`, …)

---

## 🔄 Fallback Behaviour

If Ollama is unreachable or the model returns an empty response, the system automatically switches to a **deterministic fallback** that uses the Layer 2 semantic scores to generate a structured report — no crash, no silent failure.

---

## 📄 Supported Audio Formats

`WAV` · `MP3` · `M4A` · `FLAC` · `OGG`

---

## 📝 License

This project is for academic and research purposes.

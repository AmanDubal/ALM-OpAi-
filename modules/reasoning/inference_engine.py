"""
Module 6: Audio Environment Analysis & Reasoning

Reasoning is performed via OpenRouter AI API (OpenAI-compatible interface).
Falls back to a deterministic rule-based report when the API is unavailable.
"""

import os
import json
import warnings
import re
import librosa
import numpy as np
import scipy.signal as signal
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from openai import OpenAI

warnings.filterwarnings('ignore')


# ── Noise taxonomy ────────────────────────────────────────────────────────────

class NoiseCategory(Enum):
    AMBIENT = "ambient_background"
    MACHINERY = "industrial_mechanical"
    TRAFFIC = "transportation"
    SPEECH = "human_vocal"
    ENVIRONMENTAL = "natural_elements"
    STRUCTURAL = "building_related"
    UNIDENTIFIED = "unknown_source"


@dataclass
class NoiseSignature:
    category: NoiseCategory
    frequency_range: Tuple[float, float]
    temporal_pattern: str
    intensity_level: float
    confidence: float
    harmonic_content: List[float]
    spectral_shape: str


@dataclass
class AudioEnvironmentProfile:
    dominant_noises: List[NoiseSignature]
    background_noise_floor: float
    signal_to_noise_ratio: float
    acoustic_complexity: float
    spatial_characteristics: Dict[str, Any]
    environmental_context: str
    risk_factors: List[str]
    quality_assessment: Dict[str, float]


# ── Main engine ───────────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Two-layer audio reasoning system.
    Layer 1 — acoustic analysis (spectral / temporal features).
    Layer 2 — LLM reasoning via OpenRouter AI API.
    """

    def __init__(self, sr: int = 16000, n_fft: int = 2048, hop_length: int = 512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        self.noise_signatures = self._initialize_noise_database()

        self._client: Optional[OpenAI] = None
        self._model: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4-turbo")
        self._base_url: str = "https://openrouter.ai/api/v1"

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup_openrouter(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialise the OpenRouter client."""
        key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            warnings.warn("OPENROUTER_API_KEY not set — fallback mode will be used.")
        if base_url:
            self._base_url = base_url
        if model:
            self._model = model
        self._client = OpenAI(api_key=key, base_url=self._base_url)

    # ── Public interfaces ─────────────────────────────────────────────────────

    def analyze(self, audio: np.ndarray, sr: int = None) -> Dict[str, Any]:
        """Acoustic environment analysis (Layer 1). Returns profile + report + json_export."""
        if sr is None:
            sr = self.sr
        profile = self.analyze_audio_signal(audio, sr)
        report = AudioAnalysisReporter.generate_report(profile)
        json_export = {
            'environmental_context': profile.environmental_context,
            'dominant_noises': [
                {
                    'category': n.category.value,
                    'frequency_range': n.frequency_range,
                    'temporal_pattern': n.temporal_pattern,
                    'intensity': n.intensity_level,
                    'confidence': n.confidence,
                }
                for n in profile.dominant_noises
            ],
            'acoustic_measurements': {
                'background_noise_floor_db': profile.background_noise_floor,
                'signal_to_noise_ratio_db': profile.signal_to_noise_ratio,
                'acoustic_complexity': profile.acoustic_complexity,
            },
            'spatial_characteristics': profile.spatial_characteristics,
            'quality_assessment': profile.quality_assessment,
            'risk_factors': profile.risk_factors,
        }
        return {'profile': profile, 'report': report, 'json_export': json_export}

    def generate_inference_v2(
        self,
        context: Dict[str, Any],
        risk_analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Primary reasoning entry-point.
        Calls OpenRouter LLM; falls back to deterministic output on failure.
        """
        if risk_analysis is None:
            risk_analysis = self._default_risk()
        if self._client is None:
            self.setup_openrouter()
        result = self._reason_with_openrouter(context, risk_analysis)
        return self._format_report(result, risk_analysis)

    # legacy wrapper kept for compatibility
    def generate_inference(self, context: Dict) -> str:
        speech_data = {'text': context.get('speech', ''), 'language_detected': context.get('language_detected', 'unknown')}
        sounds_raw = context.get('sounds', [])
        sound_data = [{'event': s, 'confidence': 0.7} if isinstance(s, str) else s for s in sounds_raw]
        emotion_data = {'emotional_state': context.get('emotion', 'Neutral'), 'vocal_tension': context.get('vocal_tension', 'Unknown')}
        result = self.perform_reasoning(speech_data, sound_data, emotion_data)
        return self._format_inference_output(result)

    # ── OpenRouter call ───────────────────────────────────────────────────────

    def _reason_with_openrouter(self, context: Dict, risk_analysis: Dict) -> Dict:
        try:
            prompt = self._build_prompt(context, risk_analysis)
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an emergency audio reasoning AI.\n"
                            "Analyze transcript, sound events, emotions, and risk signals.\n"
                            "Be factual and evidence-based. Do not hallucinate.\n"
                            "Return valid JSON only — no markdown fences, no extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=700,
            )
            raw = response.choices[0].message.content or ""
            return self._parse_llm_json(raw, risk_analysis)
        except Exception as exc:
            warnings.warn(f"OpenRouter call failed: {exc}")
            return self._fallback(context, risk_analysis, reason=str(exc))

    def _build_prompt(self, context: Dict, risk_analysis: Dict) -> str:
        # Extract speech
        sp = context.get("speech", {})
        transcript = sp.get("transcript", "No speech detected") if isinstance(sp, dict) else str(sp)
        language = sp.get("language", "unknown") if isinstance(sp, dict) else "unknown"

        # Extract emotion
        em = context.get("emotion_prediction", {})
        emotion = em.get("emotional_state", "neutral") if isinstance(em, dict) else str(em)

        # Extract sounds
        se = context.get("sound_events", {})
        events = se.get("events", []) if isinstance(se, dict) else []
        labels = [e.get("event", str(e)) if isinstance(e, dict) else str(e) for e in events[:5]]

        # Acoustic environment
        env = context.get("environment_prediction", {})
        env_ctx = env.get("context", "Unknown") if isinstance(env, dict) else "Unknown"

        return f"""Analyze this audio situation carefully.

TRANSCRIPT ({language}):
{transcript}

EMOTION:
{emotion}

DETECTED SOUNDS:
{", ".join(labels) if labels else "None"}

ACOUSTIC ENVIRONMENT:
{env_ctx}

RISK ANALYSIS:
- Level    : {risk_analysis.get("risk_level", "low")}
- Score    : {risk_analysis.get("risk_score", 0.0):.2f}
- Situation: {risk_analysis.get("situation_type", "normal_conversation")}
- Keywords : {", ".join(risk_analysis.get("keywords_detected", [])) or "None"}
- Signals  : {", ".join(risk_analysis.get("signals_detected", [])) or "None"}

Return STRICT JSON (no markdown):
{{
    "location": "...",
    "people": "...",
    "activity": "...",
    "emotion": "...",
    "risk_level": "low|moderate|high",
    "confidence": 0.0,
    "explanation": "1-3 sentence evidence-based summary."
}}
"""

    def _parse_llm_json(self, raw: str, risk_analysis: Dict) -> Dict:
        try:
            cleaned = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
            parsed = json.loads(cleaned)
            parsed["source"] = "openrouter_llm"
            return parsed
        except Exception:
            return self._fallback({}, risk_analysis, reason="Failed to parse LLM JSON")

    def _fallback(self, context: Dict, risk_analysis: Dict, reason: str = "") -> Dict:
        risk_level = risk_analysis.get("risk_level", "low")
        situation = risk_analysis.get("situation_type", "normal_conversation")
        keywords = risk_analysis.get("keywords_detected", [])
        activity_map = {
            "emergency": "Emergency situation detected",
            "medical": "Medical emergency indicated",
            "conflict": "Conflict or altercation detected",
            "public_event": "Public gathering or event",
            "normal_conversation": "Routine conversation",
        }
        return {
            "location": "Location unclear — insufficient evidence.",
            "people": "Unknown",
            "activity": activity_map.get(situation, "Unknown activity"),
            "emotion": "Unknown",
            "risk_level": risk_level,
            "confidence": max(0.1, risk_analysis.get("risk_score", 0.1)),
            "explanation": (
                f"Deterministic fallback ({reason}). "
                f"Risk '{risk_level}' from keyword analysis. "
                f"Indicators: {', '.join(keywords[:3]) or 'none'}."
            ),
            "source": "fallback_deterministic",
            "fallback_reason": reason,
        }

    def _default_risk(self) -> Dict:
        return {
            "risk_level": "low", "risk_score": 0.0,
            "situation_type": "normal_conversation",
            "keywords_detected": [], "signals_detected": [],
            "confidence": 0.0, "reasoning": "",
        }

    # ── Report formatter ──────────────────────────────────────────────────────

    def _format_report(self, r: Dict, risk_analysis: Dict) -> str:
        source = r.get("source", "unknown")
        source_label = "AI Reasoning (OpenRouter)" if source == "openrouter_llm" else "Deterministic Fallback"
        try:
            conf_pct = f"{float(r.get('confidence', 0)) * 100:.0f}%"
        except (TypeError, ValueError):
            conf_pct = "N/A"

        lines = [
            "=" * 72,
            "AUDIO LANGUAGE MODEL — REASONING REPORT",
            "=" * 72,
            f"📊 Analysis Source: {source_label}",
            "",
            "📍 LOCATION:",
            f"   {r.get('location', 'Unknown')}",
            "",
            "👥 NUMBER OF PEOPLE:",
            f"   {r.get('people', 'Unknown')}",
            "",
            "🎯 ACTIVITY:",
            f"   {r.get('activity', 'Unknown')}",
            "",
            "😊 EMOTIONAL TONE:",
            f"   {r.get('emotion', 'Unknown')}",
            "",
            "⚠️  RISK ASSESSMENT (Layer 2 — Semantic Context):",
            f"   Risk Level : {risk_analysis.get('risk_level', 'low').upper()}",
            f"   Risk Score : {risk_analysis.get('risk_score', 0.0):.2f}",
            f"   Situation  : {risk_analysis.get('situation_type', 'normal_conversation')}",
        ]

        kws = risk_analysis.get("keywords_detected", [])
        if kws:
            lines += ["", "🔍 KEY INDICATORS:"] + [f"   • {k}" for k in kws[:6]]

        sigs = risk_analysis.get("signals_detected", [])
        if sigs:
            lines += ["", "📡 DISPATCH SIGNALS:"] + [f"   • {s}" for s in sigs[:4]]

        lines += [
            "",
            f"✅ CONFIDENCE SCORE: {conf_pct}",
            "",
            "📋 SUMMARY:",
            f"   {r.get('explanation', 'No summary available.')}",
            "",
            "=" * 72,
        ]

        if r.get("fallback_reason"):
            lines.append(f"ℹ️  Fallback reason: {r['fallback_reason']}")

        return "\n".join(lines)

    # ── Legacy 4-stage pipeline ───────────────────────────────────────────────

    def perform_reasoning(self, speech_data, sound_data, emotion_data) -> Dict:
        return {
            'plan': "Synthesize verbal content with acoustic environment for context understanding.",
            'caption': self._stage_captioning(speech_data, sound_data, emotion_data),
            'reasoning_steps': self._stage_reasoning(speech_data, sound_data, emotion_data),
            'final_summary': self._stage_summarization(speech_data, sound_data, emotion_data, []),
        }

    def _stage_captioning(self, speech_data, sound_data, emotion_data) -> str:
        text = speech_data.get('text', 'No speech')
        lang = speech_data.get('language_detected', 'unknown')
        sound_desc = "quiet environment"
        if sound_data:
            first = sound_data[0]
            sound_desc = f"{first.get('event', 'unknown') if isinstance(first, dict) else first} environment"
        emotion = emotion_data.get('emotional_state', 'neutral')
        snippet = f"'{text[:100]}...'" if len(text) > 100 else f"'{text}'"
        return f"Speaker ({lang}) in {sound_desc} with {emotion} tone. Speech: {snippet}"

    def _stage_reasoning(self, speech_data, sound_data, emotion_data) -> List[str]:
        steps = [f"Speaker content: '{speech_data.get('text', '')}'"]
        if sound_data:
            evs = [s.get('event', str(s)) if isinstance(s, dict) else s for s in sound_data[:3]]
            steps.append(f"Background: {', '.join(evs)}")
        else:
            steps.append("Background: Quiet/Clean audio")
        state = emotion_data.get('emotional_state', 'Neutral')
        tension = emotion_data.get('vocal_tension', 'Unknown')
        steps.append(f"Vocal: {state} (tension: {tension})")
        if state == "Urgent/Stressed":
            steps.append("Context: High urgency communication in complex acoustic environment")
        elif state in ["Calm/Neutral", "Normal"]:
            steps.append("Context: Routine communication in controlled environment")
        else:
            steps.append(f"Context: {state} communication with background activity")
        return steps

    def _stage_summarization(self, speech_data, sound_data, emotion_data, steps) -> str:
        text = speech_data.get('text', 'No speech')
        lang = speech_data.get('language_detected', 'unknown')
        state = emotion_data.get('emotional_state', 'Neutral')
        env = "Quiet Room"
        if sound_data:
            first = sound_data[0]
            env = first.get('event', env) if isinstance(first, dict) else str(first)
        return f"Context: {state} interaction in {env}. Language: {lang}. Summary: {text}"

    def _format_inference_output(self, r: Dict) -> str:
        lines = ["=" * 70, "AUDIO REASONING & INFERENCE ANALYSIS", "=" * 70, ""]
        lines += ["📋 ANALYSIS PLAN:", f"   {r.get('plan', 'N/A')}", ""]
        lines += ["🎯 ACOUSTIC SCENE CAPTION:", f"   {r.get('caption', 'N/A')}", ""]
        lines += ["🧠 LOGICAL REASONING CHAIN:"]
        for i, step in enumerate(r.get('reasoning_steps', []), 1):
            lines.append(f"   [{i}] {step}")
        lines += ["", "✅ FINAL INFERENCE SUMMARY:", f"   {r.get('final_summary', 'N/A')}", "", "=" * 70]
        return "\n".join(lines)

    # ── Acoustic analysis (Layer 1) ───────────────────────────────────────────

    def analyze_audio_signal(self, y: np.ndarray, sr: int = None) -> AudioEnvironmentProfile:
        if sr is None:
            sr = self.sr
        sf = self._extract_spectral_features(y, sr)
        tf = self._extract_temporal_features(y, sr)
        na = self._perform_noise_analysis(y, sr, sf)
        bp = self._analyze_background_noise(y, sr)
        si = self._infer_spatial_characteristics(y, sr)
        ctx = self._determine_environmental_context(na, sf, tf, si)
        return AudioEnvironmentProfile(
            dominant_noises=na['dominant_noises'],
            background_noise_floor=bp['noise_floor'],
            signal_to_noise_ratio=bp['snr'],
            acoustic_complexity=self._calculate_acoustic_complexity(sf),
            spatial_characteristics=si,
            environmental_context=ctx,
            risk_factors=self._identify_risk_factors(na, bp),
            quality_assessment=self._assess_audio_quality(y, sr, sf),
        )

    def _initialize_noise_database(self):
        return {
            'car_engine': {'freq_range': (100, 500), 'harmonics': [150, 300, 450]},
            'wind_noise': {'freq_range': (100, 1000), 'harmonics': []},
            'background_hum': {'freq_range': (50, 60), 'harmonics': [50, 100, 150]},
        }

    def _extract_spectral_features(self, y, sr):
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        return {
            'magnitude': magnitude,
            'power_db': librosa.power_to_db(magnitude ** 2, ref=np.max),
            'centroid': librosa.feature.spectral_centroid(y=y, sr=sr)[0],
            'rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr)[0],
            'zcr': librosa.feature.zero_crossing_rate(y)[0],
            'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13),
            'contrast': librosa.feature.spectral_contrast(y=y, sr=sr),
            'mel_db': mel_db,
            'freqs': self.fft_freqs,
        }

    def _extract_temporal_features(self, y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)[0]
        return {
            'onset_strength': onset_env,
            'rms': rms,
            'rms_db': librosa.power_to_db(rms ** 2),
            'energy_flux': np.diff(rms),
            'attack_time': self._estimate_attack_time(rms),
            'decay_time': self._estimate_decay_time(rms),
        }

    def _perform_noise_analysis(self, y, sr, spectral_features):
        magnitude = spectral_features['magnitude']
        freqs = spectral_features['freqs']
        avg_mag = np.mean(magnitude, axis=1)
        peaks, props = signal.find_peaks(avg_mag, height=np.percentile(avg_mag, 70), distance=5)
        dom_freqs = freqs[peaks]
        dom_mags = avg_mag[peaks]
        noises = []
        for freq, mag in sorted(zip(dom_freqs, dom_mags), key=lambda x: x[1], reverse=True)[:5]:
            ns = self._classify_noise_by_frequency(freq, mag, y, sr)
            if ns.confidence > 0.3:
                noises.append(ns)
        return {'dominant_noises': noises, 'dominant_frequencies': dom_freqs[:10]}

    def _classify_noise_by_frequency(self, freq, magnitude, y, sr) -> NoiseSignature:
        if freq < 100:
            cat, conf = NoiseCategory.MACHINERY, 0.7 if self._has_harmonic_structure(y, sr, freq) else 0.5
        elif freq < 500:
            cat, conf = NoiseCategory.TRAFFIC, 0.6
        elif freq < 1500:
            cat, conf = NoiseCategory.SPEECH, 0.75 if self._detect_speech_modulation(y, sr) else 0.4
        elif freq < 3000:
            cat, conf = NoiseCategory.ENVIRONMENTAL, 0.65
        else:
            cat, conf = NoiseCategory.STRUCTURAL, 0.6
        return NoiseSignature(
            category=cat,
            frequency_range=(max(0, freq - 200), freq + 200),
            temporal_pattern=self._analyze_temporal_pattern(y, sr, freq),
            intensity_level=float(magnitude),
            confidence=conf,
            harmonic_content=self._extract_harmonics(y, sr, freq),
            spectral_shape=self._characterize_spectral_shape(y, sr, freq),
        )

    def _has_harmonic_structure(self, y, sr, fundamental) -> bool:
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        mag = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        h_energy = sum(np.mean(mag[np.argmin(np.abs(freqs - fundamental * h)), :]) for h in [2, 3, 4])
        f_energy = np.mean(mag[np.argmin(np.abs(freqs - fundamental)), :])
        return h_energy > 0.3 * f_energy

    def _detect_speech_modulation(self, y, sr) -> bool:
        rms = librosa.feature.rms(y=y)[0]
        freqs_mod = np.fft.fftfreq(len(rms), d=1 / (sr / 512))
        mag_mod = np.abs(np.fft.fft(rms))
        band = (freqs_mod > 4) & (freqs_mod < 8)
        return bool(np.any(band) and np.max(mag_mod[band]) > np.percentile(mag_mod, 70))

    def _extract_harmonics(self, y, sr, fundamental) -> List[float]:
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        mag = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        result = []
        for h in range(1, 6):
            target = fundamental * h
            if target < sr / 2:
                idx = np.argmin(np.abs(freqs - target))
                result.append(float(np.mean(mag[idx, :])))
        return result

    def _analyze_temporal_pattern(self, y, sr, freq) -> str:
        low = max(freq - 100, 1)
        high = min(freq + 100, sr / 2 - 1)
        try:
            sos = signal.butter(4, [low, high], btype='band', fs=sr, output='sos')
            filtered = signal.sosfilt(sos, y)
            envelope = np.abs(signal.hilbert(filtered))
            variability = np.std(envelope) / (np.mean(envelope) + 1e-8)
        except Exception:
            variability = 0.5
        if variability < 0.3: return "continuous_steady"
        if variability < 0.6: return "continuous_variable"
        if variability < 1.0: return "intermittent_bursty"
        return "impulsive_sporadic"

    def _characterize_spectral_shape(self, y, sr, center_freq) -> str:
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        mag = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        mask = (freqs > center_freq - 500) & (freqs < center_freq + 500)
        local = mag[mask, :].mean(axis=1)
        if len(local) == 0: return "flat_broadband"
        if np.std(local) / (np.mean(local) + 1e-8) < 0.3: return "flat_broadband"
        if np.argmax(local) < len(local) // 2: return "rising_highpass"
        if np.argmax(local) > len(local) // 2: return "falling_lowpass"
        return "peaked_narrowband"

    def _analyze_background_noise(self, y, sr) -> Dict:
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        mag = np.abs(D)
        noise_floor = float(np.percentile(librosa.power_to_db(mag ** 2, ref=np.max), 10))
        signal_power = float(np.mean(librosa.power_to_db(mag ** 2, ref=np.max)))
        return {'noise_floor': noise_floor, 'signal_power': signal_power, 'snr': float(signal_power - noise_floor)}

    def _infer_spatial_characteristics(self, y, sr) -> Dict:
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        energy = np.mean(S_db, axis=0)
        decay_slope = float(np.polyfit(range(len(energy)), energy, 1)[0])
        rt60 = self._estimate_rt60(y, sr) if len(y) > sr else 0.2
        return {
            'reverberation_time_estimated_ms': float(rt60 * 1000),
            'decay_slope': decay_slope,
            'acoustic_environment': self._classify_acoustic_environment(rt60),
            'room_size_estimate': self._estimate_room_size(rt60),
            'echo_presence': self._detect_echo(y, sr),
        }

    def _estimate_rt60(self, y, sr, freq_band=(100, 2000)) -> float:
        try:
            sos = signal.butter(4, freq_band, btype='band', fs=sr, output='sos')
            filtered = signal.sosfilt(sos, y)
            rms = librosa.feature.rms(y=filtered)[0]
            rms_db = librosa.power_to_db(rms ** 2, ref=np.max)
            if len(rms_db) > 100:
                peak_idx = int(np.argmax(rms_db))
                indices = np.where(rms_db[peak_idx:] < rms_db[peak_idx] - 60)[0]
                if len(indices):
                    return float(np.clip(librosa.frames_to_time(indices[0], sr=sr, hop_length=512), 0.01, 10.0))
        except Exception:
            pass
        return 0.2

    def _classify_acoustic_environment(self, rt60) -> str:
        if rt60 < 0.15: return "dead_anechoic"
        if rt60 < 0.5:  return "acoustic_treated"
        if rt60 < 1.5:  return "normal_office_room"
        if rt60 < 3.0:  return "large_room_hall"
        return "highly_reverberant"

    def _estimate_room_size(self, rt60) -> str:
        if rt60 < 0.3: return "very_small_close_talk"
        if rt60 < 0.7: return "small_room_office"
        if rt60 < 1.5: return "medium_room"
        if rt60 < 3.0: return "large_room"
        return "very_large_space"

    def _detect_echo(self, y, sr) -> bool:
        corr = np.correlate(y, y, mode='full')
        corr = corr[len(corr) // 2:]
        corr_norm = corr / (corr[0] + 1e-8)
        region = corr_norm[sr // 10:sr]
        return bool(len(region) > 0 and np.max(region) > 0.5)

    def _calculate_acoustic_complexity(self, sf) -> float:
        mel_db = sf['mel_db']
        mel_n = mel_db - np.min(mel_db)
        mel_n = mel_n / (np.max(mel_n) + 1e-8)
        ef = -np.sum(mel_n * np.log(mel_n + 1e-8), axis=0)
        et = -np.sum(mel_n * np.log(mel_n + 1e-8), axis=1)
        tv = np.std(np.diff(np.mean(mel_db, axis=0)))
        return float(np.clip((np.mean(ef) + np.mean(et) + tv) / 30.0, 0, 1))

    def _determine_environmental_context(self, na, sf, tf, si) -> str:
        noises = na['dominant_noises']
        if not noises: return "Silent or very quiet environment"
        ctx_map = {
            NoiseCategory.TRAFFIC: "Urban/Traffic environment",
            NoiseCategory.MACHINERY: "Industrial/Factory setting",
            NoiseCategory.SPEECH: "Social/Communication context",
            NoiseCategory.ENVIRONMENTAL: "Outdoor/Natural environment",
            NoiseCategory.AMBIENT: "General ambient background",
            NoiseCategory.STRUCTURAL: "Building/Indoor space",
        }
        ctx = ctx_map.get(noises[0].category, "General environment")
        room = si.get('acoustic_environment', 'dead_anechoic')
        if room != 'dead_anechoic':
            ctx += f" with {room} characteristics"
        return ctx

    def _identify_risk_factors(self, na, bp) -> List[str]:
        risks = []
        if bp.get('noise_floor', 0) > -20:
            risks.append("High background noise levels — potential hearing hazard")
        if bp.get('snr', 0) < 5:
            risks.append("Poor signal-to-noise ratio — difficulty understanding speech")
        if any(n.category == NoiseCategory.MACHINERY for n in na.get('dominant_noises', [])):
            risks.append("Industrial noise presence — occupational exposure concern")
        if len(na.get('dominant_noises', [])) > 5:
            risks.append("Complex multi-source acoustic environment")
        return risks

    def _assess_audio_quality(self, y, sr, sf) -> Dict:
        mel_db = sf['mel_db']
        low = np.mean(mel_db[:20, :])
        mid = np.mean(mel_db[50:80, :])
        high = np.mean(mel_db[100:, :])
        balance = float(np.clip(
            1.0 - np.std([low, mid, high]) / (np.mean([low, mid, high]) + 1e-8), 0, 1
        ))
        return {
            'dynamic_range_db': float(np.max(mel_db) - np.min(mel_db)),
            'clipping_ratio': float(np.sum(np.abs(y) > 0.99) / len(y)),
            'noise_floor_uniformity': float(np.std(np.min(mel_db, axis=0))),
            'spectral_balance_score': balance,
        }

    def _estimate_attack_time(self, rms) -> float:
        if len(rms) > 10:
            pk = int(np.argmax(rms))
            if pk > 5: return float(np.mean(np.diff(rms[:pk])))
        return 0.0

    def _estimate_decay_time(self, rms) -> float:
        if len(rms) > 10:
            pk = int(np.argmax(rms))
            if pk < len(rms) - 5: return float(np.mean(np.diff(rms[pk:])))
        return 0.0

    def get_model_info(self) -> Dict:
        return {
            'model_type': 'Audio Environment Analysis Engine + OpenRouter LLM Reasoning',
            'reasoning_backend': 'OpenRouter AI API (OpenAI-compatible)',
            'llm_model': self._model,
            'base_url': self._base_url,
            'sample_rate': self.sr,
            'fft_window_size': self.n_fft,
            'hop_length': self.hop_length,
            'noise_categories': [c.value for c in NoiseCategory],
            'reasoning_pipeline': 'Two-Layer (Acoustic + Semantic) → OpenRouter → Structured Report',
        }


# ── Report generator ──────────────────────────────────────────────────────────

class AudioAnalysisReporter:
    @staticmethod
    def generate_report(profile: AudioEnvironmentProfile) -> str:
        r = ["=" * 70, "AUDIO ENVIRONMENT ANALYSIS REPORT", "=" * 70, "",
             "📍 ENVIRONMENTAL ASSESSMENT:", f"   Context: {profile.environmental_context}", "",
             "🔊 IDENTIFIED NOISE SOURCES:"]
        if profile.dominant_noises:
            for i, n in enumerate(profile.dominant_noises, 1):
                r += [f"\n   [{i}] {n.category.value.upper()}",
                      f"       Freq Range  : {n.frequency_range[0]:.0f}–{n.frequency_range[1]:.0f} Hz",
                      f"       Pattern     : {n.temporal_pattern}",
                      f"       Intensity   : {n.intensity_level:.2f}",
                      f"       Confidence  : {n.confidence * 100:.1f}%"]
        else:
            r.append("   No significant noise sources detected")
        r += ["", "📊 ACOUSTIC MEASUREMENTS:",
              f"   Noise Floor : {profile.background_noise_floor:.1f} dB",
              f"   SNR         : {profile.signal_to_noise_ratio:.1f} dB",
              f"   Complexity  : {profile.acoustic_complexity * 100:.1f}%",
              "", "🏠 SPATIAL CHARACTERISTICS:",
              f"   Environment : {profile.spatial_characteristics.get('acoustic_environment', 'Unknown')}",
              f"   Room Size   : {profile.spatial_characteristics.get('room_size_estimate', 'Unknown')}",
              f"   RT60        : {profile.spatial_characteristics.get('reverberation_time_estimated_ms', 0):.0f} ms",
              "", "✅ RECORDING QUALITY:",
              f"   Dynamic Range: {profile.quality_assessment.get('dynamic_range_db', 0):.1f} dB",
              f"   Clipping     : {profile.quality_assessment.get('clipping_ratio', 0) * 100:.2f}%",
              "", "⚠️  RISK FACTORS:"]
        if profile.risk_factors:
            r += [f"   • {rf}" for rf in profile.risk_factors]
        else:
            r.append("   None identified")
        r += ["", "=" * 70]
        return "\n".join(r)

"""
Module: Data Adapter

Purpose:
    Orchestrates the complete audio analysis pipeline and merges outputs
    from all modules into a unified data structure.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AudioAnalysisData:
    environmental_context: str
    dominant_noises: List[Dict[str, Any]]
    acoustic_measurements: Dict[str, float]
    spatial_characteristics: Dict[str, Any]
    quality_assessment: Dict[str, float]
    risk_factors: List[str]


@dataclass
class EmotionAnalysisData:
    emotional_state: str
    confidence: float
    vocal_tension: Optional[str] = None


@dataclass
class SpeechData:
    transcript: str
    language: str = "en"
    confidence: float = 1.0


@dataclass
class SoundDetectionData:
    events: List[Dict[str, Any]]
    primary_event: Optional[str] = None


@dataclass
class RiskAnalysisData:
    risk_level: str
    risk_score: float
    situation_type: str
    keywords_detected: List[str]
    signals_detected: List[str]
    confidence: float


class DataAdapter:
    """
    Orchestrates integration of all audio analysis modules.
    Merges Layer 1 (acoustic) and Layer 2 (semantic) analyses.
    """

    def __init__(self):
        self.timestamp = None

    def merge_all_analysis(self,
                           audio_analysis: Optional[Dict[str, Any]] = None,
                           speech_data: Optional[Dict[str, Any]] = None,
                           sound_events: Optional[List[Dict[str, Any]]] = None,
                           emotion_data: Optional[Dict[str, Any]] = None,
                           risk_analysis: Optional[Dict[str, Any]] = None,
                           additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.timestamp = datetime.now().isoformat()

        merged_context = {
            "timestamp": self.timestamp,
            "metadata": {
                "pipeline_version": "2.0",
                "architecture": "two-layer-reasoning",
                "layer1": "acoustic-environment-detection",
                "layer2": "situational-context-detection"
            }
        }

        # Layer 1: Acoustic Environment
        if audio_analysis:
            merged_context["environment_prediction"] = self._extract_audio_features(audio_analysis)
        else:
            merged_context["environment_prediction"] = self._empty_audio_prediction()

        # Speech
        if speech_data:
            merged_context["speech"] = {
                "transcript": speech_data.get("transcript", ""),
                "language": speech_data.get("language", "en"),
                "confidence": speech_data.get("confidence", 0.0)
            }
        else:
            merged_context["speech"] = self._empty_speech_data()

        # Sound Events
        if sound_events:
            merged_context["sound_events"] = {
                "events": sound_events,
                "primary_event": sound_events[0].get("event", "") if sound_events else ""
            }
        else:
            merged_context["sound_events"] = self._empty_sound_events()

        # Emotion
        if emotion_data:
            merged_context["emotion_prediction"] = {
                "emotional_state": emotion_data.get("emotional_state", "neutral"),
                "confidence": emotion_data.get("confidence", 0.0),
                "vocal_tension": emotion_data.get("vocal_tension", "unknown")
            }
        else:
            merged_context["emotion_prediction"] = self._empty_emotion_data()

        # Layer 2: Risk Analysis
        if risk_analysis:
            merged_context["risk_analysis"] = self._extract_risk_analysis(risk_analysis)
        else:
            merged_context["risk_analysis"] = self._empty_risk_analysis()

        # Synthesized event pattern
        merged_context["event_pattern"] = self._synthesize_event_pattern(
            merged_context["environment_prediction"],
            merged_context["sound_events"],
            merged_context["emotion_prediction"],
            merged_context["risk_analysis"]
        )

        if additional_context:
            merged_context["additional_context"] = additional_context

        return merged_context

    def _extract_audio_features(self, audio_analysis: Dict) -> Dict[str, Any]:
        json_export = audio_analysis.get("json_export", {})
        return {
            "context": json_export.get("environmental_context", ""),
            "dominant_noises": [
                {
                    "category": noise.get("category", "unknown"),
                    "frequency_range": noise.get("frequency_range", (0, 0)),
                    "temporal_pattern": noise.get("temporal_pattern", ""),
                    "intensity": noise.get("intensity", 0.0),
                    "confidence": noise.get("confidence", 0.0)
                }
                for noise in json_export.get("dominant_noises", [])
            ],
            "acoustic_measurements": json_export.get("acoustic_measurements", {}),
            "spatial_characteristics": json_export.get("spatial_characteristics", {}),
            "quality_assessment": json_export.get("quality_assessment", {}),
            "risk_factors": json_export.get("risk_factors", [])
        }

    def _extract_risk_analysis(self, risk_analysis: Dict) -> Dict[str, Any]:
        return {
            "risk_level": risk_analysis.get("risk_level", "low"),
            "risk_score": risk_analysis.get("risk_score", 0.0),
            "situation_type": risk_analysis.get("situation_type", "normal_conversation"),
            "keywords_detected": risk_analysis.get("keywords_detected", []),
            "signals_detected": risk_analysis.get("signals_detected", []),
            "confidence": risk_analysis.get("confidence", 0.0),
            "reasoning": risk_analysis.get("reasoning", "")
        }

    def _synthesize_event_pattern(self, audio_features, sound_events,
                                   emotion, risk_analysis) -> Dict[str, Any]:
        return {
            "audio_context": audio_features.get("context", ""),
            "primary_sound": sound_events.get("primary_event", ""),
            "speaker_emotion": emotion.get("emotional_state", "neutral"),
            "risk_level": risk_analysis.get("risk_level", "low"),
            "situation_type": risk_analysis.get("situation_type", "normal_conversation"),
            "urgency": self._compute_urgency_level(risk_analysis),
            "confidence": min(
                emotion.get("confidence", 0.0),
                risk_analysis.get("confidence", 0.0)
            )
        }

    def _compute_urgency_level(self, risk_analysis: Dict) -> str:
        risk_level = risk_analysis.get("risk_level", "low")
        situation = risk_analysis.get("situation_type", "")
        if risk_level == "high" or situation in ["emergency", "medical", "conflict"]:
            return "urgent"
        elif risk_level == "moderate":
            return "elevated"
        return "routine"

    def _empty_audio_prediction(self) -> Dict[str, Any]:
        return {
            "context": "No audio analysis available",
            "dominant_noises": [],
            "acoustic_measurements": {},
            "spatial_characteristics": {},
            "quality_assessment": {},
            "risk_factors": []
        }

    def _empty_speech_data(self) -> Dict[str, Any]:
        return {"transcript": "No speech detected", "language": "unknown", "confidence": 0.0}

    def _empty_sound_events(self) -> Dict[str, Any]:
        return {"events": [], "primary_event": ""}

    def _empty_emotion_data(self) -> Dict[str, Any]:
        return {"emotional_state": "unknown", "confidence": 0.0, "vocal_tension": "unknown"}

    def _empty_risk_analysis(self) -> Dict[str, Any]:
        return {
            "risk_level": "low", "risk_score": 0.0,
            "situation_type": "normal_conversation",
            "keywords_detected": [], "signals_detected": [],
            "confidence": 0.0, "reasoning": "No transcript available."
        }

    def format_for_openrouter(self, merged_context: Dict[str, Any]) -> str:
        """Format merged context as a structured prompt for OpenRouter."""
        prompt = f"""
AUDIO ANALYSIS CONTEXT — LAYER 1 & LAYER 2 SYNTHESIS

LAYER 1: ACOUSTIC ENVIRONMENT
{self._format_audio_section(merged_context["environment_prediction"])}

TRANSCRIPT
{merged_context["speech"]["transcript"]}

SOUND EVENTS
{self._format_sound_events(merged_context["sound_events"])}

EMOTIONAL ANALYSIS
{self._format_emotion_section(merged_context["emotion_prediction"])}

LAYER 2: SITUATIONAL CONTEXT (Risk Analysis)
{self._format_risk_section(merged_context["risk_analysis"])}

SYNTHESIZED EVENT PATTERN
{self._format_event_pattern(merged_context["event_pattern"])}

---

Based on this analysis, provide structured reasoning as JSON:
{{
  "location": "...",
  "activity": "...",
  "emotion": "...",
  "risk_level": "...",
  "confidence": 0.0,
  "explanation": "Evidence-based summary."
}}
"""
        return prompt.strip()

    def _format_audio_section(self, audio_features: Dict) -> str:
        lines = [f"Environment: {audio_features.get('context', 'Unknown')}"]
        for noise in audio_features.get("dominant_noises", [])[:3]:
            lines.append(f"  - {noise.get('category', 'Unknown')}: intensity {noise.get('intensity', 0):.1f}")
        measurements = audio_features.get("acoustic_measurements", {})
        if measurements:
            lines.append(f"SNR: {measurements.get('signal_to_noise_ratio_db', 0):.1f} dB")
        return "\n".join(lines)

    def _format_sound_events(self, sound_events: Dict) -> str:
        if sound_events.get("primary_event"):
            return f"Primary: {sound_events['primary_event']}"
        return "No significant sounds detected"

    def _format_emotion_section(self, emotion: Dict) -> str:
        return (
            f"State: {emotion.get('emotional_state', 'Unknown')} "
            f"(confidence: {emotion.get('confidence', 0):.2f})"
        )

    def _format_risk_section(self, risk_analysis: Dict) -> str:
        lines = [
            f"Risk Level : {risk_analysis.get('risk_level', 'Low').upper()}",
            f"Risk Score : {risk_analysis.get('risk_score', 0):.2f}",
            f"Situation  : {risk_analysis.get('situation_type', 'normal').replace('_', ' ').title()}",
        ]
        if risk_analysis.get("keywords_detected"):
            lines.append(f"Indicators : {', '.join(risk_analysis['keywords_detected'][:5])}")
        if risk_analysis.get("signals_detected"):
            lines.append(f"Signals    : {', '.join(risk_analysis['signals_detected'][:3])}")
        return "\n".join(lines)

    def _format_event_pattern(self, pattern: Dict) -> str:
        return (
            f"Audio Context  : {pattern.get('audio_context', 'Unknown')}\n"
            f"Primary Sound  : {pattern.get('primary_sound', 'None')}\n"
            f"Speaker Emotion: {pattern.get('speaker_emotion', 'Unknown')}\n"
            f"Risk Level     : {pattern.get('risk_level', 'Low')}\n"
            f"Situation      : {pattern.get('situation_type', 'Normal')}\n"
            f"Urgency        : {pattern.get('urgency', 'Routine').upper()}"
        )

"""
Module: Semantic Context Analyzer (Layer 2)

Purpose:
    Analyzes transcript text to detect emergency-related keywords,
    vulnerability signals, and urgency indicators.
    Computes a risk score to enable semantic context detection.

Output:
    Risk analysis dictionary with risk_level, risk_score, situation_type,
    keywords_detected, signals_detected, confidence, reasoning.
"""

from typing import Dict, List, Any
from enum import Enum
import re
from dataclasses import dataclass

# ── Keyword banks ─────────────────────────────────────────────────────────────

EMERGENCY_KEYWORDS = [
    "help", "hurt", "breathing", "fell", "injured", "blood", "ambulance",
    "emergency", "accident", "pain", "crash", "fire", "trapped", "dying",
    "unconscious", "choking", "seizure", "overdose", "poisoning", "attack",
    "assault", "stabbing", "shot", "gunshot", "heart attack", "stroke",
    "allergic reaction", "anaphylaxis", "severe", "critical", "urgent",
    "dead", "death", "bleeding", "broken", "fracture",
    "unable to breathe", "can't breathe", "stop breathing"
]

INJURY_KEYWORDS = [
    "injury", "injured", "wound", "broken bone", "fracture", "sprain",
    "burn", "cut", "bleeding", "blood", "abrasion", "laceration",
    "contusion", "pain", "hurt", "damage", "trauma", "concussion",
    "head injury", "severe pain", "loss of consciousness", "unconscious"
]

VULNERABILITY_KEYWORDS = [
    "child", "kid", "baby", "toddler", "young", "little",
    "mom", "mommy", "dad", "daddy", "uncle", "auntie", "please help"
]

GUIDANCE_PHRASES = [
    "stay on the phone", "make sure", "check if", "are you safe",
    "is anyone with you", "where are you", "what is your location",
    "stay calm", "keep pressure", "apply pressure", "call ambulance",
    "call police", "emergency services", "hang up", "don't move",
    "stay still", "help is on the way", "emergency responders",
    "paramedics", "what happened", "are you hurt"
]

CONFLICT_KEYWORDS = [
    "fight", "fighting", "hit", "punch", "kick", "yelling", "screaming",
    "arguing", "conflict", "violent", "weapon", "gun", "knife", "threat",
    "threatening", "abuse", "abusive", "hit me", "hurt me", "attack me",
    "angry", "furious", "rage", "killing", "death threat", "dead"
]

MEDICAL_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "seizure", "choking",
    "difficulty breathing", "unconscious", "allergic", "allergic reaction",
    "overdose", "poisoned", "poisoning", "diabetes", "diabetic", "asthma",
    "asthma attack", "anaphylaxis", "severe allergy", "temperature",
    "fever", "blood pressure", "bleeding heavily", "lose blood"
]

PUBLIC_EVENT_KEYWORDS = [
    "crowd", "crowded", "concert", "event", "rally", "protest", "festival",
    "gathering", "people", "audience", "spectators", "crowds", "packed",
    "busy", "venue", "stadium", "arena", "theater", "hall"
]

# ── Enums ─────────────────────────────────────────────────────────────────────

class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class SituationType(Enum):
    NORMAL_CONVERSATION = "normal_conversation"
    EMERGENCY = "emergency"
    CONFLICT = "conflict"
    MEDICAL = "medical"
    PUBLIC_EVENT = "public_event"
    UNKNOWN = "unknown"


@dataclass
class RiskAnalysis:
    risk_level: str
    risk_score: float
    situation_type: str
    keywords_detected: List[str]
    signals_detected: List[str]
    confidence: float
    reasoning: str


# ── Analyzer ──────────────────────────────────────────────────────────────────

class SemanticAnalyzer:
    """
    Analyzes transcript text to detect emergency situations and contextual risks.
    Implements Layer 2 of the two-layer reasoning system.
    """

    def __init__(self):
        self.emergency_keywords = EMERGENCY_KEYWORDS
        self.injury_keywords = INJURY_KEYWORDS
        self.vulnerability_keywords = VULNERABILITY_KEYWORDS
        self.guidance_phrases = GUIDANCE_PHRASES
        self.conflict_keywords = CONFLICT_KEYWORDS
        self.medical_keywords = MEDICAL_KEYWORDS
        self.public_event_keywords = PUBLIC_EVENT_KEYWORDS

    def analyze(self, transcript: str) -> Dict[str, Any]:
        """
        Analyze transcript for risk indicators and contextual signals.

        Returns dict with: risk_level, risk_score, situation_type,
        keywords_detected, signals_detected, confidence, reasoning.
        """
        if not transcript or not isinstance(transcript, str):
            return self._create_empty_analysis()

        text_lower = transcript.lower()

        emergency_matches = self._detect_category(text_lower, self.emergency_keywords)
        injury_matches = self._detect_category(text_lower, self.injury_keywords)
        vulnerability_matches = self._detect_category(text_lower, self.vulnerability_keywords)
        guidance_matches = self._detect_category(text_lower, self.guidance_phrases)
        conflict_matches = self._detect_category(text_lower, self.conflict_keywords)
        medical_matches = self._detect_category(text_lower, self.medical_keywords)
        public_event_matches = self._detect_category(text_lower, self.public_event_keywords)

        all_keywords = (
            emergency_matches + injury_matches + vulnerability_matches +
            conflict_matches + medical_matches
        )
        all_signals = guidance_matches.copy()

        risk_score = self._compute_risk_score(
            emergency_matches, injury_matches, conflict_matches,
            medical_matches, guidance_matches, vulnerability_matches
        )

        situation_type = self._classify_situation(
            emergency_matches, injury_matches, conflict_matches,
            medical_matches, public_event_matches, guidance_matches
        )

        risk_level = self._determine_risk_level(risk_score)
        confidence = min(1.0, len(all_keywords) * 0.2 + risk_score * 0.5)

        reasoning = self._generate_reasoning(
            risk_level, situation_type, all_keywords, all_signals, risk_score
        )

        return {
            "risk_level": risk_level,
            "risk_score": float(risk_score),
            "situation_type": situation_type,
            "keywords_detected": list(set(all_keywords)),
            "signals_detected": list(set(all_signals)),
            "confidence": float(min(1.0, confidence)),
            "reasoning": reasoning,
            "category_breakdown": {
                "emergency": emergency_matches,
                "injury": injury_matches,
                "vulnerability": vulnerability_matches,
                "conflict": conflict_matches,
                "medical": medical_matches,
                "guidance": guidance_matches,
                "public_event": public_event_matches
            }
        }

    def _detect_category(self, text: str, keywords: List[str]) -> List[str]:
        detected = []
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                detected.append(keyword)
        return detected

    def _compute_risk_score(self, emergency, injury, conflict,
                             medical, guidance, vulnerability) -> float:
        score = 0.0
        score += min(1.0, len(emergency) * 0.3)
        score += len(injury) * 0.25
        score += len(medical) * 0.2
        score += len(conflict) * 0.25
        score += len(guidance) * 0.15
        score += len(vulnerability) * 0.1
        return min(1.0, score)

    def _classify_situation(self, emergency, injury, conflict,
                              medical, public_event, guidance) -> str:
        if emergency and guidance:
            return SituationType.EMERGENCY.value
        elif emergency or injury:
            return SituationType.EMERGENCY.value
        if medical:
            return SituationType.MEDICAL.value
        if conflict:
            return SituationType.CONFLICT.value
        if public_event:
            return SituationType.PUBLIC_EVENT.value
        return SituationType.NORMAL_CONVERSATION.value

    def _determine_risk_level(self, risk_score: float) -> str:
        if risk_score < 0.3:
            return RiskLevel.LOW.value
        elif risk_score < 0.6:
            return RiskLevel.MODERATE.value
        else:
            return RiskLevel.HIGH.value

    def _generate_reasoning(self, risk_level, situation_type,
                             keywords, signals, risk_score) -> str:
        parts = []
        situation_explanations = {
            "emergency": "Emergency situation detected based on language indicators and urgency signals.",
            "medical": "Medical emergency indicators detected in the transcript.",
            "conflict": "Conflict or violent situation indicators detected.",
            "public_event": "Public event or gathering context detected.",
            "normal_conversation": "Normal conversation without emergency indicators."
        }
        parts.append(situation_explanations.get(situation_type, "Unknown situation."))
        if keywords:
            parts.append(f"Key indicators: {', '.join(keywords[:3])}")
            if len(keywords) > 3:
                parts.append(f"and {len(keywords) - 3} additional indicators")
        if signals:
            parts.append(f"Dispatch/guidance signals: {', '.join(signals[:2])}")
        risk_explanations = {
            "high": "High risk situation requiring immediate intervention.",
            "moderate": "Moderate risk situation that warrants attention.",
            "low": "Low risk situation with minimal emergency indicators."
        }
        parts.append(risk_explanations.get(risk_level, "Risk level unclear."))
        return " ".join(parts)

    def _create_empty_analysis(self) -> Dict[str, Any]:
        return {
            "risk_level": RiskLevel.LOW.value,
            "risk_score": 0.0,
            "situation_type": SituationType.NORMAL_CONVERSATION.value,
            "keywords_detected": [],
            "signals_detected": [],
            "confidence": 0.0,
            "reasoning": "No transcript available for analysis.",
            "category_breakdown": {
                "emergency": [], "injury": [], "vulnerability": [],
                "conflict": [], "medical": [], "guidance": [], "public_event": []
            }
        }

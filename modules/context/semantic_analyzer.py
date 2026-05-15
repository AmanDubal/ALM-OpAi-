"""
Module: Semantic Context Analyzer

Purpose:
    Analyzes transcript text to detect emergency-related keywords, vulnerability signals,
    and urgency indicators. Computes a risk score to enable semantic context detection.

This module implements Layer 2 of the two-layer reasoning system:
- Layer 1: Acoustic Environment Detection (existing)
- Layer 2: Situational Context Detection (NEW - this module)

Working:
    - Analyzes transcript text for emergency keywords
    - Detects injury-related phrases
    - Detects vulnerability (child speaker indicators)
    - Detects urgency signals (guidance phrases)
    - Computes risk score (0-1)
    - Classifies situation type

Output:
    Risk analysis dictionary with:
    - risk_level: "low" / "moderate" / "high"
    - risk_score: float (0-1)
    - situation_type: Classified scenario type
    - keywords_detected: List of detected keywords
    - signals_detected: List of detected signals
    - confidence: Confidence in the analysis
"""

from typing import Dict, List, Tuple, Any
from enum import Enum
import re
from dataclasses import dataclass

# ============================================================================
# CONFIGURATION & KEYWORDS
# ============================================================================

# Emergency-related keywords
EMERGENCY_KEYWORDS = [
    "help", "hurt", "breathing", "fell", "injured", "blood", "ambulance",
    "emergency", "accident", "pain", "crash", "fire", "trapped", "dying",
    "unconscious", "choking", "seizure", "overdose", "poisoning", "attack",
    "assault", "stabbing", "shot", "gunshot", "heart attack", "stroke",
    "allergic reaction", "anaphylaxis", "severe", "critical", "urgent",
    "dying", "dead", "death", "bleeding", "severe bleeding", "broken",
    "fracture", "unable to breathe", "can't breathe", "stop breathing"
]

# Injury-related phrases
INJURY_KEYWORDS = [
    "injury", "injured", "wound", "broken bone", "fracture", "sprain",
    "burn", "cut", "bleeding", "blood", "abrasion", "laceration",
    "contusion", "pain", "hurt", "damage", "trauma", "concussion",
    "head injury", "severe pain", "loss of consciousness", "unconscious"
]

# Vulnerability indicators (child speaker)
VULNERABILITY_KEYWORDS = [
    "child", "kid", "baby", "toddler", "young", "little",
    "mom", "mommy", "dad", "daddy", "uncle", "auntie", "please help"
]

# Urgency/guidance phrases (dispatch call indicators)
GUIDANCE_PHRASES = [
    "stay on the phone", "make sure", "check if", "are you safe",
    "is anyone with you", "where are you", "what is your location",
    "stay calm", "keep pressure", "apply pressure", "call ambulance",
    "call police", "emergency services", "hang up", "don't move",
    "stay still", "help is on the way", "emergency responders",
    "paramedics", "what happened", "are you hurt"
]

# Conflict/violence indicators
CONFLICT_KEYWORDS = [
    "fight", "fighting", "hit", "punch", "kick", "yelling", "screaming",
    "arguing", "conflict", "violent", "weapon", "gun", "knife", "threat",
    "threatening", "abuse", "abusive", "hit me", "hurt me", "attack me",
    "angry", "furious", "rage", "killing", "death threat", "dead"
]

# Medical emergency indicators
MEDICAL_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "seizure", "choking",
    "difficulty breathing", "unconscious", "allergic", "allergic reaction",
    "overdose", "poisoned", "poisoning", "diabetes", "diabetic", "asthma",
    "asthma attack", "anaphylaxis", "severe allergy", "temperature",
    "fever", "blood pressure", "bleeding heavily", "lose blood"
]

# Public event indicators
PUBLIC_EVENT_KEYWORDS = [
    "crowd", "crowded", "concert", "event", "rally", "protest", "festival",
    "gathering", "people", "audience", "spectators", "crowds", "packed",
    "busy", "venue", "stadium", "arena", "theater", "hall"
]

# ============================================================================
# RISK CLASSIFICATION SYSTEM
# ============================================================================

class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

class SituationType(Enum):
    """Situation classification types"""
    NORMAL_CONVERSATION = "normal_conversation"
    EMERGENCY = "emergency"
    CONFLICT = "conflict"
    MEDICAL = "medical"
    PUBLIC_EVENT = "public_event"
    UNKNOWN = "unknown"

@dataclass
class RiskAnalysis:
    """Risk analysis result"""
    risk_level: str
    risk_score: float
    situation_type: str
    keywords_detected: List[str]
    signals_detected: List[str]
    confidence: float
    reasoning: str

# ============================================================================
# SEMANTIC CONTEXT ANALYZER
# ============================================================================

class SemanticAnalyzer:
    """
    Analyzes transcript text to detect emergency situations and contextual risks.
    Implements Layer 2 of the two-layer reasoning system.
    """
    
    def __init__(self):
        """Initialize semantic analyzer with keyword databases"""
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
        
        Args:
            transcript: Speech transcript text
        
        Returns:
            Dictionary with risk analysis including:
            - risk_level: "low" / "moderate" / "high"
            - risk_score: float (0-1)
            - situation_type: Type of situation detected
            - keywords_detected: List of detected keywords
            - signals_detected: List of detected signals
            - confidence: Confidence in analysis
            - reasoning: Explanation of analysis
        """
        if not transcript or not isinstance(transcript, str):
            return self._create_empty_analysis()
        
        # Normalize text for analysis
        text_lower = transcript.lower()
        
        # Detect keywords by category
        emergency_matches = self._detect_category(text_lower, self.emergency_keywords)
        injury_matches = self._detect_category(text_lower, self.injury_keywords)
        vulnerability_matches = self._detect_category(text_lower, self.vulnerability_keywords)
        guidance_matches = self._detect_category(text_lower, self.guidance_phrases)
        conflict_matches = self._detect_category(text_lower, self.conflict_keywords)
        medical_matches = self._detect_category(text_lower, self.medical_keywords)
        public_event_matches = self._detect_category(text_lower, self.public_event_keywords)
        
        # Collect all detected keywords
        all_keywords = (emergency_matches + injury_matches + vulnerability_matches +
                       conflict_matches + medical_matches)
        all_signals = guidance_matches.copy()
        
        # Compute risk score
        risk_score = self._compute_risk_score(
            emergency_matches, injury_matches, conflict_matches,
            medical_matches, guidance_matches, vulnerability_matches
        )
        
        # Classify situation
        situation_type = self._classify_situation(
            emergency_matches, injury_matches, conflict_matches,
            medical_matches, public_event_matches, guidance_matches
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        # Calculate confidence
        confidence = min(1.0, len(all_keywords) * 0.2 + risk_score * 0.5)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            risk_level, situation_type, all_keywords, all_signals, risk_score
        )
        
        return {
            "risk_level": risk_level,
            "risk_score": float(risk_score),
            "situation_type": situation_type,
            "keywords_detected": list(set(all_keywords)),  # Deduplicate
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
        """
        Detect keywords from a specific category in text.
        Uses word-boundary matching to avoid false positives.
        
        Args:
            text: Lowercase transcript text
            keywords: List of keywords to detect
        
        Returns:
            List of detected keywords
        """
        detected = []
        for keyword in keywords:
            # Use word boundaries to match whole words/phrases
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                detected.append(keyword)
        
        return detected
    
    def _compute_risk_score(self,
                           emergency: List[str],
                           injury: List[str],
                           conflict: List[str],
                           medical: List[str],
                           guidance: List[str],
                           vulnerability: List[str]) -> float:
        """
        Compute overall risk score (0-1) based on detected indicators.
        
        Scoring logic:
        - Emergency keywords: +0.3 per keyword (up to 1.0)
        - Injury keywords: +0.25 per keyword
        - Medical keywords: +0.2 per keyword
        - Conflict keywords: +0.25 per keyword
        - Guidance phrases (dispatch indicator): +0.15 per phrase
        - Vulnerability: +0.1 per indicator
        
        Args:
            emergency: Detected emergency keywords
            injury: Detected injury keywords
            conflict: Detected conflict keywords
            medical: Detected medical keywords
            guidance: Detected guidance phrases
            vulnerability: Detected vulnerability indicators
        
        Returns:
            Risk score (0.0 to 1.0)
        """
        score = 0.0
        
        # Emergency keywords (highest weight)
        score += min(1.0, len(emergency) * 0.3)
        
        # Injury keywords
        score += len(injury) * 0.25
        
        # Medical keywords
        score += len(medical) * 0.2
        
        # Conflict keywords
        score += len(conflict) * 0.25
        
        # Guidance phrases (dispatch call indicator)
        score += len(guidance) * 0.15
        
        # Vulnerability signals (child speaker)
        score += len(vulnerability) * 0.1
        
        # Clamp to [0, 1] range
        return min(1.0, score)
    
    def _classify_situation(self,
                           emergency: List[str],
                           injury: List[str],
                           conflict: List[str],
                           medical: List[str],
                           public_event: List[str],
                           guidance: List[str]) -> str:
        """
        Classify the situation type based on detected keywords.
        
        Priority order:
        1. Emergency (if emergency keywords + guidance phrases)
        2. Medical (if medical keywords)
        3. Conflict (if conflict keywords)
        4. Public Event (if public event keywords)
        5. Normal Conversation (default)
        
        Args:
            emergency: Detected emergency keywords
            injury: Detected injury keywords
            conflict: Detected conflict keywords
            medical: Detected medical keywords
            public_event: Detected public event keywords
            guidance: Detected guidance phrases
        
        Returns:
            Situation type string
        """
        # Emergency call detection (emergency + guidance)
        if emergency and guidance:
            return SituationType.EMERGENCY.value
        elif emergency or injury:
            return SituationType.EMERGENCY.value
        
        # Medical emergency
        if medical:
            return SituationType.MEDICAL.value
        
        # Conflict/fight
        if conflict:
            return SituationType.CONFLICT.value
        
        # Public event
        if public_event:
            return SituationType.PUBLIC_EVENT.value
        
        # Default
        return SituationType.NORMAL_CONVERSATION.value
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """
        Determine risk level from risk score.
        
        Mapping:
        - 0.0 - 0.3: "low"
        - 0.3 - 0.6: "moderate"
        - 0.6 - 1.0: "high"
        
        Args:
            risk_score: Computed risk score
        
        Returns:
            Risk level string
        """
        if risk_score < 0.3:
            return RiskLevel.LOW.value
        elif risk_score < 0.6:
            return RiskLevel.MODERATE.value
        else:
            return RiskLevel.HIGH.value
    
    def _generate_reasoning(self,
                           risk_level: str,
                           situation_type: str,
                           keywords: List[str],
                           signals: List[str],
                           risk_score: float) -> str:
        """
        Generate human-readable reasoning for the risk analysis.
        
        Args:
            risk_level: Determined risk level
            situation_type: Classified situation type
            keywords: Detected keywords
            signals: Detected signals
            risk_score: Computed risk score
        
        Returns:
            Reasoning explanation
        """
        parts = []
        
        # Situation type explanation
        situation_explanations = {
            "emergency": "Emergency situation detected based on language indicators and urgency signals.",
            "medical": "Medical emergency indicators detected in the transcript.",
            "conflict": "Conflict or violent situation indicators detected.",
            "public_event": "Public event or gathering context detected.",
            "normal_conversation": "Normal conversation without emergency indicators."
        }
        
        parts.append(situation_explanations.get(situation_type, "Unknown situation."))
        
        # Keyword details
        if keywords:
            parts.append(f"Key indicators: {', '.join(keywords[:3])}")
            if len(keywords) > 3:
                parts.append(f"and {len(keywords) - 3} additional indicators")
        
        # Dispatch detection
        if signals:
            parts.append(f"Dispatch/guidance signals detected: {', '.join(signals[:2])}")
        
        # Risk assessment
        risk_explanations = {
            "high": "High risk situation requiring immediate intervention.",
            "moderate": "Moderate risk situation that warrants attention.",
            "low": "Low risk situation with minimal emergency indicators."
        }
        
        parts.append(risk_explanations.get(risk_level, "Risk level unclear."))
        
        return " ".join(parts)
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        """Create default analysis for empty/invalid input"""
        return {
            "risk_level": RiskLevel.LOW.value,
            "risk_score": 0.0,
            "situation_type": SituationType.NORMAL_CONVERSATION.value,
            "keywords_detected": [],
            "signals_detected": [],
            "confidence": 0.0,
            "reasoning": "No transcript available for analysis.",
            "category_breakdown": {
                "emergency": [],
                "injury": [],
                "vulnerability": [],
                "conflict": [],
                "medical": [],
                "guidance": [],
                "public_event": []
            }
        }

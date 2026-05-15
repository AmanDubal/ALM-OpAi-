"""
Module 4: Emotion / Paralinguistic Analysis Module

Purpose:
    Capture emotional or paralinguistic cues from speech.

Working:
    - Extracts MFCC features from speech
    - Classifies emotional state (calm, stressed, angry, etc.)

Techniques Used:
    - MFCC feature extraction
    - ML classifier (simulated mapping for demo)

Output:
    - Estimated emotional tone
"""

import numpy as np
import librosa
from typing import Dict, Optional
from enum import Enum


class EmotionClass(Enum):
    """Emotion classification classes."""
    NEUTRAL = "neutral"
    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUST = "disgust"
    SURPRISED = "surprised"


class EmotionAnalyzer:
    """
    Analyzes emotional tone in audio using feature extraction.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.3):
        """
        Initialize emotion analyzer.
        
        Args:
            confidence_threshold: Minimum confidence for emotion detection
        """
        self.confidence_threshold = confidence_threshold
        self.sample_rate = 16000
        self.n_mfcc = 13
        
        # Feature ranges for emotion classification (learned from data)
        self.emotion_profiles = self._create_emotion_profiles()
    
    def analyze(self, 
               audio: np.ndarray,
               sr: int = 16000) -> Dict:
        """
        Analyze emotional tone in audio.
        
        Args:
            audio: Audio signal
            sr: Sample rate
        
        Returns:
            Dictionary containing:
                - emotion: Detected emotion
                - confidence: Confidence score (0-1)
                - emotion_probabilities: Probability for each emotion
                - features: Extracted features used
        """
        try:
            # Resample if necessary
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)
            
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Extract features
            features = self._extract_features(audio)
            
            # Classify emotion
            emotion, confidence, probabilities = self._classify_emotion(features)
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'emotion_probabilities': probabilities,
                'features': features
            }
        
        except Exception as e:
            return {
                'emotion': 'unknown',
                'confidence': 0.0,
                'emotion_probabilities': {},
                'error': str(e)
            }
    
    def _extract_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract audio features for emotion analysis.
        
        Args:
            audio: Audio signal
        
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # MFCC (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        features['mfcc_mean'] = float(np.mean(mfcc))
        features['mfcc_std'] = float(np.std(mfcc))
        features['mfcc_max'] = float(np.max(mfcc))
        
        # Zero Crossing Rate (voice quality)
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # Spectral Centroid (brightness)
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        features['spec_centroid_mean'] = float(np.mean(spec_centroid))
        features['spec_centroid_std'] = float(np.std(spec_centroid))
        
        # Spectral Rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        features['spec_rolloff_mean'] = float(np.mean(spec_rolloff))
        
        # Chroma STFT (pitch content)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))
        
        # Energy
        energy = np.sqrt(np.mean(audio ** 2))
        features['energy'] = float(energy)
        
        # Pitch variance (simple approximation)
        features['pitch_variance'] = float(np.std(librosa.feature.tempogram(y=audio, sr=self.sample_rate)))
        
        return features
    
    def _classify_emotion(self, 
                         features: Dict[str, float]) -> tuple:
        """
        Classify emotion based on extracted features.
        
        Args:
            features: Dictionary of features
        
        Returns:
            Tuple of (emotion, confidence, probabilities_dict)
        """
        probabilities = {}
        
        # Calculate similarity scores for each emotion
        for emotion_name, profile in self.emotion_profiles.items():
            score = self._calculate_similarity(features, profile)
            probabilities[emotion_name] = score
        
        # Get emotion with highest score
        if probabilities:
            emotion = max(probabilities, key=probabilities.get)
            confidence = probabilities[emotion]
            
            if confidence < self.confidence_threshold:
                emotion = 'neutral'  # Default to neutral if low confidence
                confidence = 0.5
        else:
            emotion = 'unknown'
            confidence = 0.0
        
        return emotion, float(confidence), probabilities
    
    def _calculate_similarity(self, 
                            features: Dict[str, float],
                            profile: Dict[str, tuple]) -> float:
        """
        Calculate similarity between features and emotion profile.
        
        Args:
            features: Extracted features
            profile: Emotion profile with expected ranges
        
        Returns:
            Similarity score (0-1)
        """
        similarities = []
        
        for feature_name, (min_val, max_val) in profile.items():
            if feature_name in features:
                val = features[feature_name]
                # Normalize to 0-1 range
                if max_val > min_val:
                    normalized = (val - min_val) / (max_val - min_val)
                    normalized = np.clip(normalized, 0, 1)
                    similarity = 1 - abs(0.5 - normalized)  # Distance from midpoint
                    similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.5
    
    def _create_emotion_profiles(self) -> Dict:
        """
        Create emotion profiles based on feature ranges.
        
        Returns:
            Dictionary mapping emotions to feature profiles
        """
        # Emotion profiles: feature -> (min_range, max_range)
        # These are simplified profiles based on typical emotion characteristics
        
        return {
            'calm': {
                'mfcc_std': (5, 15),
                'energy': (0.01, 0.1),
                'spec_centroid_mean': (1000, 3000),
                'zcr_std': (0.01, 0.1)
            },
            'happy': {
                'mfcc_std': (15, 30),
                'energy': (0.05, 0.3),
                'spec_centroid_mean': (3000, 6000),
                'chroma_std': (0.1, 0.4)
            },
            'sad': {
                'mfcc_std': (8, 18),
                'energy': (0.02, 0.08),
                'spec_centroid_mean': (800, 2500),
                'zcr_std': (0.02, 0.08)
            },
            'angry': {
                'mfcc_std': (20, 40),
                'energy': (0.1, 0.5),
                'spec_centroid_mean': (4000, 8000),
                'spec_rolloff_mean': (8000, 12000)
            },
            'fearful': {
                'mfcc_std': (15, 35),
                'energy': (0.05, 0.2),
                'spec_centroid_mean': (3000, 7000),
                'zcr_std': (0.08, 0.2)
            },
            'neutral': {
                'mfcc_std': (10, 25),
                'energy': (0.03, 0.15),
                'spec_centroid_mean': (2000, 5000),
                'chroma_std': (0.05, 0.3)
            }
        }
    
    def _resample(self, 
                 audio: np.ndarray,
                 sr_orig: int,
                 sr_target: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Audio signal
            sr_orig: Original sample rate
            sr_target: Target sample rate
        
        Returns:
            Resampled audio
        """
        ratio = sr_target / sr_orig
        new_length = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio) - 1, new_length),
            np.arange(len(audio)),
            audio
        )
    
    def analyze_paralinguistic_cues(self, 
                                   audio: np.ndarray,
                                   sr: int = 16000) -> Dict:
        """
        Analyze paralinguistic cues: tone, prosody, and vocal characteristics.
        Crucial for tonal languages like Mandarin/Thai interpretation.
        
        Args:
            audio: Audio signal
            sr: Sample rate
        
        Returns:
            Dictionary with paralinguistic analysis:
                - emotional_state: State estimation (e.g., "Urgent/Stressed")
                - prosody_score: Prosody quality score (0-1)
                - vocal_tension: Tension level (Low/Medium/High)
                - pitch_variation: Pitch variation score
                - intensity_level: Overall intensity (0-1)
        """
        try:
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)
            
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Analyze pitch patterns
            pitch_var = self._analyze_pitch_variation(audio)
            
            # Analyze vocal tension from spectral features
            tension = self._estimate_vocal_tension(audio)
            
            # Analyze prosody (intonation patterns)
            prosody_score = self._analyze_prosody(audio)
            
            # Analyze intensity variation
            intensity = self._analyze_intensity(audio)
            
            # Infer emotional state from combination
            emotional_state = self._infer_emotional_state(pitch_var, tension, prosody_score)
            
            return {
                'emotional_state': emotional_state,
                'prosody_score': prosody_score,
                'vocal_tension': tension,
                'pitch_variation': pitch_var,
                'intensity_level': intensity
            }
        
        except Exception as e:
            return {
                'emotional_state': 'unknown',
                'error': str(e)
            }
    
    def _analyze_pitch_variation(self, audio: np.ndarray) -> float:
        """
        Analyze pitch variation patterns.
        High variation often indicates emotional content.
        
        Args:
            audio: Audio signal
        
        Returns:
            Pitch variation score (0-1)
        """
        # Use harmonic component for pitch variation
        harmonic = librosa.effects.harmonic(audio)
        
        # Calculate spectral flux (variation in spectrum)
        S = librosa.feature.melspectrogram(y=harmonic, sr=self.sample_rate)
        spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
        
        variation = float(np.mean(spectral_flux) / (np.max(spectral_flux) + 1e-8))
        return np.clip(variation, 0, 1)
    
    def _estimate_vocal_tension(self, audio: np.ndarray) -> str:
        """
        Estimate vocal tension from spectral characteristics.
        
        Args:
            audio: Audio signal
        
        Returns:
            Tension level: "Low", "Medium", or "High"
        """
        # Higher frequency content indicates tension
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        centroid_mean = np.mean(spec_centroid)
        
        # Normalize to typical human speech range (85-255 Hz for females, 85-180 Hz for males)
        if centroid_mean < 1500:
            return "Low"
        elif centroid_mean < 3000:
            return "Medium"
        else:
            return "High"
    
    def _analyze_prosody(self, audio: np.ndarray) -> float:
        """
        Analyze prosody (rhythm and intonation patterns).
        
        Args:
            audio: Audio signal
        
        Returns:
            Prosody score (0-1)
        """
        # Use onset strength to analyze rhythm
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        
        # Calculate energy variation (prosodic features)
        energy_variation = np.std(onset_env)
        prosody_score = float(energy_variation / (np.max(onset_env) + 1e-8))
        
        return np.clip(prosody_score, 0, 1)
    
    def _analyze_intensity(self, audio: np.ndarray) -> float:
        """
        Analyze overall intensity/loudness.
        
        Args:
            audio: Audio signal
        
        Returns:
            Intensity level (0-1)
        """
        # RMS energy normalized
        rms = np.sqrt(np.mean(audio**2))
        return float(np.clip(rms, 0, 1))
    
    def _infer_emotional_state(self, 
                              pitch_var: float, 
                              tension: str, 
                              prosody: float) -> str:
        """
        Infer emotional state from combined paralinguistic features.
        
        Args:
            pitch_var: Pitch variation score
            tension: Vocal tension level
            prosody: Prosody score
        
        Returns:
            Inferred emotional state
        """
        if tension == "High" and pitch_var > 0.6:
            return "Urgent/Stressed"
        elif tension == "High" and prosody > 0.7:
            return "Angry/Agitated"
        elif tension == "Low" and pitch_var < 0.3:
            return "Calm/Neutral"
        elif prosody > 0.7:
            return "Excited/Happy"
        else:
            return "Neutral"
    
    def get_emotion_categories(self) -> list:
        """
        Get list of emotion categories.
        
        Returns:
            List of emotion names
        """
        return [e.value for e in EmotionClass]
    
    def get_model_info(self) -> Dict:
        """
        Get information about emotion analyzer.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'Feature-based Emotion Analyzer',
            'method': 'MFCC + spectral features with profile matching',
            'emotions': self.get_emotion_categories(),
            'features_used': self.n_mfcc,
            'confidence_threshold': self.confidence_threshold
        }

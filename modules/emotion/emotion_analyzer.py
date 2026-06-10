"""
Module 4: Emotion / Paralinguistic Analysis Module

Purpose:
    Capture emotional or paralinguistic cues from speech.

Working:
    - Extracts MFCC features from speech
    - Classifies emotional state (calm, stressed, angry, etc.)

Output:
    - Estimated emotional tone
"""

import numpy as np
import librosa
from typing import Dict, Optional
from enum import Enum


class EmotionClass(Enum):
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
    Analyzes emotional tone in audio using MFCC feature extraction.
    """

    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.sample_rate = 16000
        self.n_mfcc = 13
        self.emotion_profiles = self._create_emotion_profiles()

    def analyze(self, audio: np.ndarray, sr: int = 16000) -> Dict:
        try:
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            features = self._extract_features(audio)
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
        features = {}
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        features['mfcc_mean'] = float(np.mean(mfcc))
        features['mfcc_std'] = float(np.std(mfcc))
        features['mfcc_max'] = float(np.max(mfcc))
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        features['spec_centroid_mean'] = float(np.mean(spec_centroid))
        features['spec_centroid_std'] = float(np.std(spec_centroid))
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        features['spec_rolloff_mean'] = float(np.mean(spec_rolloff))
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))
        energy = np.sqrt(np.mean(audio ** 2))
        features['energy'] = float(energy)
        features['pitch_variance'] = float(
            np.std(librosa.feature.tempogram(y=audio, sr=self.sample_rate))
        )
        return features

    def _classify_emotion(self, features: Dict[str, float]) -> tuple:
        probabilities = {}
        for emotion_name, profile in self.emotion_profiles.items():
            score = self._calculate_similarity(features, profile)
            probabilities[emotion_name] = score
        if probabilities:
            emotion = max(probabilities, key=probabilities.get)
            confidence = probabilities[emotion]
            if confidence < self.confidence_threshold:
                emotion = 'neutral'
                confidence = 0.5
        else:
            emotion = 'unknown'
            confidence = 0.0
        return emotion, float(confidence), probabilities

    def _calculate_similarity(self,
                               features: Dict[str, float],
                               profile: Dict[str, tuple]) -> float:
        similarities = []
        for feature_name, (min_val, max_val) in profile.items():
            if feature_name in features:
                val = features[feature_name]
                if max_val > min_val:
                    normalized = (val - min_val) / (max_val - min_val)
                    normalized = np.clip(normalized, 0, 1)
                    similarity = 1 - abs(0.5 - normalized)
                    similarities.append(similarity)
        return float(np.mean(similarities)) if similarities else 0.5

    def _create_emotion_profiles(self) -> Dict:
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

    def _resample(self, audio: np.ndarray, sr_orig: int, sr_target: int) -> np.ndarray:
        ratio = sr_target / sr_orig
        new_length = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio) - 1, new_length),
            np.arange(len(audio)),
            audio
        )

    def analyze_paralinguistic_cues(self, audio: np.ndarray, sr: int = 16000) -> Dict:
        try:
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            pitch_var = self._analyze_pitch_variation(audio)
            tension = self._estimate_vocal_tension(audio)
            prosody_score = self._analyze_prosody(audio)
            intensity = self._analyze_intensity(audio)
            emotional_state = self._infer_emotional_state(pitch_var, tension, prosody_score)
            return {
                'emotional_state': emotional_state,
                'prosody_score': prosody_score,
                'vocal_tension': tension,
                'pitch_variation': pitch_var,
                'intensity_level': intensity
            }
        except Exception as e:
            return {'emotional_state': 'unknown', 'error': str(e)}

    def _analyze_pitch_variation(self, audio: np.ndarray) -> float:
        harmonic = librosa.effects.harmonic(audio)
        S = librosa.feature.melspectrogram(y=harmonic, sr=self.sample_rate)
        spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        variation = float(np.mean(spectral_flux) / (np.max(spectral_flux) + 1e-8))
        return float(np.clip(variation, 0, 1))

    def _estimate_vocal_tension(self, audio: np.ndarray) -> str:
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        centroid_mean = np.mean(spec_centroid)
        if centroid_mean < 1500:
            return "Low"
        elif centroid_mean < 3000:
            return "Medium"
        else:
            return "High"

    def _analyze_prosody(self, audio: np.ndarray) -> float:
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        energy_variation = np.std(onset_env)
        prosody_score = float(energy_variation / (np.max(onset_env) + 1e-8))
        return float(np.clip(prosody_score, 0, 1))

    def _analyze_intensity(self, audio: np.ndarray) -> float:
        rms = np.sqrt(np.mean(audio ** 2))
        return float(np.clip(rms, 0, 1))

    def _infer_emotional_state(self,
                               pitch_var: float,
                               tension: str,
                               prosody: float) -> str:
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
        return [e.value for e in EmotionClass]

    def get_model_info(self) -> Dict:
        return {
            'model_type': 'Feature-based Emotion Analyzer',
            'method': 'MFCC + spectral features with profile matching',
            'emotions': self.get_emotion_categories(),
            'features_used': self.n_mfcc,
            'confidence_threshold': self.confidence_threshold
        }

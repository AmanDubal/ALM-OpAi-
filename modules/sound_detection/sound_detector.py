"""
Module 3: Non-Speech Sound Event Detection Module

Purpose:
    Identify environmental and background sounds present in the audio.

Working:
    - Analyzes full audio waveform
    - Detects sound events (traffic, crowd, aircraft, alarms, machinery, etc.)

Model Used:
    - Pretrained CNN-based YAMNet (Google)

Output:
    - List of detected sound events
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import List, Dict, Optional
import warnings


YAMNET_CLASS_MAP = {
    0: 'Speech',
    1: 'Dog',
    2: 'Cat',
    3: 'Rooster',
    4: 'Frog',
    5: 'Cow',
    6: 'Pig',
    7: 'Crow',
    8: 'Rain',
    9: 'Sea waves',
    10: 'Crackling fire',
    11: 'Crickets',
    12: 'Chirping birds',
    13: 'Motor vehicle',
    14: 'Alarm clock',
    15: 'Siren',
    16: 'Car horn',
    17: 'Explosion',
    18: 'Gunshot',
    19: 'Passing by motorcycle',
    20: 'Dog barking',
    21: 'Thunder',
    22: 'Wind',
    23: 'Thunderstorm',
    24: 'Aircraft',
    25: 'Helicopter',
    26: 'Human sounds',
    27: 'Applause',
    28: 'Cheering crowd'
}


class SoundDetector:
    """
    Detects sound events in audio using YAMNet model.
    """

    def __init__(self,
                 confidence_threshold: float = 0.3,
                 top_events: int = 5):
        self.confidence_threshold = confidence_threshold
        self.top_events = top_events
        self.sample_rate = 16000
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load YAMNet model: {str(e)}")
            self.model_loaded = False
            self.model = None

    def detect(self,
               audio: np.ndarray,
               sr: int = 16000) -> Dict:
        if not self.model_loaded:
            return {
                'events': [],
                'confidences': [],
                'all_predictions': [],
                'error': 'YAMNet model not loaded'
            }
        try:
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            max_length = self.sample_rate * 30
            if len(audio) > max_length:
                audio = audio[:max_length]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores, embeddings, spectrogram = self.model(audio)
            scores = scores.numpy()
            mean_scores = np.mean(scores, axis=0)
            top_indices = np.argsort(mean_scores)[::-1][:self.top_events]
            events = []
            confidences = []
            all_predictions = []
            for idx in top_indices:
                confidence = float(mean_scores[idx])
                if confidence >= self.confidence_threshold:
                    event_name = YAMNET_CLASS_MAP.get(int(idx), f'Unknown_{idx}')
                    events.append(event_name)
                    confidences.append(confidence)
                    all_predictions.append({'event': event_name, 'confidence': confidence})
            return {
                'events': events,
                'confidences': confidences,
                'all_predictions': all_predictions,
                'top_event': events[0] if events else 'No sounds detected'
            }
        except Exception as e:
            return {
                'events': [],
                'confidences': [],
                'all_predictions': [],
                'error': str(e)
            }

    def _resample(self, audio: np.ndarray, sr_orig: int, sr_target: int) -> np.ndarray:
        try:
            import librosa
            return librosa.resample(audio, orig_sr=sr_orig, target_sr=sr_target)
        except ImportError:
            ratio = sr_target / sr_orig
            new_length = int(len(audio) * ratio)
            return np.interp(
                np.linspace(0, len(audio) - 1, new_length),
                np.arange(len(audio)),
                audio
            )

    def get_event_categories(self) -> List[str]:
        return list(YAMNET_CLASS_MAP.values())

    def get_model_info(self) -> Dict:
        return {
            'model_name': 'YAMNet',
            'loaded': self.model_loaded,
            'input_sample_rate': self.sample_rate,
            'confidence_threshold': self.confidence_threshold,
            'top_events': self.top_events,
            'num_classes': len(YAMNET_CLASS_MAP)
        }

    def filter_events(self,
                      events: List[str],
                      categories: Optional[List[str]] = None) -> List[str]:
        if categories is None:
            return events
        return [e for e in events if e in categories]

    def is_speech_present(self, events: List[str]) -> bool:
        speech_keywords = ['speech', 'voice', 'talk', 'conversation']
        return any(any(kw in e.lower() for kw in speech_keywords) for e in events)

    def categorize_events(self, events: List[str]) -> Dict[str, List[str]]:
        categories = {
            'animal_sounds': [],
            'environmental': [],
            'human_made': [],
            'nature': [],
            'vehicles': [],
            'other': []
        }
        for event in events:
            event_lower = event.lower()
            if any(x in event_lower for x in ['dog', 'cat', 'rooster', 'cow', 'pig', 'crow', 'frog', 'bird']):
                categories['animal_sounds'].append(event)
            elif any(x in event_lower for x in ['rain', 'wind', 'thunder', 'fire', 'crickets']):
                categories['environmental'].append(event)
            elif any(x in event_lower for x in ['speech', 'applause', 'cheering']):
                categories['human_made'].append(event)
            elif any(x in event_lower for x in ['car', 'motorcycle', 'helicopter', 'aircraft', 'horn', 'siren']):
                categories['vehicles'].append(event)
            else:
                categories['other'].append(event)
        return {k: v for k, v in categories.items() if v}

    def detect_industrial_urban_sounds(self,
                                       audio: np.ndarray,
                                       sr: int = 16000) -> List[Dict]:
        detection_result = self.detect(audio, sr)
        events = detection_result.get('events', [])
        confidences = detection_result.get('confidences', [])
        industrial_urban_map = {
            'Traffic': ['Motor vehicle', 'Car horn', 'Passing by motorcycle', 'Siren'],
            'Construction': ['Explosion', 'Impact', 'Machinery'],
            'Machinery': ['Mechanical sounds', 'Industrial', 'Grinding', 'Metal'],
            'Emergency Siren': ['Siren', 'Alarm', 'Alert'],
            'Background Hum': ['Electrical hum', 'Buzzing', 'Drone']
        }
        detected = []
        for event, confidence in zip(events, confidences):
            event_lower = event.lower()
            for category, keywords in industrial_urban_map.items():
                if any(kw.lower() in event_lower for kw in keywords):
                    detected.append({
                        'event': category,
                        'original_event': event,
                        'confidence': confidence
                    })
                    break
        return detected

"""
Module 2: Speech Recognition Module

Purpose:
    Extract spoken language content from the audio.

Working:
    - Uses pretrained speech-to-text model
    - Converts spoken audio into textual form
    - Supports multilingual input (Hindi & English)

Model Used:
    - OpenAI Whisper (pretrained, local)

Output:
    - Transcribed speech text
"""

import whisper
import numpy as np
from typing import Optional, Dict
import warnings


class SpeechRecognizer:
    """
    Performs speech-to-text transcription using OpenAI Whisper.
    """

    def __init__(self,
                 model_size: str = 'base',
                 language: Optional[str] = None,
                 device: str = 'cpu'):
        self.model_size = model_size
        self.language = language
        self.device = device
        try:
            self.model = whisper.load_model(model_size, device=device)
        except Exception as e:
            raise RuntimeError(f"Error loading Whisper model: {str(e)}")

    def transcribe(self,
                   audio: np.ndarray,
                   sr: int = 16000,
                   temperature: float = 0.0,
                   verbose: bool = False) -> Dict:
        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.model.transcribe(
                    audio,
                    language=self.language,
                    temperature=temperature,
                    verbose=verbose
                )
            transcript = result.get('text', '').strip()
            detected_language = result.get('language', 'unknown')
            segments = result.get('segments', [])
            if segments:
                probs = [seg.get('confidence', 0.0) for seg in segments if 'confidence' in seg]
                confidence = float(np.mean(probs)) if probs else 0.5
            else:
                confidence = 0.5
            return {
                'text': transcript,
                'language': detected_language,
                'confidence_score': confidence,
                'segments': segments,
                'model_size': self.model_size
            }
        except Exception as e:
            return {
                'text': '',
                'language': 'unknown',
                'confidence_score': 0.0,
                'segments': [],
                'error': str(e)
            }

    def transcribe_multilingual(self,
                                audio: np.ndarray,
                                sr: int = 16000,
                                target_lang: Optional[str] = None) -> Dict:
        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.model.transcribe(
                    audio,
                    language=target_lang,
                    verbose=False
                )
            transcript = result.get('text', '').strip()
            detected_lang = result.get('language', 'unknown')
            code_switching = self._detect_code_switching(transcript)
            return {
                'text': transcript,
                'language_detected': detected_lang,
                'target_language': target_lang,
                'code_switching_detected': code_switching,
                'confidence_score': 0.85
            }
        except Exception as e:
            return {
                'text': '',
                'language_detected': 'unknown',
                'error': str(e)
            }

    def _detect_code_switching(self, text: str) -> bool:
        english_pattern = any(ord(c) < 128 for c in text if c.isalpha())
        non_latin = any(ord(c) > 127 for c in text)
        return english_pattern and non_latin

    def transcribe_file(self,
                        file_path: str,
                        verbose: bool = False) -> Dict:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.model.transcribe(
                    file_path,
                    language=self.language,
                    verbose=verbose
                )
            transcript = result.get('text', '').strip()
            detected_language = result.get('language', 'unknown')
            segments = result.get('segments', [])
            return {
                'text': transcript,
                'language': detected_language,
                'confidence_score': 0.5,
                'segments': segments,
                'model_size': self.model_size
            }
        except Exception as e:
            return {
                'text': '',
                'language': 'unknown',
                'confidence_score': 0.0,
                'segments': [],
                'error': str(e)
            }

    def get_segments_with_timestamps(self, result: Dict) -> list:
        segments = result.get('segments', [])
        formatted = []
        for seg in segments:
            formatted.append({
                'start': seg.get('start', 0),
                'end': seg.get('end', 0),
                'text': seg.get('text', '').strip()
            })
        return formatted

    def detect_language(self,
                        audio: np.ndarray,
                        sr: int = 16000) -> str:
        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.model.transcribe(audio, language=None, verbose=False)
            return result.get('language', 'unknown')
        except Exception:
            return 'unknown'

    def get_model_info(self) -> Dict:
        return {
            'model_size': self.model_size,
            'device': self.device,
            'language': self.language,
            'supports_languages': ['en', 'hi', 'fr', 'es', 'de', 'zh', 'ja', 'ko', 'ar', 'pt']
        }

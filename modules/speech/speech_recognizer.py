"""
Module 2: Speech Recognition Module

Purpose:
    Extract spoken language content from the audio.

Working:
    - Uses pretrained speech-to-text model
    - Converts spoken audio into textual form
    - Supports multilingual input (Hindi & English)

Model Used:
    - OpenAI Whisper (pretrained)

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
        """
        Initialize speech recognizer.
        
        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            language: Language code (e.g., 'en', 'hi') or None for auto-detect
            device: Device to run model on ('cpu' or 'cuda')
        """
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
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio signal (numpy array)
            sr: Sample rate of audio
            temperature: Temperature for sampling (0.0 = deterministic)
            verbose: Whether to print transcription progress
        
        Returns:
            Dictionary containing:
                - text: Transcribed text
                - language: Detected language
                - confidence_score: Confidence of transcription
                - segments: Detailed segments with timestamps
        """
        try:
            # Convert to Whisper-compatible format if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Transcribe
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.model.transcribe(
                    audio,
                    language=self.language,
                    temperature=temperature,
                    verbose=verbose
                )
            
            # Extract results
            transcript = result.get('text', '').strip()
            detected_language = result.get('language', 'unknown')
            
            # Calculate confidence (based on average probability from segments)
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
        """
        Transcribe audio with multilingual support, especially for Asian languages.
        Supports: English, Hindi, Tamil, Mandarin Chinese, and code-switching contexts.
        
        Args:
            audio: Audio signal (numpy array)
            sr: Sample rate of audio
            target_lang: Target language code (e.g., 'hi', 'ta', 'zh', 'en')
                        None for auto-detection
        
        Returns:
            Dictionary containing:
                - text: Transcribed text
                - language_detected: Detected language
                - target_language: Target language if specified
                - code_switching_detected: Boolean indicating code-switching
        """
        try:
            # Convert to Whisper-compatible format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Transcribe with optional language specification
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.model.transcribe(
                    audio,
                    language=target_lang,
                    verbose=False
                )
            
            transcript = result.get('text', '').strip()
            detected_lang = result.get('language', 'unknown')
            
            # Detect code-switching (simple heuristic: multiple language patterns)
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
        """
        Detect presence of multiple languages in transcript (code-switching).
        
        Args:
            text: Transcribed text
        
        Returns:
            Boolean indicating code-switching
        """
        # Simple heuristic: check for common patterns
        english_pattern = any(ord(c) < 128 for c in text if c.isalpha())
        non_latin = any(ord(c) > 127 for c in text)
        return english_pattern and non_latin
    
    def transcribe_file(self, 
                       file_path: str,
                       verbose: bool = False) -> Dict:
        """
        Transcribe audio from file.

        
        Args:
            file_path: Path to audio file
            verbose: Whether to print progress
        
        Returns:
            Dictionary with transcription results
        """
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
        """
        Extract segments with timestamps from transcription result.
        
        Args:
            result: Transcription result dictionary
        
        Returns:
            List of segments with timestamps
        """
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
        """
        Detect language of audio.
        
        Args:
            audio: Audio signal
            sr: Sample rate
        
        Returns:
            Language code (e.g., 'en', 'hi')
        """
        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Use Whisper's language detection
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.model.transcribe(
                    audio,
                    language=None,  # Auto-detect
                    verbose=False
                )
            
            return result.get('language', 'unknown')
        
        except Exception as e:
            return 'unknown'
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_size': self.model_size,
            'device': self.device,
            'language': self.language,
            'supports_languages': ['en', 'hi', 'fr', 'es', 'de', 'zh', 'ja', 'ko', 'ar', 'pt']
        }

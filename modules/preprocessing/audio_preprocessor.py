"""
Module 1: Audio Input & Preprocessing

Purpose:
    Standardize and prepare audio input for downstream models.

Working:
    - Accepts uploaded audio files (WAV/MP3)
    - Converts audio to mono
    - Resamples to 16 kHz
    - Normalizes amplitude

Tools Used:
    - Python
    - Librosa
    - Pydub
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import os


class AudioPreprocessor:
    """
    Handles audio loading, preprocessing, and standardization.
    """

    def __init__(self,
                 target_sr: int = 16000,
                 mono: bool = True,
                 normalize: bool = True,
                 n_mels: int = 80,
                 n_fft: int = 400,
                 hop_length: int = 160):
        self.target_sr = target_sr
        self.mono = mono
        self.normalize = normalize
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def process(self, audio_path: str) -> dict:
        y, _ = librosa.load(audio_path, sr=self.target_sr)
        y = librosa.util.normalize(y)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.target_sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        duration = librosa.get_duration(y=y, sr=self.target_sr)
        chunks = []
        for i in range(0, int(duration), 30):
            start = i * self.target_sr
            end = min((i + 30) * self.target_sr, len(y))
            chunks.append(y[start:end])
        return {"raw": y, "mel": mel_db, "chunks": chunks, "sr": self.target_sr}

    def load_audio(self,
                   file_path: str,
                   duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
            raise ValueError(f"Unsupported audio format: {file_ext}")
        try:
            audio, sr = librosa.load(
                file_path,
                sr=self.target_sr,
                mono=self.mono,
                duration=duration
            )
        except Exception as e:
            raise ValueError(f"Error loading audio file: {str(e)}")
        audio = self._preprocess(audio)
        return audio, self.target_sr

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        if self.normalize:
            audio = self._normalize_amplitude(audio)
        audio = self._trim_silence(audio)
        return audio

    def _normalize_amplitude(self,
                              audio: np.ndarray,
                              target_level: float = -3.0) -> np.ndarray:
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return audio
        target_linear = 10 ** (target_level / 20)
        audio = audio * (target_linear / rms)
        audio = np.clip(audio, -1.0, 1.0)
        return audio

    def _trim_silence(self,
                      audio: np.ndarray,
                      top_db: float = 40) -> np.ndarray:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed

    def get_audio_duration(self, audio: np.ndarray) -> float:
        return len(audio) / self.target_sr

    def get_audio_info(self, audio: np.ndarray) -> dict:
        return {
            'sample_rate': self.target_sr,
            'duration_seconds': self.get_audio_duration(audio),
            'num_samples': len(audio),
            'rms_energy': float(np.sqrt(np.mean(audio ** 2))),
            'peak_amplitude': float(np.max(np.abs(audio))),
            'mono': self.mono
        }

    def save_audio(self, audio: np.ndarray, output_path: str) -> None:
        try:
            sf.write(output_path, audio, self.target_sr)
        except Exception as e:
            raise ValueError(f"Error saving audio file: {str(e)}")

    def chunk_audio(self, audio: np.ndarray, chunk_duration: float) -> list:
        chunk_samples = int(chunk_duration * self.target_sr)
        chunks = []
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) > 0:
                chunks.append(chunk)
        return chunks

    def compute_features(self, audio: np.ndarray) -> dict:
        mfcc = librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(audio)
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.target_sr)
        return {
            'mfcc_mean': float(np.mean(mfcc)),
            'mfcc_std': float(np.std(mfcc)),
            'zcr_mean': float(np.mean(zcr)),
            'spec_centroid_mean': float(np.mean(spec_centroid))
        }

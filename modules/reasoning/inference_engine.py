"""
Module 6: Audio Environment Analysis & Reasoning

Purpose:
    Analyze and understand audio environment characteristics, identify noise sources,
    and provide contextual assessment of acoustic conditions.

Working:
    - Performs spectral and temporal analysis of audio
    - Classifies noise types using frequency signatures
    - Analyzes background noise floor and SNR
    - Infers spatial and environmental characteristics
    - Generates comprehensive audio analysis reports

Output:
    - AudioEnvironmentProfile with detailed acoustic analysis
    - Natural language report of findings
"""

import librosa
import numpy as np
import scipy.signal as signal
from scipy.fft import fft
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NOISE CLASSIFICATION SYSTEM
# ============================================================================

class NoiseCategory(Enum):
    """Hierarchical noise categorization"""
    AMBIENT = "ambient_background"
    MACHINERY = "industrial_mechanical"
    TRAFFIC = "transportation"
    SPEECH = "human_vocal"
    ENVIRONMENTAL = "natural_elements"
    STRUCTURAL = "building_related"
    UNIDENTIFIED = "unknown_source"

@dataclass
class NoiseSignature:
    """Detailed noise characteristics"""
    category: NoiseCategory
    frequency_range: Tuple[float, float]
    temporal_pattern: str
    intensity_level: float
    confidence: float
    harmonic_content: List[float]
    spectral_shape: str

@dataclass
class AudioEnvironmentProfile:
    """Complete environmental analysis"""
    dominant_noises: List[NoiseSignature]
    background_noise_floor: float
    signal_to_noise_ratio: float
    acoustic_complexity: float
    spatial_characteristics: Dict[str, Any]
    environmental_context: str
    risk_factors: List[str]
    quality_assessment: Dict[str, float]

# ============================================================================
# CORE ANALYSIS ENGINE
# ============================================================================

class InferenceEngine:
    """
    Advanced audio environment understanding system.
    Identifies noise types, background characteristics, and contextual information.
    """
    
    def __init__(self, sr: int = 16000, n_fft: int = 2048, hop_length: int = 512, 
                 api_key: str = None):
        """
        Initialize audio environment analyzer.
        
        Args:
            sr: Sample rate (Hz)
            n_fft: FFT window size
            hop_length: Hop length for STFT
            api_key: Optional API key for extended reasoning (future use)
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.api_key = api_key
        self.fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Ollama attributes (initialized in setup_ollama)
        self.ollama_model = None
        self.ollama_base_url = None
        self.ollama_client = None
        
        # Noise signatures database
        self.noise_signatures = self._initialize_noise_database()
        
    def _initialize_noise_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive noise signature database"""
        return {
            # Traffic & Transportation
            'car_engine': {
                'freq_range': (100, 500),
                'harmonics': [150, 300, 450],
                'temporal': 'continuous_variable',
                'pattern_type': 'engine_rumble'
            },
            'horn_beep': {
                'freq_range': (800, 1200),
                'harmonics': [1000],
                'temporal': 'impulsive_short',
                'pattern_type': 'sharp_transient'
            },
            'tire_screech': {
                'freq_range': (1500, 3000),
                'harmonics': [2000, 2500],
                'temporal': 'burst_sustained',
                'pattern_type': 'friction_noise'
            },
            
            # Industrial/Machinery
            'machinery_rumble': {
                'freq_range': (50, 300),
                'harmonics': [100, 200, 300],
                'temporal': 'continuous_rhythmic',
                'pattern_type': 'mechanical_vibration'
            },
            'metal_grinding': {
                'freq_range': (2000, 5000),
                'harmonics': [3000, 4000],
                'temporal': 'sustained_harsh',
                'pattern_type': 'friction_grinding'
            },
            
            # Environmental
            'wind_noise': {
                'freq_range': (100, 1000),
                'harmonics': [],
                'temporal': 'continuous_turbulent',
                'pattern_type': 'broadband_noise'
            },
            'rain': {
                'freq_range': (2000, 8000),
                'harmonics': [],
                'temporal': 'continuous_random',
                'pattern_type': 'rainfall_patter'
            },
            'thunder': {
                'freq_range': (20, 200),
                'harmonics': [50, 100],
                'temporal': 'impulsive_booming',
                'pattern_type': 'low_frequency_burst'
            },
            
            # Urban/Structural
            'construction': {
                'freq_range': (500, 2000),
                'harmonics': [1000],
                'temporal': 'irregular_impulsive',
                'pattern_type': 'impact_noise'
            },
            'door_slam': {
                'freq_range': (200, 800),
                'harmonics': [400, 600],
                'temporal': 'sharp_transient',
                'pattern_type': 'impact_resonance'
            },
            'background_hum': {
                'freq_range': (50, 60),
                'harmonics': [50, 100, 150],
                'temporal': 'continuous_steady',
                'pattern_type': 'electrical_hum'
            }
        }
    
    def analyze(self, audio: np.ndarray, sr: int = None) -> Dict[str, Any]:
        """
        Public interface method for audio environment analysis.
        
        Args:
            audio: Audio signal (numpy array)
            sr: Sample rate (defaults to initialized sr)
        
        Returns:
            Dictionary containing analysis results:
                - profile: AudioEnvironmentProfile object
                - report: Natural language analysis report
                - json_export: Exportable JSON data
        """
        if sr is None:
            sr = self.sr
        
        # Perform analysis
        profile = self.analyze_audio_signal(audio, sr)
        
        # Generate report
        reporter = AudioAnalysisReporter()
        report = reporter.generate_report(profile)
        
        # Create JSON export
        json_export = {
            'environmental_context': profile.environmental_context,
            'dominant_noises': [
                {
                    'category': n.category.value,
                    'frequency_range': n.frequency_range,
                    'temporal_pattern': n.temporal_pattern,
                    'intensity': n.intensity_level,
                    'confidence': n.confidence
                }
                for n in profile.dominant_noises
            ],
            'acoustic_measurements': {
                'background_noise_floor_db': profile.background_noise_floor,
                'signal_to_noise_ratio_db': profile.signal_to_noise_ratio,
                'acoustic_complexity': profile.acoustic_complexity
            },
            'spatial_characteristics': profile.spatial_characteristics,
            'quality_assessment': profile.quality_assessment,
            'risk_factors': profile.risk_factors
        }
        
        return {
            'profile': profile,
            'report': report,
            'json_export': json_export
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about inference engine.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'Audio Environment Analysis Engine',
            'method': 'Multi-domain spectral and temporal analysis',
            'sample_rate': self.sr,
            'fft_window_size': self.n_fft,
            'hop_length': self.hop_length,
            'noise_categories': [cat.value for cat in NoiseCategory],
            'analysis_modules': [
                'Spectral Feature Extraction',
                'Temporal Pattern Analysis',
                'Noise Classification',
                'Background Noise Analysis',
                'Spatial Characteristics Inference',
                'Audio Quality Assessment'
            ],
            'reasoning_pipeline': '4-Stage (Planning → Captioning → Reasoning → Summarization)'
        }
    
    def perform_reasoning(self, speech_data: Dict, sound_data: List[Dict], 
                         emotion_data: Dict) -> Dict[str, Any]:
        """
        Perform 4-stage audio reasoning pipeline: Planning, Captioning, Reasoning, Summarizing.
        Synthesizes verbal content with acoustic environment for comprehensive context understanding.
        
        Args:
            speech_data: Speech transcription and language info
            sound_data: Detected sound events with confidence
            emotion_data: Paralinguistic and emotional analysis
        
        Returns:
            Dictionary containing:
                - plan: Planning stage output (what to analyze)
                - caption: Acoustic description
                - reasoning_steps: Chain-of-thought logic
                - final_summary: Comprehensive context summary
        """
        # Stage 1: Planning - Determine analysis needs
        plan = self._stage_planning(speech_data, sound_data, emotion_data)
        
        # Stage 2: Captioning - Describe acoustic environment
        caption = self._stage_captioning(speech_data, sound_data, emotion_data)
        
        # Stage 3: Logical Reasoning - Chain-of-thought analysis
        reasoning_steps = self._stage_reasoning(speech_data, sound_data, emotion_data)
        
        # Stage 4: Summarization - Final context synthesis
        final_summary = self._stage_summarization(speech_data, sound_data, emotion_data, reasoning_steps)
        
        return {
            'plan': plan,
            'caption': caption,
            'reasoning_steps': reasoning_steps,
            'final_summary': final_summary
        }
    
    def generate_inference(self, context: Dict) -> str:
        """
        Generate inference text from integrated audio context.
        Wrapper method that extracts components and performs reasoning.
        
        Args:
            context: Integrated context dictionary containing:
                - speech: Transcribed text
                - sounds: List of detected sound events
                - emotion: Emotional state
                - Additional optional context fields
        
        Returns:
            Formatted natural language inference text
        """
        # Extract components from integrated context
        speech_data = {
            'text': context.get('speech', 'No speech detected'),
            'language_detected': context.get('language_detected', 'unknown')
        }
        
        # Convert sound events to proper format
        sounds_raw = context.get('sounds', [])
        sound_data = []
        if isinstance(sounds_raw, list):
            for sound in sounds_raw:
                if isinstance(sound, str):
                    sound_data.append({'event': sound, 'confidence': 0.7})
                elif isinstance(sound, dict):
                    sound_data.append(sound)
        
        # Extract emotional data
        emotion_data = {
            'emotional_state': context.get('emotion', 'Neutral'),
            'vocal_tension': context.get('vocal_tension', 'Unknown')
        }
        
        # Perform reasoning
        reasoning_result = self.perform_reasoning(speech_data, sound_data, emotion_data)
        
        # Format as readable text
        inference_text = self._format_inference_output(reasoning_result)
        
        return inference_text
    
    def _format_inference_output(self, reasoning_result: Dict) -> str:
        """
        Format reasoning result into readable inference text.
        
        Args:
            reasoning_result: Output from perform_reasoning()
        
        Returns:
            Formatted text string
        """
        output = []
        output.append("=" * 70)
        output.append("AUDIO REASONING & INFERENCE ANALYSIS")
        output.append("=" * 70)
        output.append("")
        
        # Planning stage
        output.append("📋 ANALYSIS PLAN:")
        output.append(f"   {reasoning_result.get('plan', 'N/A')}")
        output.append("")
        
        # Caption stage
        output.append("🎯 ACOUSTIC SCENE CAPTION:")
        output.append(f"   {reasoning_result.get('caption', 'N/A')}")
        output.append("")
        
        # Reasoning steps
        output.append("🧠 LOGICAL REASONING CHAIN:")
        steps = reasoning_result.get('reasoning_steps', [])
        for i, step in enumerate(steps, 1):
            output.append(f"   [{i}] {step}")
        output.append("")
        
        # Final summary
        output.append("✅ FINAL INFERENCE SUMMARY:")
        output.append(f"   {reasoning_result.get('final_summary', 'N/A')}")
        output.append("")
        
        output.append("=" * 70)
        
        return "\n".join(output)
    
    
    def _stage_planning(self, speech_data: Dict, sound_data: List[Dict], 
                       emotion_data: Dict) -> str:
        """Stage 1: Planning - Identify analysis objectives"""
        return "Synthesize verbal content with acoustic environment to determine location, context, and urgency."
    
    def _stage_captioning(self, speech_data: Dict, sound_data: List[Dict], 
                         emotion_data: Dict) -> str:
        """Stage 2: Captioning - Natural language description of acoustic scene"""
        speech_text = speech_data.get('text', 'No speech')
        language = speech_data.get('language_detected', 'unknown')
        
        # Get primary sound event
        sound_desc = "quiet environment"
        if sound_data:
            primary_sound = sound_data[0].get('event', sound_data[0].get('original_event', 'Unknown')) if isinstance(sound_data[0], dict) else sound_data[0]
            sound_desc = f"{primary_sound} environment"
        
        # Get emotional tone
        emotion_state = emotion_data.get('emotional_state', 'neutral')
        
        caption = f"Speaker (detected language: {language}) in {sound_desc} with {emotion_state} tone. "
        caption += f"Speech: '{speech_text[:100]}...'" if len(speech_text) > 100 else f"Speech: '{speech_text}'"
        
        return caption
    
    def _stage_reasoning(self, speech_data: Dict, sound_data: List[Dict], 
                        emotion_data: Dict) -> List[str]:
        """Stage 3: Reasoning - Chain-of-thought logical steps"""
        steps = []
        
        # Reasoning step 1: Speech content
        speech_text = speech_data.get('text', 'No speech detected')
        steps.append(f"Speaker content: '{speech_text}'")
        
        # Reasoning step 2: Environmental context
        if sound_data:
            events = [s.get('event') if isinstance(s, dict) else s for s in sound_data[:3]]
            events_str = ', '.join(events) if events else 'Quiet'
            steps.append(f"Background environment characterized by: {events_str}")
        else:
            steps.append("Background environment: Quiet/Clean audio")
        
        # Reasoning step 3: Emotional/Paralinguistic cues
        vocal_tension = emotion_data.get('vocal_tension', 'Unknown')
        emotional_state = emotion_data.get('emotional_state', 'Neutral')
        steps.append(f"Vocal characteristics: {emotional_state} (tension: {vocal_tension})")
        
        # Reasoning step 4: Context inference
        if emotional_state == "Urgent/Stressed" and sound_data:
            steps.append("Context inference: High urgency communication in complex acoustic environment")
        elif emotional_state in ["Calm/Neutral", "Normal"]:
            steps.append("Context inference: Routine communication in controlled environment")
        else:
            steps.append(f"Context inference: {emotional_state} communication with background activity")
        
        return steps
    
    def _stage_summarization(self, speech_data: Dict, sound_data: List[Dict], 
                            emotion_data: Dict, reasoning_steps: List[str]) -> str:
        """Stage 4: Summarization - Comprehensive final output"""
        speech_text = speech_data.get('text', 'No speech')
        language = speech_data.get('language_detected', 'unknown')
        emotional_state = emotion_data.get('emotional_state', 'Neutral')
        
        # Determine environment type
        environment_id = "Quiet Room"
        if sound_data:
            environment_id = sound_data[0].get('event') if isinstance(sound_data[0], dict) else sound_data[0]
        
        # Build comprehensive summary
        summary = f"Context: {emotional_state} interaction in {environment_id}. "
        summary += f"Language: {language}. "
        summary += f"Summary: {speech_text}"
        
        return summary
    
    def analyze_audio_file(self, audio_path: str) -> AudioEnvironmentProfile:
        """
        Complete analysis pipeline for audio files
        """
        y, sr = librosa.load(audio_path, sr=self.sr)
        return self.analyze_audio_signal(y, sr)
    
    def analyze_audio_signal(self, y: np.ndarray, sr: int = None) -> AudioEnvironmentProfile:
        """
        Analyze raw audio signal for environment understanding
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            AudioEnvironmentProfile with comprehensive analysis
        """
        if sr is None:
            sr = self.sr
        
        # Extract multi-domain features
        spectral_features = self._extract_spectral_features(y, sr)
        temporal_features = self._extract_temporal_features(y, sr)
        noise_analysis = self._perform_noise_analysis(y, sr, spectral_features)
        background_profile = self._analyze_background_noise(y, sr)
        spatial_info = self._infer_spatial_characteristics(y, sr)
        
        # Generate context-aware assessment
        context = self._determine_environmental_context(
            noise_analysis, spectral_features, temporal_features, spatial_info
        )
        
        # Compile findings
        profile = AudioEnvironmentProfile(
            dominant_noises=noise_analysis['dominant_noises'],
            background_noise_floor=background_profile['noise_floor'],
            signal_to_noise_ratio=background_profile['snr'],
            acoustic_complexity=self._calculate_acoustic_complexity(spectral_features),
            spatial_characteristics=spatial_info,
            environmental_context=context,
            risk_factors=self._identify_risk_factors(noise_analysis, background_profile),
            quality_assessment=self._assess_audio_quality(y, sr, spectral_features)
        )
        
        return profile
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract comprehensive spectral characteristics"""
        # Compute STFT
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        
        # Power spectrogram
        S = librosa.power_to_db(magnitude**2, ref=np.max)
        
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Spectral rolloff (frequency cutoff)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Zero crossing rate (high-frequency content indicator)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # MFCC (perceptual representation)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return {
            'magnitude': magnitude,
            'power_db': S,
            'centroid': centroid,
            'rolloff': rolloff,
            'zcr': zcr,
            'mfcc': mfcc,
            'contrast': contrast,
            'mel_db': mel_db,
            'freqs': self.fft_freqs
        }
    
    def _extract_temporal_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze temporal dynamics and patterns"""
        # Onset detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_strength=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onsets, sr=sr)
        
        # Tempogram
        tempogram = librosa.feature.tempogram(onset_strength=onset_env, sr=sr)
        
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        rms_db = librosa.power_to_db(rms**2)
        
        # Energy flux (change over time)
        energy_flux = np.diff(rms)
        
        # Zero crossing rate temporal evolution
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        return {
            'onset_times': onset_times,
            'onset_strength': onset_env,
            'tempogram': tempogram,
            'rms': rms,
            'rms_db': rms_db,
            'energy_flux': energy_flux,
            'zcr_temporal': zcr,
            'attack_time': self._estimate_attack_time(rms),
            'decay_time': self._estimate_decay_time(rms)
        }
    
    def _perform_noise_analysis(self, y: np.ndarray, sr: int, 
                               spectral_features: Dict) -> Dict[str, Any]:
        """Identify and classify noise sources"""
        magnitude = spectral_features['magnitude']
        freqs = spectral_features['freqs']
        
        # Identify peaks in frequency spectrum
        avg_magnitude = np.mean(magnitude, axis=1)
        peaks, properties = signal.find_peaks(avg_magnitude, 
                                            height=np.percentile(avg_magnitude, 70),
                                            distance=5)
        
        dominant_frequencies = freqs[peaks]
        dominant_magnitudes = avg_magnitude[peaks]
        
        # Classify detected noises
        detected_noises = []
        for freq, mag in sorted(zip(dominant_frequencies, dominant_magnitudes), 
                               key=lambda x: x[1], reverse=True)[:5]:
            noise_sig = self._classify_noise_by_frequency(freq, mag, y, sr)
            if noise_sig.confidence > 0.3:
                detected_noises.append(noise_sig)
        
        return {
            'dominant_noises': detected_noises,
            'dominant_frequencies': dominant_frequencies[:10],
            'spectral_peaks': properties['peak_heights'][:10] if len(properties['peak_heights']) > 0 else []
        }
    
    def _classify_noise_by_frequency(self, freq: float, magnitude: float, 
                                     y: np.ndarray, sr: int) -> NoiseSignature:
        """Classify noise based on frequency characteristics"""
        
        # Determine category and confidence
        if freq < 100:
            category = NoiseCategory.MACHINERY
            name = "Low-frequency machinery or hum"
            confidence = 0.7 if self._has_harmonic_structure(y, sr, freq) else 0.5
        elif freq < 500:
            category = NoiseCategory.TRAFFIC
            name = "Engine or vehicle noise"
            confidence = 0.6
        elif freq < 1500:
            category = NoiseCategory.SPEECH
            name = "Human speech frequencies"
            confidence = 0.75 if self._detect_speech_modulation(y, sr) else 0.4
        elif freq < 3000:
            category = NoiseCategory.ENVIRONMENTAL
            name = "Wind, rain, or environmental sound"
            confidence = 0.65
        else:
            category = NoiseCategory.STRUCTURAL
            name = "High-frequency impact or friction"
            confidence = 0.6
        
        # Harmonic content
        harmonics = self._extract_harmonics(y, sr, freq)
        
        return NoiseSignature(
            category=category,
            frequency_range=(max(0, freq - 200), freq + 200),
            temporal_pattern=self._analyze_temporal_pattern(y, sr, freq),
            intensity_level=float(magnitude),
            confidence=confidence,
            harmonic_content=harmonics,
            spectral_shape=self._characterize_spectral_shape(y, sr, freq)
        )
    
    def _has_harmonic_structure(self, y: np.ndarray, sr: int, fundamental: float) -> bool:
        """Detect presence of harmonic overtones"""
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        
        # Check for harmonics at 2f, 3f, 4f
        harmonic_energy = 0
        for harmonic in [2, 3, 4]:
            target_freq = fundamental * harmonic
            idx = np.argmin(np.abs(freqs - target_freq))
            harmonic_energy += np.mean(magnitude[idx, :])
        
        fundamental_idx = np.argmin(np.abs(freqs - fundamental))
        fundamental_energy = np.mean(magnitude[fundamental_idx, :])
        
        return harmonic_energy > (0.3 * fundamental_energy)
    
    def _detect_speech_modulation(self, y: np.ndarray, sr: int) -> bool:
        """Detect characteristic speech amplitude modulation"""
        # Speech typically has 4-8 Hz modulation
        rms = librosa.feature.rms(y=y)[0]
        
        # Analyze modulation frequency
        freqs_mod = np.fft.fftfreq(len(rms), d=1/(sr/512))
        magnitude_mod = np.abs(np.fft.fft(rms))
        
        # Look for peak in 4-8 Hz range
        speech_band = (freqs_mod > 4) & (freqs_mod < 8)
        if np.any(speech_band) and np.max(magnitude_mod[speech_band]) > np.percentile(magnitude_mod, 70):
            return True
        return False
    
    def _extract_harmonics(self, y: np.ndarray, sr: int, fundamental: float) -> List[float]:
        """Extract harmonic content"""
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        
        harmonics = []
        for harmonic in range(1, 6):
            target_freq = fundamental * harmonic
            if target_freq < sr / 2:
                idx = np.argmin(np.abs(freqs - target_freq))
                harmonics.append(float(np.mean(magnitude[idx, :])))
        
        return harmonics
    
    def _analyze_temporal_pattern(self, y: np.ndarray, sr: int, freq: float) -> str:
        """Characterize how the noise evolves over time"""
        # Band-pass filter around frequency
        sos = signal.butter(4, [freq - 100, freq + 100], btype='band', 
                           fs=sr, output='sos')
        filtered = signal.sosfilt(sos, y)
        
        # Analyze envelope
        analytic = signal.hilbert(filtered)
        envelope = np.abs(analytic)
        
        # Characteristics
        envelope_mean = np.mean(envelope)
        envelope_std = np.std(envelope)
        variability = envelope_std / (envelope_mean + 1e-8)
        
        # Classify pattern
        if variability < 0.3:
            return "continuous_steady"
        elif variability < 0.6:
            return "continuous_variable"
        elif variability < 1.0:
            return "intermittent_bursty"
        else:
            return "impulsive_sporadic"
    
    def _characterize_spectral_shape(self, y: np.ndarray, sr: int, center_freq: float) -> str:
        """Describe spectral envelope shape"""
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        
        # Analyze around center frequency
        mask = (freqs > center_freq - 500) & (freqs < center_freq + 500)
        local_spectrum = magnitude[mask, :].mean(axis=1)
        
        # Shape analysis
        if np.std(local_spectrum) / (np.mean(local_spectrum) + 1e-8) < 0.3:
            return "flat_broadband"
        elif np.argmax(local_spectrum) < len(local_spectrum) // 2:
            return "rising_highpass"
        elif np.argmax(local_spectrum) > len(local_spectrum) // 2:
            return "falling_lowpass"
        else:
            return "peaked_narrowband"
    
    def _analyze_background_noise(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Detailed background noise characterization"""
        # Estimate noise floor using spectral subtraction
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        
        # Noise floor is typically lower 10% of spectrum
        noise_floor_db = np.percentile(librosa.power_to_db(magnitude**2, ref=np.max), 10)
        
        # Signal power
        signal_power = np.mean(librosa.power_to_db(magnitude**2, ref=np.max))
        
        # SNR calculation
        snr = signal_power - noise_floor_db
        
        return {
            'noise_floor': float(noise_floor_db),
            'signal_power': float(signal_power),
            'snr': float(snr),
            'dynamic_range': float(signal_power - noise_floor_db)
        }
    
    def _infer_spatial_characteristics(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Infer room characteristics and spatial properties"""
        # Estimate reverberation through decay analysis
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Energy decay curve
        energy = np.mean(S_db, axis=0)
        decay_slope = np.polyfit(range(len(energy)), energy, 1)[0]
        
        # Estimate RT60 (reverberation time)
        rt60 = self._estimate_rt60(y, sr) if len(y) > sr else 0
        
        return {
            'reverberation_time_estimated_ms': float(rt60 * 1000),
            'decay_slope': float(decay_slope),
            'acoustic_environment': self._classify_acoustic_environment(rt60, decay_slope),
            'room_size_estimate': self._estimate_room_size(rt60),
            'echo_presence': self._detect_echo(y, sr)
        }
    
    def _estimate_rt60(self, y: np.ndarray, sr: int, freq_band: Tuple[int, int] = (100, 2000)) -> float:
        """Estimate reverberation time (RT60)"""
        # Band-pass filter
        sos = signal.butter(4, freq_band, btype='band', fs=sr, output='sos')
        filtered = signal.sosfilt(sos, y)
        
        # Energy envelope
        rms = librosa.feature.rms(y=filtered)[0]
        rms_db = librosa.power_to_db(rms**2, ref=np.max)
        
        # Find where it drops 60dB (RT60)
        if len(rms_db) > 100:
            peak_idx = np.argmax(rms_db)
            peak_level = rms_db[peak_idx]
            
            # Find point 60dB below peak
            target_level = peak_level - 60
            indices = np.where(rms_db[peak_idx:] < target_level)[0]
            
            if len(indices) > 0:
                rt60_frames = indices[0]
                rt60_seconds = librosa.frames_to_time(rt60_frames, sr=sr, hop_length=512)
                return max(0.01, min(rt60_seconds, 10.0))  # Clamp between 10ms and 10s
        
        return 0.2  # Default estimate
    
    def _classify_acoustic_environment(self, rt60: float, decay_slope: float) -> str:
        """Classify environment based on acoustic characteristics"""
        if rt60 < 0.15:
            return "dead_anechoic"
        elif rt60 < 0.5:
            return "acoustic_treated"
        elif rt60 < 1.5:
            return "normal_office_room"
        elif rt60 < 3.0:
            return "large_room_hall"
        else:
            return "highly_reverberant"
    
    def _estimate_room_size(self, rt60: float) -> str:
        """Estimate room dimensions from RT60"""
        if rt60 < 0.3:
            return "very_small_close_talk"
        elif rt60 < 0.7:
            return "small_room_office"
        elif rt60 < 1.5:
            return "medium_room"
        elif rt60 < 3.0:
            return "large_room"
        else:
            return "very_large_space"
    
    def _detect_echo(self, y: np.ndarray, sr: int) -> bool:
        """Detect presence of discrete echo"""
        # Autocorrelation analysis
        correlation = np.correlate(y, y, mode='full')
        correlation = correlation[len(correlation)//2:]
        correlation_normalized = correlation / correlation[0]
        
        # Look for secondary peaks (echo)
        echo_region = correlation_normalized[sr//10:sr]  # 100ms to 1s
        if len(echo_region) > 0:
            max_secondary = np.max(echo_region)
            return max_secondary > 0.5
        return False
    
    def _calculate_acoustic_complexity(self, spectral_features: Dict) -> float:
        """Measure overall complexity of acoustic scene"""
        mel_db = spectral_features['mel_db']
        
        # Entropy-based complexity
        # Normalize to probability distribution
        mel_normalized = mel_db - np.min(mel_db)
        mel_normalized = mel_normalized / (np.max(mel_normalized) + 1e-8)
        
        # Calculate entropy across frequency bands
        entropy_freq = -np.sum(mel_normalized * np.log(mel_normalized + 1e-8), axis=0)
        entropy_time = -np.sum(mel_normalized * np.log(mel_normalized + 1e-8), axis=1)
        
        # Temporal variability
        temporal_var = np.std(np.diff(np.mean(mel_db, axis=0)))
        
        # Combined complexity score (0-1)
        complexity = (np.mean(entropy_freq) + np.mean(entropy_time) + temporal_var) / 30.0
        return float(np.clip(complexity, 0, 1))
    
    def _determine_environmental_context(self, noise_analysis: Dict, spectral_features: Dict,
                                        temporal_features: Dict, spatial_info: Dict) -> str:
        """Generate contextual interpretation of environment"""
        dominant_noises = noise_analysis['dominant_noises']
        
        if len(dominant_noises) == 0:
            return "Silent or very quiet environment"
        
        # Analyze context from dominant noises
        categories = [n.category for n in dominant_noises[:3]]
        
        context_map = {
            NoiseCategory.TRAFFIC: "Urban/Traffic environment",
            NoiseCategory.MACHINERY: "Industrial/Factory setting",
            NoiseCategory.SPEECH: "Social/Communication context",
            NoiseCategory.ENVIRONMENTAL: "Outdoor/Natural environment",
            NoiseCategory.AMBIENT: "General ambient background",
            NoiseCategory.STRUCTURAL: "Building/Indoor space"
        }
        
        # Combine categories
        if categories:
            primary = categories[0]
            context_text = context_map.get(primary, "General environment")
            
            # Add spatial information
            room_info = spatial_info['acoustic_environment']
            if room_info != "dead_anechoic":
                context_text += f" with {room_info} characteristics"
            
            # Add SNR information
            if noise_analysis.get('snr', 0) < 10:
                context_text += ", high background noise"
            
            return context_text
        
        return "Indeterminate environment"
    
    def _identify_risk_factors(self, noise_analysis: Dict, background_profile: Dict) -> List[str]:
        """Identify acoustic hazards and quality concerns"""
        risks = []
        
        # High noise exposure
        if background_profile.get('noise_floor', 0) > -20:
            risks.append("High background noise levels - potential hearing hazard")
        
        # Low SNR
        if background_profile.get('snr', 0) < 5:
            risks.append("Poor signal-to-noise ratio - difficulty understanding speech")
        
        # Dominant low frequencies (machinery)
        dominant_noises = noise_analysis.get('dominant_noises', [])
        if any(n.category == NoiseCategory.MACHINERY for n in dominant_noises):
            risks.append("Industrial noise presence - occupational exposure concern")
        
        # Impulsive noises
        if any(n.temporal_pattern == 'impulsive_short' for n in dominant_noises):
            risks.append("Impulsive noise events - sudden acoustic transients present")
        
        # High spectral complexity
        if len(dominant_noises) > 5:
            risks.append("Complex multi-source acoustic environment")
        
        return risks
    
    def _assess_audio_quality(self, y: np.ndarray, sr: int, 
                             spectral_features: Dict) -> Dict[str, float]:
        """Assess recording and environmental quality"""
        mel_db = spectral_features['mel_db']
        
        # Dynamic range
        dynamic_range = np.max(mel_db) - np.min(mel_db)
        
        # Clipping detection
        clipping_ratio = np.sum(np.abs(y) > 0.99) / len(y)
        
        # Noise uniformity (lower is better - indicates noise floor)
        noise_uniformity = np.std(np.min(mel_db, axis=0))
        
        # Spectral balance
        low_freq = np.mean(mel_db[:20, :])
        mid_freq = np.mean(mel_db[50:80, :])
        high_freq = np.mean(mel_db[100:, :])
        spectral_balance = 1.0 - (np.std([low_freq, mid_freq, high_freq]) / (np.mean([low_freq, mid_freq, high_freq]) + 1e-8))
        
        return {
            'dynamic_range_db': float(dynamic_range),
            'clipping_ratio': float(clipping_ratio),
            'noise_floor_uniformity': float(noise_uniformity),
            'spectral_balance_score': float(np.clip(spectral_balance, 0, 1))
        }
    
    def _estimate_attack_time(self, rms: np.ndarray) -> float:
        """Estimate sound attack time"""
        if len(rms) > 10:
            peak_idx = np.argmax(rms)
            if peak_idx > 5:
                attack = np.mean(np.diff(rms[:peak_idx]))
                return float(attack)
        return 0.0
    
    def _estimate_decay_time(self, rms: np.ndarray) -> float:
        """Estimate sound decay time"""
        if len(rms) > 10:
            peak_idx = np.argmax(rms)
            if peak_idx < len(rms) - 5:
                decay = np.mean(np.diff(rms[peak_idx:]))
                return float(decay)
        return 0.0
    
    # ========================================================================
    # OLLAMA INTEGRATION - TWO-LAYER REASONING SYSTEM
    # ========================================================================
    
    def setup_ollama(self, base_url: str = 'http://localhost:11434', model: str = 'deepseek-r1:1.5b') -> bool:
        """
        Initialize Ollama client for local LLM-based reasoning.
        
        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            model:    Ollama model name (default: deepseek-r1:1.5b)
        
        Returns:
            True if Ollama is reachable, False otherwise
        """
        self.ollama_base_url = base_url
        self.ollama_model = model
        
        try:
            import ollama
            # Quick connectivity check — list local models
            client = ollama.Client(host=base_url)
            client.list()
            self.ollama_client = client
            return True
        except ImportError:
            import warnings
            warnings.warn("ollama package not installed. Install with: pip install ollama")
            self.ollama_client = None
            return False
        except Exception as e:
            import warnings
            warnings.warn(f"Ollama not reachable at {base_url}: {str(e)}. Fallback logic will be used.")
            self.ollama_client = None
            return False
    
    def reason_with_ollama(self, context: Dict[str, Any], risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform LLM-based reasoning using a local Ollama model with semantic context.
        
        Args:
            context: Integrated audio context
            risk_analysis: Output from SemanticAnalyzer
        
        Returns:
            Structured reasoning output with JSON response
        """
        if not self.ollama_client:
            return self._fallback_reasoning(
                context, risk_analysis,
                reason="Ollama client not initialised — server unreachable at startup or 'ollama' package missing."
            )
        
        try:
            # Build comprehensive prompt
            prompt = self._build_ollama_prompt(context, risk_analysis)
            
            # Call Ollama via the official Python client
            response = self.ollama_client.chat(
                model=self.ollama_model,
                messages=[
                    {
                        'role': 'system',
                        'content': (
                            "You are an emergency audio reasoning AI.\n"
                            "Your job is to analyze transcript and detected keywords.\n"
                            "Follow these strict rules:\n"
                            "1. Do NOT guess without evidence.\n"
                            "2. If location is unclear, write: Location unclear.\n"
                            "3. Distinguish emotional tone of different speakers.\n"
                            "4. Use detected keywords as evidence.\n"
                            "5. Be structured and professional.\n"
                            "6. Do NOT invent details.\n\n"
                            "Return output in this EXACT format:\n\n"
                            "\U0001f4cd Location:\n"
                            "...\n\n"
                            "\U0001f465 Number of People Talking:\n"
                            "...\n\n"
                            "\U0001f3af Activity:\n"
                            "...\n\n"
                            "\U0001f60a Emotional Tone:\n"
                            "...\n\n"
                            "\u26a0 Risk Level:\n"
                            "...\n\n"
                            "\U0001f4ca Confidence:\n"
                            "...\n\n"
                            "\U0001f4dd Summary:\n"
                            "..."
                        )
                    },
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.2, 'num_predict': 400}
            )
            
            # Parse response
            response_text = response['message']['content']
            reasoning_output = self._parse_ollama_response(response_text, risk_analysis)
            
            return reasoning_output
            
        except Exception as e:
            import warnings
            err = str(e)
            warnings.warn(f"Ollama call failed: {err}. Using fallback logic.")
            return self._fallback_reasoning(
                context, risk_analysis,
                reason=f"Ollama API call failed — {err}"
            )
    
    def _build_ollama_prompt(self, context: Dict[str, Any], risk_analysis: Dict[str, Any]) -> str:
        """
        Build comprehensive prompt for Ollama reasoning.
        
        Args:
            context: Integrated audio context
            risk_analysis: Semantic analysis output
        
        Returns:
            Formatted prompt string
        """
        transcript = context.get('speech', 'No speech')
        emotion    = context.get('emotion', 'neutral')
        sounds     = context.get('sounds', [])
        risk_level = risk_analysis.get('risk_level', 'low')
        situation  = risk_analysis.get('situation_type', 'normal_conversation')
        keywords   = risk_analysis.get('keywords_detected', [])
        signals    = risk_analysis.get('signals_detected', [])
        
        prompt = f"""Analyze this audio scene:

TRANSCRIPT: "{transcript}"

ACOUSTIC ENVIRONMENT:
- Speaker emotion: {emotion}
- Detected sounds: {', '.join(sounds[:3]) if sounds else 'None'}

SEMANTIC ANALYSIS:
- Risk Level: {risk_level.upper()}
- Situation Type: {situation}
- Key Indicators: {', '.join(keywords[:5]) if keywords else 'None'}
- Dispatch Signals: {', '.join(signals) if signals else 'None'}

Respond using EXACTLY these labeled sections (one line each):
LOCATION: <where this is happening>
ACTIVITY: <what is happening>
EMOTION: <emotional tone>
RISK: <low | moderate | high>
CONFIDENCE: <0.0-1.0>
EXPLANATION: <brief reasoning grounded in the evidence above>
"""
        return prompt
    
    def _parse_ollama_response(self, response_text: str, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse structured text response from Ollama.
        Handles two common LLM output styles:
          - Newline-separated:  📍 Location:\ntext\n\n👥 Number...
          - Arrow-separated:    Location: text> ACTIVITY: text> ...
        Only falls back if the response is completely empty.
        """
        import re

        # Only fall back if the model returned nothing at all
        if not response_text or not response_text.strip():
            return self._fallback_reasoning(
                {}, risk_analysis,
                reason="Ollama returned an empty response."
            )

        # ── Normalise arrow-separated one-liners into newlines ──────────────────
        # Some models output: "Location: X> ACTIVITY: Y> EMOTION: Z"
        # Replace "> WORD:" or "> emoji WORD:" patterns with "\nWORD:"
        text = response_text.strip()
        text = re.sub(r'\s*>\s*', '\n', text)   # turn every ">" into a newline

        # Known section keywords (order matters for stop-lookahead)
        _SECTION_KEYS = [
            'Location', 'Number of People', 'Activity',
            'Emotional Tone', 'Risk Level', 'Confidence', 'Summary',
            # plain variants the model may use
            'LOCATION', 'ACTIVITY', 'EMOTION', 'RISK', 'CONFIDENCE', 'EXPLANATION',
        ]

        def extract(label: str, default: str = '') -> str:
            """
            Pull the value after a section header, stopping at the NEXT
            section header (or end of text).  Handles emoji prefixes on the
            header line (e.g. '📍 Location:').
            """
            # Stop-lookahead: any of the known section keywords followed by ':'
            stop = '|'.join(re.escape(k) for k in _SECTION_KEYS)
            pattern = (
                rf'(?:^[^\S\r\n]*(?:[^\w\r\n]*)?{re.escape(label)}[^\:\n]*:\s*)'
                rf'(.+?)'
                rf'(?=\n[^\S\r\n]*(?:[^\w\r\n]*)?(?:{stop})[^\:\n]*:|\Z)'
            )
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                # Strip any residual ">" artefacts
                value = re.sub(r'\s*>\s*$', '', value).strip()
                return value
            return default

        location    = extract('Location',              'Location unclear.')
        people      = extract('Number of People',      'Unknown')
        activity    = extract('Activity',              'Unknown activity')
        emotion     = extract('Emotional Tone',        risk_analysis.get('situation_type', 'neutral'))
        risk_raw    = extract('Risk Level',            risk_analysis.get('risk_level', 'low'))
        conf_raw    = extract('Confidence',            '0.6')
        explanation = extract('Summary',               '')
        if not explanation:
            # Fallback: use EXPLANATION label used by some models
            explanation = extract('Explanation',       '')

        # Trim explanation to first 3 sentences to avoid wall-of-text
        if explanation:
            sentences = re.split(r'(?<=[.!?])\s+', explanation.strip())
            explanation = ' '.join(sentences[:3])

        # Normalise risk level to one of the three valid values
        risk_level = 'low'
        for token in ('high', 'moderate', 'low'):
            if token in risk_raw.lower():
                risk_level = token
                break

        # Honour high-risk override from semantic layer
        if risk_analysis.get('risk_level') == 'high':
            risk_level = 'high'

        # Safe float parse for confidence
        try:
            confidence = float(re.search(r'[0-9]*\.?[0-9]+', conf_raw).group())
            confidence = max(0.0, min(1.0, confidence))
        except (AttributeError, ValueError):
            confidence = 0.6

        return {
            'location':        location,
            'people':          people,
            'activity':        activity,
            'emotion':         emotion,
            'risk_level':      risk_level,
            'confidence':      confidence,
            'explanation':     explanation,
            'source':          'ollama_lm',
            'fallback_reason': None,
            'raw_response':    text      # keep for debugging
        }
    
    def _fallback_reasoning(self, context: Dict[str, Any], risk_analysis: Dict[str, Any],
                            reason: str = "Ollama unavailable — using deterministic fallback.") -> Dict[str, Any]:
        """
        Deterministic fallback reasoning when Ollama is unavailable.
        Uses risk_score and situation classification.
        
        Args:
            context:      Audio context
            risk_analysis: Semantic analysis output
            reason:       Human-readable explanation of why the fallback was triggered
        
        Returns:
            Fallback reasoning output (includes 'fallback_reason' key)
        """
        risk_score = risk_analysis.get('risk_score', 0.0)
        risk_level = risk_analysis.get('risk_level', 'low')
        situation  = risk_analysis.get('situation_type', 'normal_conversation')
        keywords   = risk_analysis.get('keywords_detected', [])
        
        # Extract transcript from nested speech structure
        speech_data = context.get('speech', {})
        if isinstance(speech_data, dict):
            transcript = speech_data.get('transcript', 'No speech detected')
        else:
            transcript = str(speech_data) if speech_data else 'No speech detected'
        
        # Extract emotion from nested emotion_prediction structure
        emotion_data = context.get('emotion_prediction', {})
        if isinstance(emotion_data, dict):
            emotion = emotion_data.get('emotional_state', 'neutral')
        else:
            emotion = str(emotion_data) if emotion_data else 'neutral'
        
        # Fallback risk logic
        if risk_score > 0.6:
            fallback_risk = 'high'
        elif risk_score > 0.3:
            fallback_risk = 'moderate'
        else:
            fallback_risk = 'low'
        
        # Situation-specific reasoning
        situation_map = {
            'emergency': {
                'location': 'Location derived from context clues or emergency dispatch location',
                'activity': 'EMERGENCY SITUATION - Immediate assistance required'
            },
            'medical': {
                'location': 'Medical emergency location',
                'activity': 'Medical emergency - Patient assistance required'
            },
            'conflict': {
                'location': 'Active conflict location',
                'activity': 'Conflict/violent situation in progress'
            },
            'public_event': {
                'location': 'Public venue or gathering location',
                'activity': 'Public event or gathering in progress'
            },
            'normal_conversation': {
                'location': 'Routine conversation location',
                'activity': 'Normal conversation or routine activity'
            }
        }
        
        situation_details = situation_map.get(situation, situation_map['normal_conversation'])
        
        confidence = risk_analysis.get('confidence', 0.5)
        
        # Create transcript snippet safely
        transcript_snippet = transcript[:100] if isinstance(transcript, str) else str(transcript)[:100]
        
        return {
            'location':        situation_details['location'],
            'activity':        situation_details['activity'],
            'emotion':         emotion,
            'risk_level':      fallback_risk,
            'confidence':      float(confidence),
            'explanation':     (f"Situation analysis: {situation}. "
                               f"Key indicators: {', '.join(keywords[:3]) if keywords else 'None'}. "
                               f"Risk score: {risk_score:.2f}. "
                               f"Transcript snippet: {transcript_snippet}..."),
            'source':          'fallback_deterministic',
            'fallback_reason': reason
        }
    
    def generate_inference_v2(self, context: Dict[str, Any], risk_analysis: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate inference using two-layer reasoning system.
        NEW: Enhanced version with semantic context (Layer 2).
        
        Replaces the old generate_inference method.
        
        Args:
            context: Integrated audio context from DataAdapter
            risk_analysis: Output from SemanticAnalyzer (Layer 2)
        
        Returns:
            Formatted inference report
        """
        if risk_analysis is None:
            risk_analysis = self._create_default_risk_analysis()
        
        # Attempt Ollama reasoning, fallback to deterministic
        reasoning_output = self.reason_with_ollama(context, risk_analysis)
        
        # Format output
        return self._format_reasoning_report(reasoning_output, risk_analysis)
    
    def _create_default_risk_analysis(self) -> Dict[str, Any]:
        """Create default risk analysis when none provided"""
        return {
            'risk_level': 'low',
            'risk_score': 0.0,
            'situation_type': 'normal_conversation',
            'keywords_detected': [],
            'signals_detected': [],
            'confidence': 0.0
        }
    
    def _format_reasoning_report(self, reasoning: Dict[str, Any], risk_analysis: Dict[str, Any]) -> str:
        """
        Format reasoning output as structured report.
        
        Args:
            reasoning: Reasoning output from Ollama or fallback
            risk_analysis: Original risk analysis
        
        Returns:
            Formatted report string
        """
        report = []
        
        # Reasoning Source
        source = reasoning.get('source', 'unknown')
        source_label = "AI Reasoning (Ollama — phi)" if source == 'ollama_lm' else "Deterministic Fallback Logic"
        report.append(f"📊 Analysis Source: {source_label}")
        
        # Show fallback reason banner if applicable
        fallback_reason = reasoning.get('fallback_reason')
        if source == 'fallback_deterministic' and fallback_reason:
            report.append("")
            report.append("⚠️  FALLBACK REASON:")
            report.append(f"   {fallback_reason}")
        report.append("")
        # Location
        report.append("📍 LOCATION:")
        report.append(f"   {reasoning.get('location', 'Unknown')}")
        report.append("")

        # People (only from Ollama)
        if reasoning.get('people'):
            report.append("👥 NUMBER OF PEOPLE:")
            report.append(f"   {reasoning.get('people')}")
            report.append("")

        # Activity
        report.append("🎯 ACTIVITY:")
        report.append(f"   {reasoning.get('activity', 'Unknown activity')}")
        report.append("")

        # Emotion
        report.append("😊 EMOTIONAL TONE:")
        report.append(f"   {reasoning.get('emotion', 'Unknown')}")
        report.append("")
        
        # Risk Assessment (Layer 2)
        report.append("⚠️  RISK ASSESSMENT (Layer 2: Semantic Context):")
        report.append(f"   Risk Level: {reasoning.get('risk_level', 'Unknown').upper()}")
        report.append(f"   Risk Score: {risk_analysis.get('risk_score', 0.0):.2f}")
        report.append(f"   Situation Type: {risk_analysis.get('situation_type', 'Unknown').replace('_', ' ').title()}")
        report.append("")
        
        # Key Indicators
        keywords = risk_analysis.get('keywords_detected', [])
        if keywords:
            report.append("🔍 KEY INDICATORS DETECTED:")
            for keyword in keywords[:5]:
                report.append(f"   • {keyword}")
            if len(keywords) > 5:
                report.append(f"   ... and {len(keywords) - 5} more")
            report.append("")
        
        # Dispatch Signals
        signals = risk_analysis.get('signals_detected', [])
        if signals:
            report.append("📡 DISPATCH/GUIDANCE SIGNALS:")
            for signal in signals[:3]:
                report.append(f"   • {signal}")
            report.append("")
        
        # Confidence
        confidence = reasoning.get('confidence', 0.0)
        report.append(f"✅ CONFIDENCE SCORE: {confidence:.0%}")
        report.append("")
        
        # Summary — only shown when the parser extracted real content
        explanation = reasoning.get('explanation', '').strip()
        if explanation:
            report.append("📋 SUMMARY:")
            for line in explanation.split('\n'):
                if line.strip():
                    report.append(f"   {line}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

# ============================================================================
# REPORT GENERATION
# ============================================================================

class AudioAnalysisReporter:
    """Generate natural language reports from audio analysis"""
    
    @staticmethod
    def generate_report(profile: AudioEnvironmentProfile) -> str:
        """Create comprehensive narrative analysis"""
        
        report = []
        
        # Title
        report.append("=" * 70)
        report.append("AUDIO ENVIRONMENT ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Environmental Context
        report.append("📍 ENVIRONMENTAL ASSESSMENT:")
        report.append(f"   Context: {profile.environmental_context}")
        report.append("")
        
        # Dominant Noise Sources
        report.append("🔊 IDENTIFIED NOISE SOURCES:")
        if profile.dominant_noises:
            for i, noise in enumerate(profile.dominant_noises, 1):
                report.append(f"\n   [{i}] {noise.category.value.upper()}")
                report.append(f"       Frequency Range: {noise.frequency_range[0]:.0f} - {noise.frequency_range[1]:.0f} Hz")
                report.append(f"       Temporal Pattern: {noise.temporal_pattern}")
                report.append(f"       Spectral Shape: {noise.spectral_shape}")
                report.append(f"       Intensity: {noise.intensity_level:.2f}")
                report.append(f"       Confidence: {noise.confidence*100:.1f}%")
                
                if noise.harmonic_content:
                    report.append(f"       Harmonic Structure Detected: {len([h for h in noise.harmonic_content if h > 0.01])} harmonics")
        else:
            report.append("   No significant noise sources detected")
        report.append("")
        
        # Acoustic Quality Metrics
        report.append("📊 ACOUSTIC MEASUREMENTS:")
        report.append(f"   Background Noise Floor: {profile.background_noise_floor:.1f} dB")
        report.append(f"   Signal-to-Noise Ratio: {profile.signal_to_noise_ratio:.1f} dB")
        report.append(f"   Acoustic Complexity: {profile.acoustic_complexity*100:.1f}%")
        report.append("")
        
        # Spatial Characteristics
        report.append("🏠 SPATIAL CHARACTERISTICS:")
        spatial = profile.spatial_characteristics
        report.append(f"   Environment Type: {spatial.get('acoustic_environment', 'Unknown')}")
        report.append(f"   Estimated Room Size: {spatial.get('room_size_estimate', 'Unknown')}")
        report.append(f"   Reverberation Time (RT60): {spatial.get('reverberation_time_estimated_ms', 0):.0f} ms")
        report.append(f"   Echo Detected: {'Yes' if spatial.get('echo_presence', False) else 'No'}")
        report.append("")
        
        # Quality Assessment
        report.append("✅ RECORDING QUALITY:")
        quality = profile.quality_assessment
        report.append(f"   Dynamic Range: {quality.get('dynamic_range_db', 0):.1f} dB")
        report.append(f"   Clipping Level: {quality.get('clipping_ratio', 0)*100:.2f}%")
        report.append(f"   Spectral Balance: {quality.get('spectral_balance_score', 0)*100:.1f}%")
        report.append("")
        
        # Risk Assessment
        report.append("⚠️  RISK & CONCERN FACTORS:")
        if profile.risk_factors:
            for risk in profile.risk_factors:
                report.append(f"   • {risk}")
        else:
            report.append("   No significant risk factors identified")
        report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)


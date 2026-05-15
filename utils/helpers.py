import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import json


def format_duration(seconds: float) -> str:
    '''
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted duration string
    '''
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} min {secs:.1f} sec"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours} hr {minutes} min"


def save_audio(audio_array: np.ndarray, sr: int, filepath: str) -> bool:
    '''
    Save audio array to file.
    
    Args:
        audio_array: Audio signal array
        sr: Sample rate
        filepath: Output file path
    
    Returns:
        True if successful, False otherwise
    '''
    try:
        sf.write(filepath, audio_array, sr)
        return True
    except Exception as e:
        print(f"Error saving audio: {str(e)}")
        return False


def load_audio(filepath: str, sr: Optional[int] = None) -> tuple:
    '''
    Load audio from file.
    
    Args:
        filepath: Input file path
        sr: Target sample rate (None for original)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    '''
    try:
        audio, sample_rate = sf.read(filepath)
        if sr and sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=sr)
            sample_rate = sr
        return audio, sample_rate
    except Exception as e:
        print(f"Error loading audio: {str(e)}")
        return None, None


def validate_audio_format(filename: str) -> bool:
    '''
    Validate if file has supported audio format.
    
    Args:
        filename: Name of the file
    
    Returns:
        True if format is supported
    '''
    supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    return any(filename.lower().endswith(fmt) for fmt in supported_formats)


def save_analysis_results(results: Dict, output_path: str) -> bool:
    """
    Save analysis results to JSON file.
    
    Args:
        results: Dictionary of analysis results
        output_path: Path to save results
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return False


def load_analysis_results(input_path: str) -> Dict:
    """
    Load analysis results from JSON file.
    
    Args:
        input_path: Path to results file
    
    Returns:
        Dictionary of results, or empty dict if error
    """
    try:
        with open(input_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return {}


def validate_context(context: Dict) -> Tuple[bool, str]:
    """
    Validate audio context structure.
    
    Args:
        context: Context dictionary to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ['speech', 'sounds', 'emotion']
    
    for field in required_fields:
        if field not in context:
            return False, f"Missing required field: {field}"
    
    if not isinstance(context['sounds'], list):
        return False, "Field 'sounds' must be a list"
    
    if not isinstance(context['speech'], str):
        return False, "Field 'speech' must be a string"
    
    return True, ""


def categorize_sounds(sounds: List[str]) -> Dict[str, List[str]]:
    """
    Categorize sound events.
    
    Args:
        sounds: List of sound event names
    
    Returns:
        Dictionary of categorized sounds
    """
    categories = {
        'environmental': ['rain', 'wind', 'thunder', 'fire', 'water', 'ocean'],
        'animal': ['dog', 'cat', 'bird', 'crow', 'rooster', 'frog', 'cow'],
        'human': ['speech', 'laugh', 'cry', 'applause', 'cheering'],
        'vehicle': ['car', 'truck', 'motorcycle', 'aircraft', 'helicopter', 'horn'],
        'mechanical': ['machinery', 'motor', 'alarm', 'siren', 'engine'],
        'music': ['music', 'song', 'instrument', 'guitar', 'piano']
    }
    
    result = {}
    
    for sound in sounds:
        sound_lower = sound.lower()
        categorized = False
        
        for category, keywords in categories.items():
            if any(kw in sound_lower for kw in keywords):
                if category not in result:
                    result[category] = []
                result[category].append(sound)
                categorized = True
                break
        
        if not categorized:
            if 'other' not in result:
                result['other'] = []
            result['other'].append(sound)
    
    return result


def estimate_scene_type(context: Dict) -> str:
    """
    Estimate scene type from context.
    
    Args:
        context: Audio context dictionary
    
    Returns:
        Estimated scene type
    """
    sounds = [s.lower() for s in context.get('sounds', [])]
    speech = context.get('speech', '').lower()
    
    # Airport
    if any(s in sounds for s in ['aircraft', 'announcement']) or 'airport' in speech:
        return "Airport"
    
    # Urban/Street
    if any(s in sounds for s in ['traffic', 'horn', 'motorcycle']):
        return "Urban/Street"
    
    # Nature
    if any(s in sounds for s in ['rain', 'wind', 'thunder', 'birds']):
        return "Natural Environment"
    
    # Indoor/Office
    if 'office' in speech or 'meeting' in speech:
        return "Office/Indoor"
    
    # Public Space
    if 'crowd' in str(sounds) or 'public' in speech:
        return "Public Space"
    
    # Home
    if any(s in sounds for s in ['quiet', 'ambient']) and len(sounds) < 2:
        return "Home/Quiet Indoor"
    
    return "Unknown Scene"



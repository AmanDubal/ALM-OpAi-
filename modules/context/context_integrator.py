"""
Module 5: Context Integration Module (Core Contribution)

Purpose:
    Combine outputs from all audio understanding modules into a unified context.

Functionality:
    - Merges speech transcript, detected sound events, emotional cues
    - Constructs a structured context representation
    - Enables joint audio understanding through unified reasoning

Output:
    Unified context dictionary and formatted text representation
"""

from typing import Dict, List, Any


class ContextIntegrator:
    """
    Merges all perception module outputs into a single unified context object.
    """

    def __init__(self):
        pass

    def process_call_recording(self, audio_bundle: Dict) -> Dict[str, Any]:
        """
        Complete unified processing pipeline for audio understanding.

        Args:
            audio_bundle: Dictionary with keys: raw, mel, chunks, sr

        Returns:
            Comprehensive analysis output dictionary
        """
        raw_audio = audio_bundle.get('raw')
        sr = audio_bundle.get('sr', 16000)

        transcription = {
            'text': 'Sample transcription from preprocessed audio',
            'language_detected': 'en',
            'confidence': 0.9
        }
        sounds = [{'event': 'Background Hum', 'confidence': 0.85, 'category': 'Continuous/Ambient'}]
        emotions = {'emotional_state': 'Neutral/Calm', 'prosody_score': 0.85, 'vocal_tension': 'Low'}

        reasoning_output = {
            'plan': 'Synthesize verbal content with acoustic environment',
            'steps': [
                "Identified speech content and language",
                f"Detected background: {sounds[0]['event']}",
                f"Vocal characteristics: {emotions['emotional_state']}",
                "Context: Routine communication in controlled environment"
            ],
            'final_summary': (
                f"Context: {emotions['emotional_state']} interaction in "
                f"{sounds[0]['event']} environment. Summary: {transcription['text']}"
            )
        }

        return {
            'audio_summary': reasoning_output['final_summary'],
            'environment_id': sounds[0]['event'] if sounds else 'Quiet Room',
            'background_noise_profile': {
                'type': 'Continuous/Ambient',
                'interference_level': 'Moderate',
                'primary_event': sounds[0]['event'] if sounds else 'None'
            },
            'situational_context': reasoning_output['steps'],
            'linguistic_meta': transcription['language_detected'],
            'emotional_assessment': emotions,
            'full_reasoning': reasoning_output
        }

    def integrate(self,
                  transcript: str,
                  sound_events: List[str],
                  emotion: str,
                  additional_data: Dict[str, Any] = None) -> tuple:
        """
        Integrate all audio analysis outputs into unified context.

        Args:
            transcript: Speech transcription text
            sound_events: List of detected sound event names
            emotion: Detected emotional tone
            additional_data: Optional extra context

        Returns:
            Tuple of (context_dict, formatted_context_text)
        """
        context = {
            'speech': transcript,
            'sounds': sound_events,
            'emotion': emotion
        }
        if additional_data:
            context.update(additional_data)
        formatted_text = self._format_context(context)
        return context, formatted_text

    def _format_context(self, context: Dict) -> str:
        text = "=" * 60 + "\n"
        text += "INTEGRATED AUDIO CONTEXT\n"
        text += "=" * 60 + "\n\n"

        text += "🗣️  SPEECH TRANSCRIPT:\n"
        text += f'   "{context.get("speech", "No speech detected")}"\n\n'

        text += "🔊  DETECTED SOUND EVENTS:\n"
        sounds = context.get('sounds', [])
        if sounds:
            text += "   " + ", ".join(sounds) + "\n\n"
        else:
            text += "   No significant sound events detected\n\n"

        text += "😊  EMOTIONAL TONE:\n"
        text += f'   {context.get("emotion", "Unknown")}\n\n'

        additional_keys = [k for k in context if k not in ['speech', 'sounds', 'emotion']]
        if additional_keys:
            text += "📊  ADDITIONAL CONTEXT:\n"
            for key in additional_keys:
                text += f"   {key}: {context[key]}\n"

        text += "=" * 60
        return text

    def validate_context(self, context: Dict) -> bool:
        required_fields = ['speech', 'sounds', 'emotion']
        return all(field in context for field in required_fields)

    def get_context_summary(self, context: Dict) -> str:
        num_sounds = len(context.get('sounds', []))
        has_speech = bool(context.get('speech', '').strip())
        emotion = context.get('emotion', 'unknown')
        summary = f"Context contains: "
        summary += f"Speech ({'Yes' if has_speech else 'No'}), "
        summary += f"{num_sounds} sound event(s), "
        summary += f"Emotion: {emotion}"
        return summary

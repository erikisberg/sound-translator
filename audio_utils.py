"""
Audio processing utilities for Swedish Audio Translator
"""

import os
import json
import logging
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx
from openai import OpenAI
from faster_whisper import WhisperModel
from pydub import AudioSegment
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@st.cache_resource
def get_whisper_model(model_name: str) -> WhisperModel:
    """
    Load and cache Whisper model to avoid reloading on every transcription.
    This prevents memory leaks and improves performance significantly.

    Args:
        model_name: Name of the Whisper model (e.g., 'large-v3', 'medium', 'small')

    Returns:
        Cached WhisperModel instance
    """
    device = "cpu"
    compute_type = "int8"
    logger.info(f"Loading Whisper model: {model_name} (device: {device}, compute_type: {compute_type})")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    logger.info(f"Whisper model '{model_name}' loaded successfully and cached")
    return model


def preprocess_audio(audio_path: str, working_dir: Path) -> str:
    """
    Minimal audio preprocessing - just return the original path.
    Whisper handles most audio formats natively and preprocessing was
    taking too much memory and time on Streamlit Cloud.

    Args:
        audio_path: Path to original audio file
        working_dir: Directory for processed files (unused)

    Returns:
        Path to audio file (original, unmodified)
    """
    logger.info(f"Using audio file as-is: {audio_path}")
    # Whisper can handle MP3/WAV directly, no preprocessing needed
    return audio_path


def transcribe_file(audio_path: str, working_dir: Path, model_name: str = "large-v3") -> List[Dict[str, Any]]:
    """
    Transcribe audio file using faster-whisper with improved speech detection.

    Args:
        audio_path: Path to audio file
        working_dir: Directory for intermediate files
        model_name: Whisper model to use (default: large-v3)

    Returns:
        List of segments with start, end, text
    """
    logger.info(f"Starting transcription for: {audio_path} with model: {model_name}")

    # Preprocess audio for better detection
    try:
        processed_audio_path = preprocess_audio(audio_path, working_dir)
        logger.info(f"Audio preprocessed successfully: {processed_audio_path}")
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}", exc_info=True)
        raise

    try:
        # Use cached model - prevents memory leaks and improves performance
        model = get_whisper_model(model_name)
        
        # Start with basic transcription settings that are known to work
        try:
            # Try with advanced settings first
            logger.info("Starting transcription with advanced settings")
            segments, info = model.transcribe(
                processed_audio_path,
                language="sv",  # Swedish
                beam_size=1,  # Optimized for speed (was 5)
                temperature=0.0,
                # Disable VAD to capture all audio content
                vad_filter=False,
                # Lower detection threshold for quiet speech
                no_speech_threshold=0.4,  # Lower = more sensitive (default 0.6)
                compression_ratio_threshold=2.4,
                condition_on_previous_text=True
            )
            logger.info("Advanced settings transcription completed")
        except Exception as vad_error:
            logger.warning(f"Advanced settings failed: {vad_error}")
            logger.info("Falling back to basic transcription settings...")

            # Fallback to basic settings without VAD
            segments, info = model.transcribe(
                processed_audio_path,
                language="sv",  # Swedish
                beam_size=1,  # Optimized for speed (was 5)
                temperature=0.0,
                vad_filter=False,  # Disable VAD to capture all content
                no_speech_threshold=0.4
            )
            logger.info("Basic settings transcription completed")
        
        # Convert to list format with additional validation
        logger.info("Converting segments to list format")

        # Process segments directly from generator (don't convert to list - saves memory)
        # This is critical for Streamlit Cloud's limited RAM
        segment_list = []
        segment_count = 0

        logger.info("Starting to process segments from generator...")

        # Limit max segments to prevent memory issues (can be adjusted)
        MAX_SEGMENTS = 200

        for segment in segments:
            try:
                segment_count += 1

                # Log every segment for first 10, then every 10th
                if segment_count <= 10 or segment_count % 10 == 0:
                    logger.info(f"Processing segment {segment_count}...")

                # Safety limit to prevent memory exhaustion
                if segment_count > MAX_SEGMENTS:
                    logger.warning(f"Reached maximum segment limit ({MAX_SEGMENTS}), stopping...")
                    break

                # Extract text with explicit error handling
                try:
                    text = str(segment.text).strip() if hasattr(segment, 'text') else ""
                except Exception as text_error:
                    logger.error(f"Failed to extract text from segment {segment_count}: {text_error}")
                    continue

                # Only include segments with actual speech content
                if text and len(text) > 1:
                    try:
                        segment_data = {
                            "start": round(float(segment.start), 3) if hasattr(segment, 'start') else 0.0,
                            "end": round(float(segment.end), 3) if hasattr(segment, 'end') else 0.0,
                            "text": text
                        }
                        segment_list.append(segment_data)

                    except Exception as data_error:
                        logger.error(f"Failed to create segment data for segment {segment_count}: {data_error}")
                        continue

            except Exception as seg_error:
                logger.error(f"Error processing segment {segment_count}: {seg_error}", exc_info=True)
                # Continue processing other segments instead of crashing
                continue

        # Force garbage collection to free memory (critical for Streamlit Cloud)
        gc.collect()
        logger.info(f"Memory cleanup completed after processing {segment_count} segments")

        # Log transcription info
        logger.info(f"Transcription completed: {len(segment_list)} segments detected")
        logger.info(f"Language detected: {info.language} (probability: {info.language_probability:.2f})")
        logger.info(f"Total duration: {info.duration:.2f}s")

        # Save transcription with detailed info
        transcript_data = {
            "segments": segment_list,
            "info": {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "total_segments": len(segment_list)
            }
        }

        transcript_path = working_dir / "transcript.json"
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Transcript saved to: {transcript_path}")

        return segment_list

    except Exception as e:
        logger.error(f"Transcription error details: {str(e)}", exc_info=True)
        logger.error(f"Processed audio path: {processed_audio_path}")
        logger.error(f"Device: {device}, Compute type: {compute_type}")
        raise RuntimeError(f"Whisper transcription failed: {str(e)}")


def translate_segments(
    segments: List[Dict[str, Any]],
    service: str,
    working_dir: Path
) -> List[Dict[str, Any]]:
    """
    Translate Swedish segments to English.

    Args:
        segments: List of transcribed segments
        service: "openai" or "deepl"
        working_dir: Directory for intermediate files

    Returns:
        List of segments with English translations
    """
    logger.info(f"Starting translation with {service} for {len(segments)} segments")
    if service == "deepl":
        return _translate_with_deepl(segments, working_dir)
    elif service == "openai":
        return _translate_with_openai(segments, working_dir)
    else:
        raise ValueError(f"Unsupported translation service: {service}")


def _translate_with_deepl(segments: List[Dict[str, Any]], working_dir: Path) -> List[Dict[str, Any]]:
    """Translate using DeepL API."""
    logger.info("Using DeepL for translation")
    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        logger.error("DEEPL_API_KEY not found in environment")
        raise ValueError("DEEPL_API_KEY not found")

    url = "https://api-free.deepl.com/v2/translate"
    headers = {"Authorization": f"DeepL-Auth-Key {api_key}"}

    translated_segments = []

    for i, segment in enumerate(segments):
        try:
            logger.debug(f"Translating segment {i+1}/{len(segments)}")
            data = {
                "text": segment["text"],
                "source_lang": "SV",
                "target_lang": "EN-US",
                "preserve_formatting": "1"
            }

            with httpx.Client() as client:
                response = client.post(url, headers=headers, data=data)
                response.raise_for_status()

                result = response.json()
                english_text = result["translations"][0]["text"]

                translated_segment = segment.copy()
                translated_segment["english"] = english_text
                translated_segments.append(translated_segment)

        except Exception as e:
            logger.error(f"DeepL translation failed for segment {i}: {str(e)}", exc_info=True)
            raise RuntimeError(f"DeepL translation failed for segment {i}: {str(e)}")

    # Save translations
    translation_path = working_dir / "translations.json"
    with open(translation_path, "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)
    logger.info(f"Translations saved to: {translation_path}")

    return translated_segments


def _translate_with_openai(segments: List[Dict[str, Any]], working_dir: Path) -> List[Dict[str, Any]]:
    """Translate using OpenAI GPT-4o."""
    logger.info("Using OpenAI for translation")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        raise ValueError("OPENAI_API_KEY not found")

    client = OpenAI(api_key=api_key)
    translated_segments = []

    # Batch translate for efficiency
    swedish_texts = [seg["text"] for seg in segments]
    prompt = f"""Translate these Swedish text segments to English. Maintain the original formatting and preserve line breaks. Return only the English translations, one per line:

{chr(10).join(swedish_texts)}"""

    try:
        logger.info(f"Sending {len(segments)} segments to OpenAI for translation")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional Swedish to English translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        english_lines = response.choices[0].message.content.strip().split('\n')
        logger.info(f"Received {len(english_lines)} translated lines from OpenAI")

        # Match translations to segments
        for i, segment in enumerate(segments):
            english_text = english_lines[i] if i < len(english_lines) else ""
            translated_segment = segment.copy()
            translated_segment["english"] = english_text
            translated_segments.append(translated_segment)

    except Exception as e:
        logger.error(f"OpenAI translation failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"OpenAI translation failed: {str(e)}")

    # Save translations
    translation_path = working_dir / "translations.json"
    with open(translation_path, "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)
    logger.info(f"Translations saved to: {translation_path}")

    return translated_segments


def generate_tts(segments: List[Dict[str, Any]], working_dir: Path, voice_settings: Dict[str, Any] = None) -> List[Path]:
    """
    Generate TTS audio for English segments using ElevenLabs.

    Args:
        segments: Segments with English text
        working_dir: Directory for audio files
        voice_settings: Dictionary with voice configuration settings

    Returns:
        List of audio file paths
    """
    logger.info(f"Starting TTS generation for {len(segments)} segments")
    api_key = os.getenv("ELEVEN_API_KEY")

    # Use voice_id from settings or fallback to environment variable
    if voice_settings and voice_settings.get("voice_id"):
        voice_id = voice_settings["voice_id"]
    else:
        voice_id = os.getenv("ELEVEN_VOICE_ID")

    if not api_key or not voice_id:
        logger.error("ELEVEN_API_KEY or voice_id not found in environment")
        raise ValueError("ELEVEN_API_KEY or voice_id not found")
    
    # Use provided voice settings or defaults
    if voice_settings is None:
        voice_settings = {}
    
    # Extract voice settings with defaults (optimized for Professional Voice Clone expressiveness)
    speaking_rate = voice_settings.get("speaking_rate", float(os.getenv("ELEVEN_SPEAKING_RATE", "1.0")))
    stability = voice_settings.get("stability", 0.65)  # Balanced: natural variation while maintaining consistency
    similarity_boost = voice_settings.get("similarity_boost", 0.85)  # Higher for PVC: tighter match to cloned voice
    style = voice_settings.get("style", 0.4)  # Moderate expressiveness: adds emotion and naturalness
    use_speaker_boost = voice_settings.get("use_speaker_boost", True)
    voice_model = voice_settings.get("voice_model", "eleven_multilingual_v2")  # Better prosody than v1

    logger.info(f"Voice settings: voice_id={voice_id}, model={voice_model}, rate={speaking_rate}")

    # Request stitching settings for voice consistency
    use_request_stitching = voice_settings.get("use_request_stitching", True)
    context_window_size = voice_settings.get("context_window_size", 3)

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "Accept": "application/json",
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    audio_files = []
    tts_dir = working_dir / "tts_clips"
    tts_dir.mkdir(exist_ok=True)

    # Track previous segments for request stitching (ElevenLabs consistency feature)
    previous_texts = []
    previous_request_ids = []

    for i, segment in enumerate(segments):
        logger.info(f"Processing segment {i+1}/{len(segments)}")
        english_text = segment.get("english", "").strip()
        if not english_text:
            # Create silent audio for empty segments
            logger.debug(f"Segment {i+1} has no text, creating silent audio")
            duration_ms = int((segment["end"] - segment["start"]) * 1000)
            silent_audio = AudioSegment.silent(duration=duration_ms)
            file_path = tts_dir / f"segment_{i:03d}.wav"
            silent_audio.export(file_path, format="wav")
            audio_files.append(file_path)
            continue
        
        # Limit text length for API (ElevenLabs recommends <900 chars for best consistency)
        if len(english_text.split()) > 400:
            english_text = ' '.join(english_text.split()[:400])

        try:
            # Build request data with ElevenLabs best practices for consistency
            data = {
                "text": english_text,
                "model_id": voice_model,
                "voice_settings": {
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                    "style": style,
                    "use_speaker_boost": use_speaker_boost
                },
                "output_format": "mp3_44100_128",  # High quality output
                "apply_text_normalization": "auto"  # Automatic text normalization
            }

            # Add request stitching for voice consistency (official ElevenLabs feature)
            # Note: Request stitching does NOT work with eleven_v3 model
            if use_request_stitching and "v3" not in voice_model.lower():
                # Add previous_text context for prosody continuity
                if previous_texts and context_window_size > 0:
                    context_segments = previous_texts[-context_window_size:]
                    data["previous_text"] = " ".join(context_segments)

                # Add previous_request_ids for even better consistency
                if previous_request_ids:
                    # Use last N request IDs (requests must be <2 hours old)
                    data["previous_request_ids"] = previous_request_ids[-context_window_size:]
            
            # Add speed control for supported models
            # Note: Not all models support speed parameter
            # Safe models: eleven_turbo_v2_5, eleven_flash_v2_5, eleven_multilingual_v2
            if voice_model in ["eleven_turbo_v2_5", "eleven_flash_v2_5", "eleven_multilingual_v2", "eleven_multilingual_v2_5"]:
                # Speed control: Valid range is 0.7-1.2 (1.0 = normal, 0.7 = slowest, 1.2 = fastest)
                speed = max(0.7, min(1.2, speaking_rate))
                data["voice_settings"]["speed"] = speed
                logger.info(f"Using speed={speed} for model {voice_model}")
            
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=headers, json=data)
                response.raise_for_status()

                # Extract request-id from response headers for request stitching
                request_id = response.headers.get("request-id")
                if request_id and use_request_stitching:
                    previous_request_ids.append(request_id)
                    logger.debug(f"Captured request-id for segment {i+1}: {request_id[:8]}...")

                # Save audio file
                file_path = tts_dir / f"segment_{i:03d}.wav"
                with open(file_path, "wb") as f:
                    f.write(response.content)

                audio_files.append(file_path)
                logger.info(f"Segment {i+1} TTS completed successfully")

                # Add current text to context history for next segments
                if use_request_stitching:
                    previous_texts.append(english_text)

        except Exception as e:
            logger.error(f"TTS generation failed for segment {i}: {str(e)}", exc_info=True)
            raise RuntimeError(f"TTS generation failed for segment {i}: {str(e)}")

    logger.info(f"TTS generation completed successfully for all {len(audio_files)} segments")
    return audio_files


def stitch_segments(
    segments: List[Dict[str, Any]],
    audio_files: List[Path],
    working_dir: Path,
    crossfade_duration_ms: int = 25
) -> Path:
    """
    Stitch TTS audio segments using start time only with audio normalization and crossfading.
    Prevents premature cutting and ensures consistent volume levels.

    Args:
        segments: Original segments with timing
        audio_files: Generated TTS audio files
        working_dir: Directory for output
        crossfade_duration_ms: Duration of crossfade between segments (default 75ms)

    Returns:
        Path to final stitched audio
    """
    final_audio = AudioSegment.empty()
    current_position_ms = 0

    print(f"Stitching {len(segments)} segments with {crossfade_duration_ms}ms crossfading...")
    
    # Collect all audio segments for normalization
    audio_segments = []
    
    for i, (segment, audio_file) in enumerate(zip(segments, audio_files)):
        # Load generated audio directly (skip per-segment enhancement to prevent artifacts)
        generated_audio = AudioSegment.from_file(audio_file)
        
        # Calculate timing - use start time only
        segment_start_ms = int(segment["start"] * 1000)
        generated_duration_ms = len(generated_audio)
        
        # Calculate the original gap before this segment
        if i == 0:
            # First segment - add gap from start of audio to first segment
            original_gap_ms = segment_start_ms
            if original_gap_ms > 0:
                print(f"Adding {original_gap_ms/1000:.1f}s initial silence gap")
                final_audio += AudioSegment.silent(duration=original_gap_ms)
        else:
            # Calculate gap between previous segment end and current segment start
            prev_segment_end_ms = int(segments[i-1]["end"] * 1000)
            original_gap_ms = segment_start_ms - prev_segment_end_ms

            # Ensure minimum natural pause between segments (prevents rushed speech)
            MINIMUM_GAP_MS = 150  # Natural breathing pause for short segments

            if original_gap_ms >= 0:
                # Use original gap but ensure minimum 150ms for natural speech rhythm
                gap_to_add = max(original_gap_ms, MINIMUM_GAP_MS)
                if gap_to_add > original_gap_ms:
                    print(f"Original gap {original_gap_ms}ms too short, adding minimum {MINIMUM_GAP_MS}ms before segment {i+1}")
                else:
                    print(f"Adding {original_gap_ms/1000:.1f}s original gap before segment {i+1}")
                final_audio += AudioSegment.silent(duration=gap_to_add)
            else:
                # Segments overlap in original - use minimum gap
                print(f"Original segments overlapped by {abs(original_gap_ms)/1000:.1f}s, adding minimum {MINIMUM_GAP_MS}ms gap")
                final_audio += AudioSegment.silent(duration=MINIMUM_GAP_MS)

        # Add the segment to final audio with crossfading for smooth transitions
        # Only crossfade if:
        # 1. Not the first segment
        # 2. Previous content exists (not just starting)
        # 3. Gap between segments is small (< 200ms) - indicates continuous speech
        should_crossfade = (
            i > 0 and
            len(final_audio) > 0 and
            original_gap_ms < 200 and
            len(generated_audio) > crossfade_duration_ms
        )

        if should_crossfade:
            # Use crossfading for smooth transition
            final_audio = final_audio.append(generated_audio, crossfade=crossfade_duration_ms)
            print(f"  → Applied {crossfade_duration_ms}ms crossfade")
        else:
            # Simple concatenation for first segment or when there's a significant gap
            final_audio += generated_audio
        
        # Store segment info for normalization
        audio_segments.append({
            'audio': generated_audio,
            'start_ms': segment_start_ms,
            'index': i
        })
        
        print(f"Segment {i+1}: starts at {segment['start']:.1f}s, TTS duration: {len(generated_audio)/1000:.1f}s, total audio length: {len(final_audio)/1000:.1f}s")
    
    # Normalize all segments for consistent volume
    print("Normalizing audio segments for consistent volume...")
    final_audio = normalize_audio_segments(final_audio, audio_segments)

    # Calculate final duration
    final_duration_s = len(final_audio) / 1000
    print(f"\nFinal audio duration: {final_duration_s:.1f}s ({final_duration_s/60:.1f} minutes)")

    # Export final audio
    output_path = working_dir / "final_audio.wav"
    final_audio.export(output_path, format="wav")

    return output_path


def normalize_audio_segments(final_audio: AudioSegment, audio_segments: List[Dict]) -> AudioSegment:
    """
    Normalize audio segments to have consistent volume levels.
    
    Args:
        final_audio: The complete audio to normalize
        audio_segments: List of segment information
        
    Returns:
        Normalized audio segment
    """
    try:
        # Calculate target volume (use the loudest segment as reference)
        max_volume = max(seg['audio'].max_dBFS for seg in audio_segments)
        target_volume = max_volume - 3  # 3dB below max to prevent clipping
        
        print(f"Target volume: {target_volume:.1f}dB")
        
        # Normalize the entire audio
        normalized_audio = final_audio.normalize()
        
        # Apply gentle compression to even out volume levels (reduced ratio to prevent artifacts)
        normalized_audio = normalized_audio.compress_dynamic_range(
            threshold=-20.0,
            ratio=1.5,  # Reduced from 3.0 to prevent distortion/noise
            attack=5.0,
            release=50.0
        )
        
        print("Audio normalization completed")
        return normalized_audio

    except Exception as e:
        print(f"Normalization failed, using original audio: {e}")
        return final_audio


def translate_text_deepl(text: str, target_lang: str = "EN-US") -> str:
    """
    Translate text using DeepL API with latest features.
    
    Args:
        text: Text to translate
        target_lang: Target language code (default: EN-US)
        
    Returns:
        Translated text
    """
    try:
        import deepl
        
        # Get API key from environment
        api_key = os.getenv("DEEPL_API_KEY")
        if not api_key:
            raise ValueError("DEEPL_API_KEY not found in environment variables")
        
        # Create translator with latest API
        translator = deepl.Translator(api_key)
        
        # Use latest DeepL features
        result = translator.translate_text(
            text,
            target_lang=target_lang,
            source_lang="SV",  # Swedish
            # New features from 2024
            context="Audio transcription from Swedish speech",  # Context parameter for better accuracy
            model_type="next_gen",  # Use next-generation models
            preserve_formatting=True,  # Preserve original formatting
            formality="default"  # Maintain appropriate formality level
        )
        
        return result.text
        
    except Exception as e:
        print(f"DeepL translation failed: {e}")
        # Fallback to OpenAI if DeepL fails
        return translate_text_openai(text, target_lang)


def translate_text_openai(text: str, target_lang: str = "EN-US") -> str:
    """
    Translate text using OpenAI GPT-4o as fallback.
    
    Args:
        text: Text to translate
        target_lang: Target language code (default: EN-US)
        
    Returns:
        Translated text
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        client = OpenAI(api_key=api_key)
        
        prompt = f"""Translate this Swedish text to English. Maintain the original formatting and preserve line breaks:

{text}"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional Swedish to English translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"OpenAI translation failed: {e}")
        # Return original text if all translation methods fail
        return text 
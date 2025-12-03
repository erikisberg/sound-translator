"""
Audio processing utilities for Swedish Audio Translator
"""

import os
import json
import logging
import gc
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, TypeVar
import httpx
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
from faster_whisper import WhisperModel
from pydub import AudioSegment
import streamlit as st

T = TypeVar('T')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DeepL Glossary ID (optional - for custom terminology)
# Set to None to disable glossary usage
DEEPL_GLOSSARY_ID = "ddb25cb9-deab-451c-add7-8da430b5adf4"


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    operation_name: str = "operation"
) -> T:
    """
    Retry a function with exponential backoff on failure.

    Args:
        func: Function to retry (should take no arguments, use lambda if needed)
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_factor: Multiplier for delay after each retry (default: 2.0)
        operation_name: Name of operation for logging

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError,
                APIConnectionError, APITimeoutError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(f"{operation_name} failed (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"{operation_name} failed after {max_retries} attempts: {str(e)}")
        except RateLimitError as e:
            # OpenAI rate limiting - use longer backoff
            rate_limit_delay = 5.0 * (backoff_factor ** attempt)  # 5s, 10s, 20s
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(f"{operation_name} rate limited (OpenAI). Retrying in {rate_limit_delay:.1f}s...")
                time.sleep(rate_limit_delay)
            else:
                logger.error(f"{operation_name} rate limited after {max_retries} attempts")
        except httpx.HTTPStatusError as e:
            # Handle rate limiting (429) and server errors (5xx) with retry
            if e.response.status_code == 429:
                # Rate limiting - use longer backoff
                rate_limit_delay = 5.0 * (backoff_factor ** attempt)  # 5s, 10s, 20s
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"{operation_name} rate limited (429). Retrying in {rate_limit_delay:.1f}s...")
                    time.sleep(rate_limit_delay)
                else:
                    logger.error(f"{operation_name} rate limited after {max_retries} attempts")
            elif e.response.status_code >= 500:
                # Server errors - retry with normal backoff
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"{operation_name} server error ({e.response.status_code}). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    logger.error(f"{operation_name} server error after {max_retries} attempts: {str(e)}")
            else:
                # Other HTTP errors (4xx) - don't retry
                logger.error(f"{operation_name} failed with non-retryable HTTP error: {str(e)}")
                raise
        except APIStatusError as e:
            # OpenAI API status errors (similar to HTTPStatusError)
            if e.status_code == 429:
                # Rate limiting
                rate_limit_delay = 5.0 * (backoff_factor ** attempt)
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"{operation_name} rate limited (OpenAI 429). Retrying in {rate_limit_delay:.1f}s...")
                    time.sleep(rate_limit_delay)
                else:
                    logger.error(f"{operation_name} rate limited after {max_retries} attempts")
            elif e.status_code >= 500:
                # Server errors
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"{operation_name} OpenAI server error ({e.status_code}). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    logger.error(f"{operation_name} OpenAI server error after {max_retries} attempts: {str(e)}")
            else:
                # Other status errors (4xx) - don't retry
                logger.error(f"{operation_name} failed with non-retryable OpenAI status error: {str(e)}")
                raise
        except Exception as e:
            # For other non-network errors, don't retry
            logger.error(f"{operation_name} failed with non-retryable error: {str(e)}")
            raise

    # If we get here, all retries failed
    raise last_exception


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
    logger.info(f"⚠️ CACHE MISS - Loading Whisper model from disk: {model_name} (device: {device}, compute_type: {compute_type})")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    logger.info(f"✅ Whisper model '{model_name}' loaded successfully and cached for future use")
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


def _call_openai_whisper_api(audio_path: str, language: str = "sv") -> List[Dict[str, Any]]:
    """
    Make a single API call to OpenAI Whisper.

    Args:
        audio_path: Path to audio file (<25MB)
        language: ISO 639-1 language code (default: "sv" for Swedish)

    Returns:
        List of segments with start, end, text

    Raises:
        ValueError: If OPENAI_API_KEY not found
        RuntimeError: If API call fails after retries
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        raise ValueError("OPENAI_API_KEY not found. Please set it in .env or secrets.toml")

    client = OpenAI(api_key=api_key)

    def transcribe_request():
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="verbose_json",  # Required for timestamps
                timestamp_granularities=["segment"]  # Get segment-level timestamps
            )
        return response

    # Use existing retry logic with longer delay for API calls
    logger.info(f"Calling OpenAI Whisper API for: {audio_path}")
    response = retry_with_backoff(
        transcribe_request,
        max_retries=3,
        initial_delay=2.0,  # Longer for API calls
        operation_name="OpenAI Whisper API transcription"
    )

    # Convert to same format as faster-whisper
    segments = []
    if hasattr(response, 'segments') and response.segments:
        for seg in response.segments:
            segments.append({
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": seg.text.strip()
            })
        logger.info(f"OpenAI Whisper API returned {len(segments)} segments")
    else:
        # Fallback if no segments (shouldn't happen with verbose_json)
        logger.warning("No segments in API response, creating single segment from full text")
        segments.append({
            "start": 0.0,
            "end": 0.0,
            "text": response.text if hasattr(response, 'text') else ""
        })

    return segments


def preprocess_for_openai(audio_path: str, working_dir: Path) -> Optional[str]:
    """
    Compress audio if too large for OpenAI API (25MB limit).

    Args:
        audio_path: Path to audio file
        working_dir: Directory for intermediate files

    Returns:
        Path to compressed audio, or None if still too large (needs splitting)
    """
    file_size_mb = Path(audio_path).stat().st_size / (1024 * 1024)

    if file_size_mb <= 24:
        logger.info(f"File size {file_size_mb:.1f}MB is under 25MB limit, no compression needed")
        return audio_path

    logger.info(f"File size {file_size_mb:.1f}MB exceeds 25MB limit, compressing...")

    try:
        audio = AudioSegment.from_file(audio_path)
        compressed_path = working_dir / "compressed_for_api.mp3"

        # Export as compressed MP3 (64kbps mono is good for speech)
        audio.export(
            compressed_path,
            format="mp3",
            bitrate="64k",
            parameters=["-ac", "1"]  # Mono audio
        )

        new_size_mb = compressed_path.stat().st_size / (1024 * 1024)
        logger.info(f"Compressed from {file_size_mb:.1f}MB to {new_size_mb:.1f}MB")

        if new_size_mb > 24:
            logger.warning(f"Compressed size {new_size_mb:.1f}MB still exceeds limit, will need to split")
            return None  # Trigger splitting logic

        return str(compressed_path)

    except Exception as e:
        logger.error(f"Audio compression failed: {e}", exc_info=True)
        return None  # Trigger splitting logic as fallback


def _transcribe_split_audio(audio_path: str, working_dir: Path, language: str = "sv") -> List[Dict[str, Any]]:
    """
    Split large audio files into chunks and transcribe each with OpenAI API.

    Args:
        audio_path: Path to audio file (>25MB)
        working_dir: Directory for intermediate files
        language: ISO 639-1 language code

    Returns:
        List of merged segments with adjusted timestamps
    """
    logger.info(f"Splitting audio file for OpenAI API: {audio_path}")

    audio = AudioSegment.from_file(audio_path)
    total_duration_ms = len(audio)

    # Calculate chunk size (10 minutes = 600,000ms)
    chunk_duration_ms = 10 * 60 * 1000
    num_chunks = (total_duration_ms + chunk_duration_ms - 1) // chunk_duration_ms  # Ceiling division

    logger.info(f"Splitting {total_duration_ms/1000:.1f}s audio into {num_chunks} chunks of ~10 minutes each")

    all_segments = []
    cumulative_offset = 0.0

    for i in range(num_chunks):
        start_ms = i * chunk_duration_ms
        end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)

        chunk = audio[start_ms:end_ms]
        chunk_path = working_dir / f"chunk_{i:03d}.mp3"

        # Export as compressed MP3 to stay under 25MB
        chunk.export(chunk_path, format="mp3", bitrate="64k", parameters=["-ac", "1"])

        chunk_size_mb = chunk_path.stat().st_size / (1024 * 1024)
        logger.info(f"Processing chunk {i+1}/{num_chunks}: {chunk_size_mb:.1f}MB")

        # Transcribe this chunk
        try:
            segments = _call_openai_whisper_api(str(chunk_path), language)

            # Adjust timestamps by cumulative offset
            for seg in segments:
                seg["start"] += cumulative_offset
                seg["end"] += cumulative_offset
                all_segments.append(seg)

            # Update offset for next chunk
            cumulative_offset += len(chunk) / 1000.0

            # Clean up chunk file
            chunk_path.unlink()

        except Exception as e:
            logger.error(f"Chunk {i+1} transcription failed: {e}", exc_info=True)
            # Clean up and re-raise
            if chunk_path.exists():
                chunk_path.unlink()
            raise RuntimeError(f"Chunk {i+1} transcription failed: {e}")

    logger.info(f"Split transcription completed: {len(all_segments)} total segments from {num_chunks} chunks")
    return all_segments


def transcribe_with_openai_whisper(audio_path: str, working_dir: Path, language: str = "sv") -> List[Dict[str, Any]]:
    """
    Transcribe audio using OpenAI Whisper API.

    Handles:
    - Files up to 25MB (direct API call)
    - Files >25MB (compression, then splitting if needed)
    - Swedish language specification
    - Segment-level timestamps
    - Rate limiting with retry logic

    Args:
        audio_path: Path to audio file
        working_dir: Directory for intermediate files
        language: ISO 639-1 language code (default: "sv" for Swedish)

    Returns:
        List of segments with start, end, text (same format as faster-whisper)

    Raises:
        ValueError: If OPENAI_API_KEY not found
        RuntimeError: If transcription fails after retries
    """
    logger.info(f"Starting OpenAI Whisper API transcription for: {audio_path}")

    # Check and compress file if needed
    processed_path = preprocess_for_openai(audio_path, working_dir)

    if processed_path is None:
        # File too large even after compression, need to split
        logger.info("File requires splitting into chunks")
        return _transcribe_split_audio(audio_path, working_dir, language)

    # File is small enough, transcribe directly
    segments = _call_openai_whisper_api(processed_path, language)

    # Save transcript
    transcript_path = working_dir / "transcript.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    logger.info(f"Transcript saved to: {transcript_path}")

    logger.info(f"OpenAI Whisper transcription completed: {len(segments)} segments detected")
    return segments


def transcribe_file(audio_path: str, working_dir: Path, model_name: str = "large-v3-turbo", use_api: bool = False) -> List[Dict[str, Any]]:
    """
    Transcribe audio file using OpenAI Whisper API (default) or local faster-whisper.

    Args:
        audio_path: Path to audio file
        working_dir: Directory for intermediate files
        model_name: Whisper model to use for local transcription (default: large-v3-turbo)
        use_api: If True, use OpenAI Whisper API. If False, use local faster-whisper (default: False)

    Returns:
        List of segments with start, end, text
    """
    logger.info(f"Starting transcription for: {audio_path}")

    # Use OpenAI API if requested
    if use_api:
        logger.info("Using OpenAI Whisper API for transcription")
        return transcribe_with_openai_whisper(audio_path, working_dir, language="sv")

    # Use local faster-whisper
    logger.info(f"Using local faster-whisper model: {model_name}")

    # Preprocess audio for better detection
    try:
        processed_audio_path = preprocess_audio(audio_path, working_dir)
        logger.info(f"Audio preprocessed successfully: {processed_audio_path}")
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}", exc_info=True)
        raise

    try:
        # Use cached model - prevents memory leaks and improves performance
        logger.info(f"Requesting Whisper model '{model_name}' from cache...")
        model = get_whisper_model(model_name)
        logger.info(f"✅ Got model '{model_name}' (cached or freshly loaded)")

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
    """Translate using DeepL API with retry logic and connection reuse."""
    logger.info("Using DeepL for translation")
    if DEEPL_GLOSSARY_ID:
        logger.info(f"DeepL glossary enabled: {DEEPL_GLOSSARY_ID}")

    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        logger.error("DEEPL_API_KEY not found in environment")
        raise ValueError("DEEPL_API_KEY not found")

    url = "https://api-free.deepl.com/v2/translate"
    headers = {"Authorization": f"DeepL-Auth-Key {api_key}"}

    translated_segments = []

    # Reuse HTTP client for all requests (better performance and reliability)
    with httpx.Client(timeout=30.0) as client:
        for i, segment in enumerate(segments):
            try:
                logger.debug(f"Translating segment {i+1}/{len(segments)}")
                data = {
                    "text": segment["text"],
                    "source_lang": "SV",      # Required for glossary
                    "target_lang": "EN-US",
                    "preserve_formatting": "1"
                }

                # Add glossary if configured (optional)
                if DEEPL_GLOSSARY_ID:
                    data["glossary_id"] = DEEPL_GLOSSARY_ID

                # Use retry with exponential backoff for network resilience
                def translate_request():
                    response = client.post(url, headers=headers, data=data)
                    response.raise_for_status()
                    return response.json()

                result = retry_with_backoff(
                    translate_request,
                    max_retries=3,
                    initial_delay=1.0,
                    operation_name=f"DeepL translation (segment {i+1}/{len(segments)})"
                )

                english_text = result["translations"][0]["text"]

                translated_segment = segment.copy()
                translated_segment["english"] = english_text
                translated_segments.append(translated_segment)

                # Rate limiting: Add delay to stay under DeepL Free API limits (~3 req/s)
                # This prevents hitting 429 Too Many Requests errors
                if i < len(segments) - 1:  # Don't delay after last segment
                    time.sleep(0.3)  # Max ~3 requests/second

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
    
    # Extract voice settings with defaults (optimized for long-form consistency per ElevenLabs recommendations)
    speaking_rate = voice_settings.get("speaking_rate", float(os.getenv("ELEVEN_SPEAKING_RATE", "1.0")))
    stability = voice_settings.get("stability", 0.50)  # ElevenLabs recommended: ~50 for consistency
    similarity_boost = voice_settings.get("similarity_boost", 0.75)  # ElevenLabs recommended: ~75
    style = voice_settings.get("style", 0.0)  # ElevenLabs recommended: 0 for long-form to prevent drift
    use_speaker_boost = voice_settings.get("use_speaker_boost", True)
    voice_model = voice_settings.get("voice_model", "eleven_multilingual_v2")  # Most stable for long-form

    logger.info(f"Voice settings: voice_id={voice_id}, model={voice_model}, rate={speaking_rate}")

    # Request stitching settings for voice consistency
    use_request_stitching = voice_settings.get("use_request_stitching", True)
    context_window_size = voice_settings.get("context_window_size", 5)  # Increased from 3 for longer sessions

    # output_format is a query parameter, not body parameter
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?output_format=mp3_44100_128"
    headers = {
        "Accept": "audio/mpeg",
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    audio_files = []
    tts_dir = working_dir / "tts_clips"
    tts_dir.mkdir(exist_ok=True)

    # Track previous segments for request stitching (ElevenLabs consistency feature)
    previous_texts = []
    previous_request_ids = []

    # Reuse HTTP client for all requests (better performance and reliability)
    with httpx.Client(timeout=30.0) as client:
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
                    "apply_text_normalization": "auto"  # Automatic text normalization
                }

                # Add seed if provided (for reproducible voice generation)
                seed_value = voice_settings.get("seed")
                if seed_value is not None:
                    data["seed"] = int(seed_value)

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

                    # Add next_text lookahead for smoother transitions (reduces end-of-session drift)
                    if i < len(segments) - 1:
                        next_texts = []
                        for j in range(i + 1, min(i + 1 + context_window_size, len(segments))):
                            next_seg_text = segments[j].get("english", "").strip()
                            if next_seg_text:
                                next_texts.append(next_seg_text)
                        if next_texts:
                            data["next_text"] = " ".join(next_texts)

                # Add speed control for supported models
                # Note: Not all models support speed parameter
                # Safe models: eleven_turbo_v2_5, eleven_flash_v2_5, eleven_multilingual_v2
                if voice_model in ["eleven_turbo_v2_5", "eleven_flash_v2_5", "eleven_multilingual_v2", "eleven_multilingual_v2_5"]:
                    # Speed control: Valid range is 0.7-1.2 (1.0 = normal, 0.7 = slowest, 1.2 = fastest)
                    speed = max(0.7, min(1.2, speaking_rate))
                    data["voice_settings"]["speed"] = speed
                    logger.info(f"Using speed={speed} for model {voice_model}")

                # Use retry with exponential backoff for network resilience
                def tts_request():
                    response = client.post(url, headers=headers, json=data)
                    response.raise_for_status()
                    return response

                response = retry_with_backoff(
                    tts_request,
                    max_retries=3,
                    initial_delay=1.0,
                    operation_name=f"ElevenLabs TTS (segment {i+1}/{len(segments)})"
                )

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

    # Track max volume for normalization (replaces audio_segments array to save memory)
    max_volume_db = float('-inf')

    # Target loudness for per-segment normalization (in dBFS)
    # -16 dBFS is a good target for speech (leaves headroom, consistent loudness)
    TARGET_LOUDNESS_DBFS = -16.0

    for i, (segment, audio_file) in enumerate(zip(segments, audio_files)):
        # Load generated audio
        generated_audio = AudioSegment.from_file(audio_file)

        # Normalize each segment individually to target loudness BEFORE stitching
        # This fixes the ElevenLabs volume drift issue where later segments are quieter
        if generated_audio.dBFS > float('-inf') and len(generated_audio) > 100:
            loudness_diff = TARGET_LOUDNESS_DBFS - generated_audio.dBFS
            generated_audio = generated_audio.apply_gain(loudness_diff)
            logger.debug(f"Segment {i+1}: normalized by {loudness_diff:+.1f}dB to {TARGET_LOUDNESS_DBFS}dBFS")

        # Track max volume for final normalization
        if generated_audio.max_dBFS > float('-inf'):
            max_volume_db = max(max_volume_db, generated_audio.max_dBFS)
        
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

        print(f"Segment {i+1}: starts at {segment['start']:.1f}s, TTS duration: {len(generated_audio)/1000:.1f}s, total audio length: {len(final_audio)/1000:.1f}s")

    # Normalize all segments for consistent volume
    print("Normalizing audio segments for consistent volume...")
    final_audio = normalize_audio_segments(final_audio, max_volume_db)

    # Calculate final duration
    final_duration_s = len(final_audio) / 1000
    print(f"\nFinal audio duration: {final_duration_s:.1f}s ({final_duration_s/60:.1f} minutes)")

    # Export final audio
    output_path = working_dir / "final_audio.wav"
    final_audio.export(output_path, format="wav")

    return output_path


def normalize_audio_segments(final_audio: AudioSegment, max_volume_db: float) -> AudioSegment:
    """
    Final normalization pass to ensure consistent volume levels.

    Note: Per-segment normalization is done in stitch_segments() BEFORE stitching.
    This function handles final peak normalization and gentle compression.

    Args:
        final_audio: The complete audio to normalize
        max_volume_db: Maximum volume level in dB from all segments

    Returns:
        Normalized audio segment
    """
    try:
        print(f"Input audio level: {final_audio.dBFS:.1f}dBFS (max peak: {max_volume_db:.1f}dB)")

        # Peak normalize to -1dBFS (leaves headroom, prevents clipping)
        if final_audio.max_dBFS > float('-inf'):
            peak_headroom = -1.0 - final_audio.max_dBFS
            normalized_audio = final_audio.apply_gain(peak_headroom)
            print(f"Peak normalized by {peak_headroom:+.1f}dB to -1dBFS")
        else:
            normalized_audio = final_audio

        # Apply compression to even out any remaining volume variations
        # Using moderate settings that won't introduce artifacts
        normalized_audio = normalized_audio.compress_dynamic_range(
            threshold=-18.0,  # Start compressing at -18dBFS
            ratio=2.5,        # Moderate compression (increased from 1.5)
            attack=10.0,      # 10ms attack (natural for speech)
            release=100.0     # 100ms release (smooth recovery)
        )

        print(f"Final audio level: {normalized_audio.dBFS:.1f}dBFS")
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
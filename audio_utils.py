"""
Audio processing utilities for Swedish Audio Translator
"""

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx
from openai import OpenAI
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range

# Make torch optional - it's only used for CUDA detection (not available on macOS anyway)
try:
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"Warning: torch import failed ({e}), defaulting to CPU device")
    TORCH_AVAILABLE = False


def preprocess_audio(audio_path: str, working_dir: Path) -> str:
    """
    Preprocess audio to improve speech detection.
    
    Args:
        audio_path: Path to original audio file
        working_dir: Directory for processed files
        
    Returns:
        Path to preprocessed audio file
    """
    try:
        print(f"Loading audio file: {audio_path}")
        
        # Load audio with error handling
        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as load_error:
            print(f"Failed to load with pydub, trying different approach: {load_error}")
            # Try loading as specific format
            if audio_path.lower().endswith('.mp3'):
                audio = AudioSegment.from_mp3(audio_path)
            elif audio_path.lower().endswith('.wav'):
                audio = AudioSegment.from_wav(audio_path)
            else:
                raise load_error
        
        print(f"Original audio: {len(audio)}ms duration, {audio.frame_rate}Hz, {audio.channels} channels")
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
            print("Converted to mono")
        
        # Normalize volume to improve quiet speech detection
        try:
            audio = normalize(audio)
            print("Audio normalized")
        except Exception as norm_error:
            print(f"Normalization failed, continuing: {norm_error}")
        
        # Apply gentle compression to even out volume levels
        try:
            audio = compress_dynamic_range(audio, threshold=-20.0, ratio=2.0, attack=5.0, release=50.0)
            print("Compression applied")
        except Exception as comp_error:
            print(f"Compression failed, continuing: {comp_error}")
        
        # High-pass filter to reduce low-frequency noise
        try:
            audio = audio.high_pass_filter(80)
            print("High-pass filter applied")
        except Exception as filter_error:
            print(f"High-pass filter failed, continuing: {filter_error}")
        
        # Convert to WAV format for consistent processing
        processed_path = working_dir / "preprocessed_audio.wav"
        
        # Export with consistent format
        audio.export(
            processed_path, 
            format="wav", 
            parameters=[
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",      # Mono
                "-sample_fmt", "s16"  # 16-bit samples
            ]
        )
        
        print(f"Audio preprocessed successfully: {processed_path}")
        print(f"Final specs: {len(audio)}ms duration, 16000Hz sample rate")
        return str(processed_path)
        
    except Exception as e:
        print(f"Audio preprocessing failed, using original: {str(e)}")
        # If preprocessing fails completely, return original path
        return audio_path


def transcribe_file(audio_path: str, working_dir: Path) -> List[Dict[str, Any]]:
    """
    Transcribe audio file using faster-whisper with improved speech detection.
    
    Args:
        audio_path: Path to audio file
        working_dir: Directory for intermediate files
        
    Returns:
        List of segments with start, end, text
    """
    # Preprocess audio for better detection
    processed_audio_path = preprocess_audio(audio_path, working_dir)

    # Try GPU first, fallback to CPU
    device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    try:
        model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
        
        # Start with basic transcription settings that are known to work
        try:
            # Try with advanced settings first
            segments, info = model.transcribe(
                processed_audio_path,
                language="sv",  # Swedish
                beam_size=5,
                best_of=5,
                temperature=0.0,
                # More conservative VAD settings
                vad_filter=True,
                # Lower detection threshold for quiet speech
                no_speech_threshold=0.4,  # Lower = more sensitive (default 0.6)
                compression_ratio_threshold=2.4,
                condition_on_previous_text=True
            )
        except Exception as vad_error:
            print(f"Advanced settings failed: {vad_error}")
            print("Falling back to basic transcription settings...")
            
            # Fallback to basic settings without VAD parameters
            segments, info = model.transcribe(
                processed_audio_path,
                language="sv",  # Swedish
                beam_size=5,
                temperature=0.0,
                no_speech_threshold=0.4
            )
        
        # Convert to list format with additional validation
        segment_list = []
        for segment in segments:
            text = segment.text.strip()
            # Only include segments with actual speech content
            if text and len(text) > 1:  # Skip very short or empty segments
                segment_data = {
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": text
                }
                
                # Add word-level data if available (optional)
                try:
                    if hasattr(segment, 'words') and segment.words:
                        segment_data["words"] = [
                            {
                                "start": round(word.start, 3),
                                "end": round(word.end, 3),
                                "word": word.word,
                                "probability": round(word.probability, 3) if hasattr(word, 'probability') else None
                            }
                            for word in segment.words
                        ]
                except Exception as word_error:
                    print(f"Word-level timestamps not available: {word_error}")
                
                segment_list.append(segment_data)
        
        # Log transcription info
        print(f"Transcription completed: {len(segment_list)} segments detected")
        print(f"Language detected: {info.language} (probability: {info.language_probability:.2f})")
        print(f"Total duration: {info.duration:.2f}s")
        
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
        
        return segment_list
        
    except Exception as e:
        print(f"Transcription error details: {str(e)}")
        print(f"Processed audio path: {processed_audio_path}")
        print(f"Device: {device}, Compute type: {compute_type}")
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
    if service == "deepl":
        return _translate_with_deepl(segments, working_dir)
    elif service == "openai":
        return _translate_with_openai(segments, working_dir)
    else:
        raise ValueError(f"Unsupported translation service: {service}")


def _translate_with_deepl(segments: List[Dict[str, Any]], working_dir: Path) -> List[Dict[str, Any]]:
    """Translate using DeepL API."""
    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        raise ValueError("DEEPL_API_KEY not found")
    
    url = "https://api-free.deepl.com/v2/translate"
    headers = {"Authorization": f"DeepL-Auth-Key {api_key}"}
    
    translated_segments = []
    
    for i, segment in enumerate(segments):
        try:
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
            raise RuntimeError(f"DeepL translation failed for segment {i}: {str(e)}")
    
    # Save translations
    translation_path = working_dir / "translations.json"
    with open(translation_path, "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)
    
    return translated_segments


def _translate_with_openai(segments: List[Dict[str, Any]], working_dir: Path) -> List[Dict[str, Any]]:
    """Translate using OpenAI GPT-4o."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    client = OpenAI(api_key=api_key)
    translated_segments = []
    
    # Batch translate for efficiency
    swedish_texts = [seg["text"] for seg in segments]
    prompt = f"""Translate these Swedish text segments to English. Maintain the original formatting and preserve line breaks. Return only the English translations, one per line:

{chr(10).join(swedish_texts)}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional Swedish to English translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        english_lines = response.choices[0].message.content.strip().split('\n')
        
        # Match translations to segments
        for i, segment in enumerate(segments):
            english_text = english_lines[i] if i < len(english_lines) else ""
            translated_segment = segment.copy()
            translated_segment["english"] = english_text
            translated_segments.append(translated_segment)
            
    except Exception as e:
        raise RuntimeError(f"OpenAI translation failed: {str(e)}")
    
    # Save translations
    translation_path = working_dir / "translations.json"
    with open(translation_path, "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)
    
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
    api_key = os.getenv("ELEVEN_API_KEY")
    
    # Use voice_id from settings or fallback to environment variable
    if voice_settings and voice_settings.get("voice_id"):
        voice_id = voice_settings["voice_id"]
    else:
        voice_id = os.getenv("ELEVEN_VOICE_ID")
    
    if not api_key or not voice_id:
        raise ValueError("ELEVEN_API_KEY or voice_id not found")
    
    # Use provided voice settings or defaults
    if voice_settings is None:
        voice_settings = {}
    
    # Extract voice settings with defaults (based on ElevenLabs best practices)
    speaking_rate = voice_settings.get("speaking_rate", float(os.getenv("ELEVEN_SPEAKING_RATE", "0.7")))
    stability = voice_settings.get("stability", 0.5)  # ElevenLabs recommended: ~50 for consistency
    similarity_boost = voice_settings.get("similarity_boost", 0.75)  # ElevenLabs recommended: ~75
    style = voice_settings.get("style", 0.0)
    use_speaker_boost = voice_settings.get("use_speaker_boost", True)
    voice_model = voice_settings.get("voice_model", "eleven_monolingual_v1")

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
        english_text = segment.get("english", "").strip()
        if not english_text:
            # Create silent audio for empty segments
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
            
            # Add speaking rate for compatible models (v3, v2.5 and turbo models)
            if "v3" in voice_model or "turbo" in voice_model or "v2" in voice_model:
                # Speaking rate control (0.25 - 4.0)
                rate = max(0.25, min(4.0, speaking_rate))
                data["voice_settings"]["speaking_rate"] = rate
            
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=headers, json=data)
                response.raise_for_status()

                # Extract request-id from response headers for request stitching
                request_id = response.headers.get("request-id")
                if request_id and use_request_stitching:
                    previous_request_ids.append(request_id)
                    print(f"  → Captured request-id for segment {i+1}: {request_id[:8]}...")

                # Save audio file
                file_path = tts_dir / f"segment_{i:03d}.wav"
                with open(file_path, "wb") as f:
                    f.write(response.content)

                audio_files.append(file_path)

                # Add current text to context history for next segments
                if use_request_stitching:
                    previous_texts.append(english_text)

        except Exception as e:
            raise RuntimeError(f"TTS generation failed for segment {i}: {str(e)}")
    
    return audio_files


def enhance_audio_segment(audio_file: Path, working_dir: Path) -> Path:
    """
    Enhance audio quality using available enhancement services.
    Supports Dolby.io (premium) and basic AudioSegment normalization (free).
    
    Args:
        audio_file: Path to audio file to enhance
        working_dir: Directory for enhanced files
        
    Returns:
        Path to enhanced audio file
    """
    dolby_api_key = os.getenv("DOLBY_API_KEY")
    
    enhanced_dir = working_dir / "enhanced_clips"
    enhanced_dir.mkdir(exist_ok=True)
    
    # Method 1: Dolby.io Professional Enhancement
    if dolby_api_key:
        try:
            with open(audio_file, "rb") as f:
                audio_data = f.read()
            
            url = "https://api.dolby.com/media/enhance"
            headers = {
                "Authorization": f"Bearer {dolby_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            payload = {
                "input": base64.b64encode(audio_data).decode(),
                "output": {"format": "wav"},
                "filter": {
                    "enhance": {
                        "type": "speech",
                        "amount": "medium",
                        "dynamics": {"range_control": {"enable": True, "amount": "medium"}},
                        "noise": {"reduction": {"enable": True, "amount": "medium"}}
                    }
                }
            }
            
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    enhanced_audio = base64.b64decode(result["output"])
                    
                    enhanced_file = enhanced_dir / f"dolby_enhanced_{audio_file.name}"
                    with open(enhanced_file, "wb") as f:
                        f.write(enhanced_audio)
                    
                    return enhanced_file
                    
        except Exception as e:
            print(f"Dolby.io enhancement failed: {str(e)}")
    
    # Method 2: Basic Enhancement (free, using pydub)
    try:
        audio = AudioSegment.from_file(audio_file)
        
        # Basic audio improvements
        enhanced_audio = audio
        
        # Normalize volume to prevent clipping
        enhanced_audio = enhanced_audio.normalize()
        
        # Apply gentle compression to even out levels
        enhanced_audio = enhanced_audio.compress_dynamic_range(threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
        
        # Slight high-pass filter to reduce low-frequency noise (remove frequencies below 80Hz)
        enhanced_audio = enhanced_audio.high_pass_filter(80)
        
        # Save enhanced audio
        enhanced_file = enhanced_dir / f"basic_enhanced_{audio_file.name}"
        enhanced_audio.export(enhanced_file, format="wav")
        
        return enhanced_file
        
    except Exception as e:
        print(f"Basic audio enhancement failed: {str(e)}")
        return audio_file


def stitch_segments(
    segments: List[Dict[str, Any]],
    audio_files: List[Path],
    working_dir: Path,
    enable_enhancement: bool = True,
    crossfade_duration_ms: int = 75
) -> Path:
    """
    Stitch TTS audio segments using start time only with audio normalization and crossfading.
    Prevents premature cutting and ensures consistent volume levels.

    Args:
        segments: Original segments with timing
        audio_files: Generated TTS audio files
        working_dir: Directory for output
        enable_enhancement: Whether to enhance audio quality
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
        # Optionally enhance audio quality
        if enable_enhancement:
            print(f"Enhancing audio segment {i+1}/{len(segments)}...")
            audio_file = enhance_audio_segment(audio_file, working_dir)
        
        # Load generated audio
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
            
            if original_gap_ms > 0:
                print(f"Adding {original_gap_ms/1000:.1f}s original gap before segment {i+1}")
                final_audio += AudioSegment.silent(duration=original_gap_ms)
            elif original_gap_ms < 0:
                # Segments overlap in original - add small buffer
                buffer_ms = 100  # 100ms buffer for overlapping segments
                print(f"Original segments overlapped by {abs(original_gap_ms)/1000:.1f}s, adding {buffer_ms}ms buffer")
                final_audio += AudioSegment.silent(duration=buffer_ms)

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
    print(f"Final audio duration: {final_duration_s:.1f}s ({final_duration_s/60:.1f} minutes)")
    
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
        
        # Apply gentle compression to even out volume levels
        normalized_audio = normalized_audio.compress_dynamic_range(
            threshold=-20.0, 
            ratio=3.0, 
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
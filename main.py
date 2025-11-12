"""
Swedish Audio Translator - Streamlit App
Transcribe Swedish audio → Translate to English → Generate English TTS
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv

from audio_utils import (
    transcribe_file,
    translate_segments,
    generate_tts,
    stitch_segments
)

# Load environment variables
load_dotenv()

# App configuration
st.set_page_config(
    page_title="Swedish Audio Translator",
            page_icon="",
    layout="wide"
)

def init_session_state() -> None:
    """Initialize session state variables."""
    if "segments" not in st.session_state:
        st.session_state.segments = []
    if "translated_segments" not in st.session_state:
        st.session_state.translated_segments = []
    if "final_audio_path" not in st.session_state:
        st.session_state.final_audio_path = None
    if "working_dir" not in st.session_state:
        st.session_state.working_dir = Path(tempfile.mkdtemp(prefix="audio_translator_"))
        st.session_state.working_dir.mkdir(exist_ok=True)

def create_segments_dataframe(segments: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a DataFrame from segments for editing."""
    if not segments:
        return pd.DataFrame(columns=["Start", "End", "Swedish", "English"])
    
    df_data = []
    for i, seg in enumerate(segments):
        english_text = ""
        if i < len(st.session_state.translated_segments):
            english_text = st.session_state.translated_segments[i].get("english", "")
        
        df_data.append({
            "Start": f"{seg['start']:.2f}s",
            "End": f"{seg['end']:.2f}s", 
            "Swedish": seg["text"],
            "English": english_text
        })
    
    return pd.DataFrame(df_data)

def main():
    """Main Streamlit application."""
    init_session_state()
    
    st.title("Swedish Audio Translator")
    st.markdown("Upload Swedish audio → Transcribe → Translate → Generate English TTS")
    
    # API Key Status
    with st.sidebar:
        st.header("API Configuration")
        
        # Check API keys
        openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        deepl_key = st.secrets.get("DEEPL_API_KEY") or os.getenv("DEEPL_API_KEY")
        eleven_key = st.secrets.get("ELEVEN_API_KEY") or os.getenv("ELEVEN_API_KEY")
        eleven_voice = st.secrets.get("ELEVEN_VOICE_ID") or os.getenv("ELEVEN_VOICE_ID")
        
        st.write("**Translation Service:**")
        if deepl_key:
            st.success("DeepL API Key found")
            translation_service = "deepl"
        elif openai_key:
            st.success("OpenAI API Key found")
            translation_service = "openai"
        else:
            st.error("No translation API key found")
            translation_service = None
            
        st.write("**TTS Service:**")
        if eleven_key:
            st.success("ElevenLabs API Key found")
            if eleven_voice:
                st.success(f"Voice ID: {eleven_voice}")
            else:
                st.warning("Voice ID not configured")
        else:
            st.error("ElevenLabs API key missing")
            
        st.write(f"**Working Directory:** `{st.session_state.working_dir}`")

        # Whisper Model Selection
        st.header("Transcription Settings")
        whisper_model = st.selectbox(
            "Whisper Model",
            options=[
                "large-v3-turbo",  # Best quality, slowest
                "medium",          # Good balance, 2x faster
                "small",           # OK quality, 3x faster
                "base",            # Basic, 5x faster
                "tiny"             # Fastest, 8x faster
            ],
            index=1,
            help="Choose transcription speed vs quality trade-off"
        )

        # Model info
        model_info = {
            "large-v3-turbo": "Best quality, ~1.5GB, slowest",
            "medium": "Good balance, ~770MB, 2x faster",
            "small": "OK quality, ~490MB, 3x faster",
            "base": "Basic, ~140MB, 5x faster",
            "tiny": "Fastest, ~75MB, 8x faster"
        }
        st.info(f"**{whisper_model}**: {model_info[whisper_model]}")

        # Store in session state
        st.session_state.whisper_model = whisper_model

        # ElevenLabs Voice Settings
        if eleven_key and eleven_voice:
            st.header("Voice Settings")
            
            # Speaking Rate
            speaking_rate = st.slider(
                "Speaking Rate",
                min_value=0.7,
                max_value=1.2,
                value=float(st.secrets.get("ELEVEN_SPEAKING_RATE", "1.0")),
                step=0.05,
                help="Adjust speech speed: 0.7 = slowest, 1.0 = normal (default), 1.2 = fastest"
            )
            
            # Professional Voice Clone Optimization Info
            st.info("**🎙️ Optimized for Professional Voice Clones:**\n"
                   "Settings are pre-configured for natural, expressive output with cloned voices.\n"
                   "• **Stability (0.65)**: Balanced variation for natural speech\n"
                   "• **Similarity (0.85)**: Tight match to cloned voice characteristics\n"
                   "• **Style (0.4)**: Moderate expressiveness to reduce 'robotic' feel\n"
                   "• **Model (multilingual_v2)**: Best prosody + request stitching support")

            # Advanced Voice Settings
            with st.expander("Advanced Voice Settings"):
                # Voice ID Selection
                voice_id = st.selectbox(
                    "Voice ID",
                    options=[
                        "PqclsDjBR66GIIQ7oVAc",  # Your custom voice (default)
                        eleven_voice if eleven_voice and eleven_voice != "PqclsDjBR66GIIQ7oVAc" else None,  # Current configured voice if different
                        "21m00Tcm4TlvDq8ikWAM",  # Rachel (Professional female)
                        "AZnzlk1XvdvUeBnXmlld",  # Domi (Professional female)
                        "EXAVITQu4vr4xnSDxMaL",  # Bella (Professional female)
                        "VR6AewLTigWG4xSOukaG",  # Josh (Professional male)
                        "pNInz6obpgDQGcFmaJgB",  # Adam (Professional male)
                        "yoZ06aMxZJJ28mfd3POQ",  # Sam (Professional male)
                        "custom"  # Custom voice ID input
                    ],
                    index=0,
                    help="Choose from popular ElevenLabs voices or enter a custom voice ID"
                )
                
                # Custom Voice ID Input
                if voice_id == "custom":
                    custom_voice_id = st.text_input(
                        "Custom Voice ID",
                        value="",
                        help="Enter your custom ElevenLabs voice ID (found in your ElevenLabs dashboard)"
                    )
                    if custom_voice_id:
                        voice_id = custom_voice_id
                    
                    st.info("**How to find your Voice ID**: Go to ElevenLabs Dashboard → Voice Library → Click on a voice → Copy the Voice ID from the URL or voice details")
                
                # Voice ID Display
                if voice_id and voice_id != "Select a voice":
                    st.success(f"Selected Voice ID: `{voice_id}`")
                    
                    # Voice Preview Information
                    voice_info = {
                        "PqclsDjBR66GIIQ7oVAc": "Your Custom Voice - Personalized voice for your brand",
                        "21m00Tcm4TlvDq8ikWAM": "Rachel - Professional female voice, clear and articulate",
                        "AZnzlk1XvdvUeBnXmlld": "Domi - Professional female voice, warm and engaging",
                        "EXAVITQu4vr4xnSDxMaL": "Bella - Professional female voice, confident and clear",
                        "VR6AewLTigWG4xSOukaG": "Josh - Professional male voice, deep and authoritative",
                        "pNInz6obpgDQGcFmaJgB": "Adam - Professional male voice, clear and professional",
                        "yoZ06aMxZJJ28mfd3POQ": "Sam - Professional male voice, friendly and approachable"
                    }
                    
                    if voice_id in voice_info:
                        st.info(f"**Voice Preview**: {voice_info[voice_id]}")
                    elif voice_id == "custom":
                        st.info("**Custom Voice**: Using your custom voice ID")
                    else:
                        st.info("**Voice**: Using configured voice from environment")
                
                # Voice Model Selection
                voice_model = st.selectbox(
                    "Voice Model",
                    options=[
                        "eleven_multilingual_v2",   # ⭐ Recommended for PVC - 29 languages, 10k chars
                        "eleven_flash_v2_5",        # Ultra-fast (~75ms) - 32 languages, 40k chars
                        "eleven_turbo_v2_5",        # Balanced quality/speed - 32 languages, 40k chars
                        "eleven_v3",                # Most expressive (alpha) - 70+ languages, 3k chars
                        "eleven_turbo_v2",          # Legacy turbo (English only)
                        "eleven_flash_v2",          # Legacy flash (English only)
                    ],
                    index=0,
                    help="For Professional Voice Clones: multilingual_v2 recommended (best prosody + request stitching support)"
                )

                # Model Information
                if voice_model == "eleven_multilingual_v2":
                    st.success("**eleven_multilingual_v2**: ⭐ RECOMMENDED for PVC\n"
                              "• Best prosody and natural expression\n"
                              "• Request stitching support for voice consistency\n"
                              "• 29 languages, 10,000 character limit\n"
                              "• Most stable for long-form content")
                elif voice_model == "eleven_flash_v2_5":
                    st.info("**eleven_flash_v2_5**: Ultra-fast model (~75ms latency)\n"
                           "• Perfect for real-time applications and Agents\n"
                           "• 32 languages supported\n"
                           "• 40,000 character limit\n"
                           "• 50% lower cost per character")
                elif voice_model == "eleven_turbo_v2_5":
                    st.info("**eleven_turbo_v2_5**: Balanced quality and speed\n"
                           "• Good balance between quality and latency (~250-300ms)\n"
                           "• 32 languages supported\n"
                           "• 40,000 character limit\n"
                           "• 50% lower cost per character")
                elif voice_model == "eleven_v3":
                    st.warning("**eleven_v3**: Most emotionally expressive (alpha)\n"
                              "• Human-like speech with high emotional range\n"
                              "• 70+ languages supported\n"
                              "• ⚠️ 3,000 character limit (~3 minutes)\n"
                              "• ⚠️ NO request stitching support - may have voice inconsistency\n"
                              "• Best for audiobooks, character dialogue, emotional content")
                elif voice_model == "eleven_turbo_v2":
                    st.info("**eleven_turbo_v2**: Legacy model (English only)\n"
                           "• Consider upgrading to eleven_turbo_v2_5 for multilingual support")
                elif voice_model == "eleven_flash_v2":
                    st.info("**eleven_flash_v2**: Legacy model (English only)\n"
                           "• Consider upgrading to eleven_flash_v2_5 for multilingual support")
                
                stability = st.slider(
                    "Stability",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.65,
                    step=0.05,
                    help="Lower = more variation, Higher = more consistent. 0.65 = balanced natural variation"
                )

                similarity_boost = st.slider(
                    "Similarity Boost",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.85,
                    step=0.05,
                    help="How closely to match the cloned voice. 0.85 = tight match for PVC"
                )

                style = st.slider(
                    "Style / Expressiveness",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.4,
                    step=0.05,
                    help="Emotional expression & naturalness. 0.4 = moderate expressiveness (recommended for PVC)"
                )
                
                use_speaker_boost = st.checkbox(
                    "Speaker Boost",
                    value=True,
                    help="Boost speaker clarity"
                )

            # Audio Processing Information
            st.info("**Audio Processing Features:**\n"
                   "• **Natural Timing**: Uses start time only to prevent speech cutting\n"
                   "• **Original Gaps**: Preserves exact pauses from your Swedish audio\n"
                   "• **Volume Normalization**: Ensures consistent audio levels across all segments\n"
                   "• **Smart Stitching**: Maintains natural rhythm and flow")
            
            # Store settings in session state
            st.session_state.voice_settings = {
                "voice_id": voice_id if voice_id and voice_id != "Select a voice" else eleven_voice,
                "speaking_rate": speaking_rate,
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": use_speaker_boost,
                "voice_model": voice_model
            }
            
            # Settings Preview
            with st.expander("Current Voice Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Voice ID:** {voice_id if voice_id and voice_id != 'Select a voice' else eleven_voice}")
                    st.write(f"**Speaking Rate:** {speaking_rate:.2f}")
                    st.write(f"**Stability:** {stability:.2f}")
                with col2:
                    st.write(f"**Similarity:** {similarity_boost:.2f}")
                    st.write(f"**Style:** {style:.2f}")
                    st.write(f"**Model:** {voice_model}")

    # File Upload
    st.header("Upload Audio File")
    
    # Model-specific information
    if 'voice_settings' in st.session_state and st.session_state.voice_settings.get('voice_model') == 'eleven_v3':
        st.info("**Note**: Using eleven_v3 model - each text segment will be limited to 3,000 characters (~3 minutes of speech)")
    
    uploaded_file = st.file_uploader(
        "Choose a Swedish audio file (WAV/MP3, ≤20 minutes)",
        type=["wav", "mp3"],
        help="Supported formats: WAV, MP3. Maximum duration: 20 minutes."
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        upload_path = st.session_state.working_dir / f"uploaded_{uploaded_file.name}"
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File uploaded: {uploaded_file.name}")
        st.audio(uploaded_file)
        
        # Transcription
        st.header("Transcription")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("Transcribe", type="primary"):
                try:
                    # Use status container for better UX
                    with st.status("Transcribing audio...", expanded=True) as status:
                        st.write("Loading Whisper model...")

                        # Get selected model from session state
                        model_name = st.session_state.get('whisper_model', 'large-v3-turbo')

                        # Transcribe with selected model
                        segments = transcribe_file(
                            str(upload_path),
                            st.session_state.working_dir,
                            model_name=model_name
                        )

                        st.write(f"✓ Transcription complete! Found {len(segments)} segments")
                        status.update(label="Transcription complete!", state="complete")

                    st.session_state.segments = segments
                    st.session_state.translated_segments = []  # Reset translations

                    if segments:
                        st.success(f"Found {len(segments)} segments")

                        # Show transcription preview
                        with st.expander("Transcription Preview", expanded=True):
                            for i, seg in enumerate(segments[:5]):  # Show first 5 segments
                                st.write(f"**{i+1}.** [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
                            if len(segments) > 5:
                                st.write(f"... and {len(segments) - 5} more segments")
                    else:
                        st.warning("No speech segments detected. Try adjusting audio volume or check for background noise.")

                except Exception as e:
                        st.error(f"Transcription failed: {str(e)}")
                        # Show detailed error information
                        with st.expander("Debug Information"):
                            st.write("**Error details:**", str(e))
                            st.write("**Audio file path:**", str(upload_path))
                            st.write("**File exists:**", upload_path.exists())
                            if upload_path.exists():
                                st.write("**File size:**", f"{upload_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        with col2:
            if st.session_state.segments:
                st.write(f"**{len(st.session_state.segments)} segments transcribed**")
                
                # Add transcription statistics
                if st.session_state.segments:
                    total_duration = sum(seg["end"] - seg["start"] for seg in st.session_state.segments)
                    st.write(f"**Total speech duration:** {total_duration:.1f}s")
                    
                    # Check for potential issues
                    short_segments = [seg for seg in st.session_state.segments if (seg["end"] - seg["start"]) < 1.0]
                    if short_segments:
                        st.info(f"{len(short_segments)} segments are very short (<1s)")
            else:
                st.write("No transcription yet")
            
            # Segments Display and Editing
            if st.session_state.segments:
                st.header("Transcript Editor")
                
                # Manual segment addition
                with st.expander("Add Missing Segment"):
                    st.write("If you hear speech that wasn't transcribed, you can manually add it:")
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        start_time = st.number_input(
                            "Start time (seconds)", 
                            min_value=0.0, 
                            max_value=3600.0, 
                            step=0.1,
                            format="%.1f"
                        )
                    with col2:
                        end_time = st.number_input(
                            "End time (seconds)", 
                            min_value=start_time, 
                            max_value=3600.0, 
                            step=0.1,
                            format="%.1f"
                        )
                    with col3:
                        if st.button("Add Segment"):
                            if end_time > start_time:
                                new_segment = {
                                    "start": start_time,
                                    "end": end_time,
                                    "text": "[Manual segment - add Swedish text below]"
                                }
                                
                                # Insert in correct time order
                                inserted = False
                                for i, seg in enumerate(st.session_state.segments):
                                    if start_time < seg["start"]:
                                        st.session_state.segments.insert(i, new_segment)
                                        inserted = True
                                        break
                                
                                if not inserted:
                                    st.session_state.segments.append(new_segment)
                                
                                st.success("Segment added!")
                                st.rerun()
                            else:
                                st.error("End time must be after start time")
                
                # Create editable dataframe
                df = create_segments_dataframe(st.session_state.segments)
                
                # Add segment management controls
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write(f"**Total segments:** {len(df)}")
                with col2:
                    if st.button("Reset All Segments"):
                        if st.session_state.get('confirm_reset', False):
                            st.session_state.segments = []
                            st.session_state.translated_segments = []
                            st.session_state.confirm_reset = False
                            st.success("All segments cleared!")
                            st.rerun()
                        else:
                            st.session_state.confirm_reset = True
                            st.warning("Click again to confirm reset")
                
                edited_df = st.data_editor(
                    df,
                    column_config={
                        "Start": st.column_config.TextColumn("Start", disabled=True),
                        "End": st.column_config.TextColumn("End", disabled=True),
                        "Swedish": st.column_config.TextColumn("Swedish", width="large"),
                        "English": st.column_config.TextColumn("English", width="large")
                    },
                    hide_index=True,
                    use_container_width=True,
                    num_rows="dynamic"  # Allow adding/removing rows
                )

                # Timing info
                st.info("**Note:** Timing shows original Swedish audio. Natural pauses (min 150ms) are automatically added between segments during voice generation.")

                # Translation
                st.header("Translation")
                col1, col2 = st.columns([1, 3])
                        
                with col1:
                    if st.button("Translate All", type="primary", disabled=not translation_service):
                        with st.spinner(f"Translating with {translation_service.upper()}..."):
                            try:
                                # Update segments from edited dataframe first (to preserve deletions/edits)
                                updated_segments = []
                                for i, (_, row) in enumerate(edited_df.iterrows()):
                                    if i < len(st.session_state.segments):
                                        seg = st.session_state.segments[i].copy()
                                        seg["text"] = row["Swedish"]
                                        updated_segments.append(seg)

                                # Update session state with edited segments
                                st.session_state.segments = updated_segments

                                # Now translate the updated segments
                                translated = translate_segments(
                                    updated_segments,
                                    service=translation_service,
                                    working_dir=st.session_state.working_dir
                                )
                                st.session_state.translated_segments = translated
                                st.success(f"Translated {len(translated)} segments")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Translation failed: {str(e)}")
                
                with col2:
                    if st.session_state.translated_segments:
                        st.write(f"**{len(st.session_state.translated_segments)} segments translated**")
                
                # TTS Generation
                st.header("Text-to-Speech")
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    tts_ready = (eleven_key and eleven_voice and 
                               len(edited_df) > 0 and 
                               edited_df["English"].notna().any())
                    
                    if st.button("Generate Voice", type="primary", disabled=not tts_ready):
                        with st.spinner("Generating TTS with natural timing and audio normalization..."):
                            try:
                                # Use edited text from dataframe (preserves deletions/edits)
                                updated_segments = []
                                for i, (_, row) in enumerate(edited_df.iterrows()):
                                    if i < len(st.session_state.segments):
                                        seg = st.session_state.segments[i].copy()
                                        seg["text"] = row["Swedish"]
                                        seg["english"] = row["English"] or ""
                                        updated_segments.append(seg)

                                # Update session state with edited segments
                                st.session_state.segments = updated_segments
                                st.session_state.translated_segments = updated_segments

                                # Generate TTS for each segment with custom settings
                                voice_settings = getattr(st.session_state, 'voice_settings', {})
                                audio_files = generate_tts(
                                    updated_segments,
                                    working_dir=st.session_state.working_dir,
                                    voice_settings=voice_settings
                                )
                                
                                # Stitch segments with proper timing
                                final_path = stitch_segments(
                                    updated_segments,
                                    audio_files,
                                    st.session_state.working_dir
                                )
                                
                                st.session_state.final_audio_path = final_path
                                st.success("Audio generation complete!")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"TTS generation failed: {str(e)}")
                
                with col2:
                    if not tts_ready and not (eleven_key and eleven_voice):
                        st.warning("ElevenLabs API configuration required")
                    elif not tts_ready:
                        st.info("Add English translations to generate voice")
                
                # Final Audio Player
                if st.session_state.final_audio_path and Path(st.session_state.final_audio_path).exists():
                    st.header("Final Audio")
                    
                    with open(st.session_state.final_audio_path, "rb") as f:
                        audio_bytes = f.read()
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.audio(audio_bytes, format="audio/wav")
                    
                    with col2:
                        st.download_button(
                            label="Download Audio",
                            data=audio_bytes,
                            file_name="translated_audio.wav",
                            mime="audio/wav"
                        )

    # Troubleshooting Section
    with st.expander("Troubleshooting"):
        st.subheader("Common Issues & Solutions")
        
        st.write("**Missing Speech Segments:**")
        st.write("- Audio may be too quiet - try normalizing volume before upload")
        st.write("- Background noise may interfere - use audio with clear speech")
        st.write("- Very short utterances (<1s) may be filtered out")
        st.write("- Use the 'Add Missing Segment' feature above to manually add missed parts")
        
        st.write("**Text Not Showing in UI:**")
        st.write("- Check the 'Transcription Preview' section for detected segments")
        st.write("- If preview shows segments but editor is empty, try refreshing the page")
        st.write("- Large files may take longer to process - wait for the progress bar to complete")
        
        st.write("**Audio Quality Tips:**")
        st.write("- Use WAV format for best results (MP3 is also supported)")
        st.write("- Keep background noise to a minimum")
        st.write("- Ensure clear pronunciation and moderate speaking pace")
        st.write("- File size limit: ~200MB, Duration limit: 20 minutes")
        
        st.write("**Translation Issues:**")
        st.write("- Make sure API keys are properly configured in .env file")
        st.write("- DeepL has monthly usage limits on free tier")
        st.write("- OpenAI translation may be more expensive but handles context better")
        
        st.write("**TTS Generation Problems:**")
        st.write("- ElevenLabs has character limits per API call (~400 words)")
        st.write("- Very long segments are automatically split")
        st.write("- Adjust speaking rate if speech sounds too fast/slow")
        st.write("- Voice ID must be valid - check ElevenLabs dashboard for correct IDs")
        st.write("- Popular voices are pre-loaded, or use custom voice ID for your own voices")
        st.write("- Audio timing uses start time only to prevent speech cutting")
        st.write("- Original gaps and pauses are preserved from your Swedish audio")


if __name__ == "__main__":
    main() 
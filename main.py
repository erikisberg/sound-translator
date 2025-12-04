"""
Swedish Audio Translator - Streamlit App
Transcribe Swedish audio → Translate to English → Generate English TTS

Version: 1.1.0 - Voice consistency improvements
"""

import streamlit as st
import os
import tempfile
import atexit
import shutil
import logging
import uuid
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
from session_manager import (
    Session,
    SessionManager,
    get_session_manager,
    export_session_json,
    import_session_json,
    generate_session_name
)
from database import is_db_available

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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
    # Session management
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "session_name" not in st.session_state:
        st.session_state.session_name = ""
    if "last_saved_at" not in st.session_state:
        st.session_state.last_saved_at = None
    if "source_filename" not in st.session_state:
        st.session_state.source_filename = None
    if "working_dir" not in st.session_state:
        st.session_state.working_dir = Path(tempfile.mkdtemp(prefix="audio_translator_"))
        st.session_state.working_dir.mkdir(exist_ok=True)

        # Register cleanup handler to remove temp directory on exit
        def cleanup_working_dir():
            """Remove temporary working directory and all its contents."""
            try:
                working_dir = st.session_state.working_dir
                if working_dir.exists():
                    shutil.rmtree(working_dir, ignore_errors=True)
                    logger.info(f"Cleaned up temp directory: {working_dir}")
            except Exception as e:
                logger.error(f"Failed to cleanup temp directory: {e}")

        atexit.register(cleanup_working_dir)

def save_current_session(status: str = "in_progress") -> bool:
    """Save the current session to database."""
    if not is_db_available():
        logger.warning("save_current_session: DB not available")
        return False

    manager = get_session_manager()

    # Create or update session - generate new ID if none exists
    session_id = st.session_state.current_session_id or str(uuid.uuid4())

    # Get session name - prefer existing name, then session_state, then generate
    session_name = st.session_state.session_name
    if not session_name:
        session_name = generate_session_name(st.session_state.source_filename)

    logger.info(f"save_current_session: id={session_id[:8]}..., name='{session_name}', "
                f"session_state.session_name='{st.session_state.session_name}', "
                f"segments={len(st.session_state.translated_segments or st.session_state.segments)}")

    session = Session(
        id=session_id,
        name=session_name,
        status=status,
        source_filename=st.session_state.source_filename,
        settings={
            "use_openai_api": st.session_state.get("use_openai_api", False),
            "whisper_model": st.session_state.get("whisper_model"),
            "voice_settings": st.session_state.get("voice_settings", {}),
        },
        segments=[
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text", ""),
                "english": seg.get("english", ""),
            }
            for seg in st.session_state.translated_segments or st.session_state.segments
        ],
    )

    # If no session ID yet, use the generated one
    if not st.session_state.current_session_id:
        st.session_state.current_session_id = session.id
        logger.info(f"save_current_session: New session created with id={session.id[:8]}...")

    if manager.save_session(session):
        st.session_state.session_name = session.name
        st.session_state.last_saved_at = session.updated_at
        return True
    return False


def load_session_into_state(session: Session) -> None:
    """Load a session's data into session state."""
    logger.info(f"load_session_into_state: Loading session id={session.id[:8]}..., name='{session.name}', "
                f"segments={len(session.segments)}")

    st.session_state.current_session_id = session.id
    st.session_state.session_name = session.name
    # Note: session_name_input widget state is set in init check before widget renders
    st.session_state.source_filename = session.source_filename
    st.session_state.last_saved_at = session.updated_at

    # Load segments
    st.session_state.segments = session.segments
    st.session_state.translated_segments = [
        seg for seg in session.segments if seg.get("english")
    ]

    logger.info(f"load_session_into_state: Loaded {len(st.session_state.segments)} segments, "
                f"session_name now='{st.session_state.session_name}'")

    # Load settings
    if session.settings:
        if "use_openai_api" in session.settings:
            st.session_state.use_openai_api = session.settings["use_openai_api"]
        if "whisper_model" in session.settings:
            st.session_state.whisper_model = session.settings["whisper_model"]
        if "voice_settings" in session.settings:
            st.session_state.voice_settings = session.settings["voice_settings"]


def main():
    """Main Streamlit application."""
    init_session_state()
    
    st.title("Swedish Audio Translator")
    st.markdown("Upload Swedish audio → Transcribe → Translate → Generate English TTS")
    
    # Sidebar
    with st.sidebar:
        # Session Management Section
        st.header("Session")
        db_available = is_db_available()

        if db_available:
            # Auto-generate session name (no manual input)
            if not st.session_state.session_name:
                st.session_state.session_name = generate_session_name(st.session_state.source_filename)

            # Display session info
            st.write(f"**{st.session_state.session_name}**")
            if st.session_state.last_saved_at:
                st.caption(f"Saved: {st.session_state.last_saved_at[:19]}")
            elif st.session_state.current_session_id:
                st.caption("Session active")
            else:
                st.caption("New session")

            # Session action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save", use_container_width=True):
                    if save_current_session():
                        st.success("Saved!")
                    else:
                        st.error("Save failed")

            with col2:
                if st.button("New", use_container_width=True):
                    # Reset session state
                    st.session_state.current_session_id = None
                    st.session_state.session_name = ""
                    st.session_state.last_saved_at = None
                    st.session_state.source_filename = None
                    st.session_state.segments = []
                    st.session_state.translated_segments = []
                    st.session_state.final_audio_path = None
                    st.rerun()

            # Session History
            with st.expander("Session History"):
                manager = get_session_manager()
                sessions = manager.list_sessions(limit=20)

                if sessions:
                    for sess in sessions:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            # Format display
                            status_icon = {"draft": "", "in_progress": "", "completed": ""}.get(sess["status"], "")
                            segment_info = f"{sess['segment_count']} segments" if sess["segment_count"] else "empty"
                            date_str = sess["updated_at"][:10] if sess["updated_at"] else ""

                            # Extract voice settings for display
                            settings_data = sess.get("settings") or {}
                            vs = settings_data.get("voice_settings") or {}
                            if vs:
                                sr = vs.get("speaking_rate", "-")
                                stab = vs.get("stability", "-")
                                sim = vs.get("similarity_boost", "-")
                                sty = vs.get("style", "-")
                                settings_str = f"SR:{sr} S:{stab} SB:{sim} ST:{sty}"
                            else:
                                settings_str = None

                            st.write(f"**{status_icon} {sess['name']}**")
                            if settings_str:
                                st.caption(f"{date_str} | {segment_info} | {settings_str}")
                            else:
                                st.caption(f"{date_str} | {segment_info}")

                        with col2:
                            if st.button("Load", key=f"load_{sess['id']}", use_container_width=True):
                                loaded = manager.load_session(sess["id"])
                                if loaded:
                                    load_session_into_state(loaded)
                                    st.success(f"Loaded: {loaded.name}")
                                    st.rerun()

                    # Delete session option
                    st.divider()
                    delete_id = st.selectbox(
                        "Delete session:",
                        options=[""] + [s["id"] for s in sessions],
                        format_func=lambda x: next((s["name"] for s in sessions if s["id"] == x), "Select...") if x else "Select...",
                        key="delete_session_select"
                    )
                    if delete_id and st.button("Delete Selected", type="secondary"):
                        if manager.delete_session(delete_id):
                            st.success("Deleted!")
                            st.rerun()
                else:
                    st.info("No saved sessions yet")

            # Export/Import
            with st.expander("Export / Import"):
                # Export current session
                if st.session_state.segments:
                    session_for_export = Session(
                        id=st.session_state.current_session_id or "export",
                        name=st.session_state.session_name or "Exported Session",
                        source_filename=st.session_state.source_filename,
                        segments=[
                            {
                                "start": seg.get("start"),
                                "end": seg.get("end"),
                                "text": seg.get("text", ""),
                                "english": seg.get("english", ""),
                            }
                            for seg in st.session_state.translated_segments or st.session_state.segments
                        ],
                    )
                    export_json = export_session_json(session_for_export)
                    st.download_button(
                        "Export Session (JSON)",
                        data=export_json,
                        file_name=f"{st.session_state.session_name or 'session'}.json",
                        mime="application/json",
                        use_container_width=True
                    )

                # Import session
                uploaded_session = st.file_uploader(
                    "Import Session",
                    type=["json"],
                    help="Upload a previously exported session file",
                    key="import_session_file"
                )
                if uploaded_session:
                    try:
                        json_content = uploaded_session.read().decode("utf-8")
                        imported = import_session_json(json_content)
                        if imported:
                            load_session_into_state(imported)
                            # Save imported session to DB
                            if save_current_session():
                                st.success(f"Imported: {imported.name}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Import failed: {e}")

        else:
            st.warning("Database not configured. Sessions will not be saved.")
            st.caption("Add DATABASE_URL to secrets.toml to enable session storage.")

            # Still allow export/import without DB
            with st.expander("Export / Import"):
                if st.session_state.segments:
                    session_for_export = Session(
                        name=st.session_state.session_name or "Exported Session",
                        source_filename=st.session_state.source_filename,
                        segments=[
                            {
                                "start": seg.get("start"),
                                "end": seg.get("end"),
                                "text": seg.get("text", ""),
                                "english": seg.get("english", ""),
                            }
                            for seg in st.session_state.translated_segments or st.session_state.segments
                        ],
                    )
                    export_json = export_session_json(session_for_export)
                    st.download_button(
                        "Export Session (JSON)",
                        data=export_json,
                        file_name=f"{st.session_state.session_name or 'session'}.json",
                        mime="application/json",
                        use_container_width=True
                    )

                uploaded_session = st.file_uploader(
                    "Import Session",
                    type=["json"],
                    key="import_session_no_db"
                )
                if uploaded_session:
                    try:
                        json_content = uploaded_session.read().decode("utf-8")
                        imported = import_session_json(json_content)
                        if imported:
                            load_session_into_state(imported)
                            st.success(f"Imported: {imported.name}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Import failed: {e}")

        st.divider()

        # API Key Status
        st.header("API Configuration")
        
        # Check API keys
        openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        deepl_key = st.secrets.get("DEEPL_API_KEY") or os.getenv("DEEPL_API_KEY")
        eleven_key = st.secrets.get("ELEVEN_API_KEY") or os.getenv("ELEVEN_API_KEY")
        eleven_voice = st.secrets.get("ELEVEN_VOICE_ID") or os.getenv("ELEVEN_VOICE_ID")
        
        st.write("**Transcription Service:**")
        if openai_key:
            st.success("OpenAI API Key found (Whisper API available)")
        else:
            st.warning("OpenAI API Key not found - only local transcription available")

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

        # Transcription Settings
        st.header("Transcription Settings")

        # Transcription method selector
        transcription_method = st.radio(
            "Transcription Method",
            options=["OpenAI API (Recommended)", "Local (Free)"],
            index=0,  # Default to OpenAI API
            help="OpenAI API provides better accuracy with fewer hallucinations ($0.006/min). Local is free but may have accuracy issues."
        )

        use_openai_api = transcription_method == "OpenAI API (Recommended)"
        st.session_state.use_openai_api = use_openai_api

        # Show cost estimate or free label
        if use_openai_api:
            st.success("**Using OpenAI Whisper API**: Best quality, no hallucinations")
            st.info("**Cost**: ~$0.12 for 20-minute file ($0.006/minute)")
            st.session_state.whisper_model = None  # Not used for API
        else:
            st.warning("**Using Local Whisper**: Free but may have hallucination issues")

            # Show model selector only for local mode
            whisper_model = st.selectbox(
                "Local Whisper Model",
                options=[
                    "large-v3-turbo",  # Best quality, faster (default)
                    "large-v3",        # Best quality
                    "medium",          # Good balance
                    "small",           # OK quality, faster
                    "base",            # Basic, fast
                    "tiny"             # Fastest
                ],
                index=0,
                help="Choose transcription speed vs quality trade-off. Model is cached and reused for better performance."
            )

            # Model info
            model_info = {
                "large-v3": "Best quality, ~1.5GB (cached after first load)",
                "large-v3-turbo": "Best quality optimized, ~1.5GB, faster inference",
                "medium": "Good balance, ~770MB",
                "small": "OK quality, ~490MB",
                "base": "Basic, ~140MB",
                "tiny": "Fastest, ~75MB"
            }
            st.info(f"**{whisper_model}**: {model_info[whisper_model]}")

            # Store in session state
            st.session_state.whisper_model = whisper_model

        # ElevenLabs Voice Settings
        if eleven_key and eleven_voice:
            st.header("Voice Settings")

            # Get saved voice settings (from loaded session) or use defaults
            saved_settings = st.session_state.get("voice_settings", {})

            # Speaking Rate
            speaking_rate = st.slider(
                "Speaking Rate",
                min_value=0.7,
                max_value=1.2,
                value=saved_settings.get("speaking_rate", float(st.secrets.get("ELEVEN_SPEAKING_RATE", "1.0"))),
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
                voice_options = [
                    "PqclsDjBR66GIIQ7oVAc",  # Your custom voice (default)
                    eleven_voice if eleven_voice and eleven_voice != "PqclsDjBR66GIIQ7oVAc" else None,
                    "21m00Tcm4TlvDq8ikWAM",  # Rachel (Professional female)
                    "AZnzlk1XvdvUeBnXmlld",  # Domi (Professional female)
                    "EXAVITQu4vr4xnSDxMaL",  # Bella (Professional female)
                    "VR6AewLTigWG4xSOukaG",  # Josh (Professional male)
                    "pNInz6obpgDQGcFmaJgB",  # Adam (Professional male)
                    "yoZ06aMxZJJ28mfd3POQ",  # Sam (Professional male)
                    "custom"  # Custom voice ID input
                ]
                voice_options = [v for v in voice_options if v]  # Remove None

                # Get saved voice_id or default to first option
                saved_voice_id = saved_settings.get("voice_id", voice_options[0])
                default_voice_index = voice_options.index(saved_voice_id) if saved_voice_id in voice_options else 0

                voice_id = st.selectbox(
                    "Voice ID",
                    options=voice_options,
                    index=default_voice_index,
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
                model_options = [
                    "eleven_multilingual_v2",   # ⭐ Recommended for PVC - 29 languages, 10k chars
                    "eleven_flash_v2_5",        # Ultra-fast (~75ms) - 32 languages, 40k chars
                    "eleven_turbo_v2_5",        # Balanced quality/speed - 32 languages, 40k chars
                    "eleven_v3",                # Most expressive (alpha) - 70+ languages, 3k chars
                    "eleven_turbo_v2",          # Legacy turbo (English only)
                    "eleven_flash_v2",          # Legacy flash (English only)
                ]
                saved_voice_model = saved_settings.get("voice_model", "eleven_multilingual_v2")
                default_model_index = model_options.index(saved_voice_model) if saved_voice_model in model_options else 0

                voice_model = st.selectbox(
                    "Voice Model",
                    options=model_options,
                    index=default_model_index,
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
                    value=saved_settings.get("stability", 0.65),
                    step=0.05,
                    help="Lower = more variation, Higher = more consistent. 0.65 = balanced natural variation"
                )

                similarity_boost = st.slider(
                    "Similarity Boost",
                    min_value=0.0,
                    max_value=1.0,
                    value=saved_settings.get("similarity_boost", 0.85),
                    step=0.05,
                    help="How closely to match the cloned voice. 0.85 = tight match for PVC"
                )

                style = st.slider(
                    "Style / Expressiveness",
                    min_value=0.0,
                    max_value=1.0,
                    value=saved_settings.get("style", 0.4),
                    step=0.05,
                    help="Emotional expression & naturalness. 0.4 = moderate expressiveness (recommended for PVC)"
                )

                use_speaker_boost = st.checkbox(
                    "Speaker Boost",
                    value=saved_settings.get("use_speaker_boost", True),
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
                        # Get transcription method from session state
                        use_api = st.session_state.get('use_openai_api', False)

                        if use_api:
                            st.write("Using OpenAI Whisper API for transcription...")
                        else:
                            st.write("Loading local Whisper model...")
                            model_name = st.session_state.get('whisper_model', 'large-v3-turbo')

                        # Transcribe with selected method
                        segments = transcribe_file(
                            str(upload_path),
                            st.session_state.working_dir,
                            model_name=st.session_state.get('whisper_model', 'large-v3-turbo'),
                            use_api=use_api
                        )

                        st.write(f"✓ Transcription complete! Found {len(segments)} segments")
                        status.update(label="Transcription complete!", state="complete")

                    st.session_state.segments = segments
                    st.session_state.translated_segments = []  # Reset translations
                    st.session_state.source_filename = uploaded_file.name
                    # Update session name based on new filename
                    st.session_state.session_name = generate_session_name(uploaded_file.name)

                    # Auto-save after transcription
                    if is_db_available() and segments:
                        if save_current_session("in_progress"):
                            st.toast("Session auto-saved")

                    if segments:
                        st.success(f"Found {len(segments)} segments")
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


    # Segments Display and Editing - show if we have segments (from upload OR loaded session)
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
            num_rows="dynamic"
        )

        # Save table edits button
        if st.button("Save Table Changes", help="Save your manual edits to Swedish/English text"):
            # Sync edited dataframe back to session state
            updated_segments = []
            for i, (_, row) in enumerate(edited_df.iterrows()):
                if i < len(st.session_state.segments):
                    seg = st.session_state.segments[i].copy()
                    seg["text"] = row["Swedish"]
                    seg["english"] = row["English"] if pd.notna(row["English"]) else ""
                    updated_segments.append(seg)

            st.session_state.segments = updated_segments
            st.session_state.translated_segments = [s for s in updated_segments if s.get("english")]

            # Save to database
            if is_db_available():
                if save_current_session():
                    st.success("Changes saved!")
                else:
                    st.error("Failed to save")
            else:
                st.success("Changes saved to session!")
            st.rerun()

        # Timing info
        st.info("**Note:** Timing shows original Swedish audio. Natural pauses (min 150ms) are automatically added between segments during voice generation.")

        # Translation
        st.header("Translation")
        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("Translate All", type="primary", disabled=not translation_service):
                with st.spinner(f"Translating with {translation_service.upper()}..."):
                    try:
                        # Translate segments from session state
                        # (user should click "Save Table Changes" first if they made edits)
                        translated = translate_segments(
                            st.session_state.segments,
                            service=translation_service,
                            working_dir=st.session_state.working_dir
                        )
                        st.session_state.translated_segments = translated
                        st.success(f"Translated {len(translated)} segments")

                        # Auto-save after translation
                        if is_db_available():
                            if save_current_session("in_progress"):
                                st.toast("Session auto-saved")

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
            # Check if we have translated segments with English text
            has_translations = (
                st.session_state.translated_segments and
                any(seg.get("english") for seg in st.session_state.translated_segments)
            )
            tts_ready = eleven_key and eleven_voice and has_translations

            if st.button("Generate Voice", type="primary", disabled=not tts_ready):
                with st.spinner("Generating TTS with natural timing and audio normalization..."):
                    try:
                        # Use translated segments from session state
                        # (user should click "Save Table Changes" first if they made edits)
                        segments_to_use = st.session_state.translated_segments

                        # Generate TTS for each segment with custom settings
                        voice_settings = getattr(st.session_state, 'voice_settings', {})
                        audio_files = generate_tts(
                            segments_to_use,
                            working_dir=st.session_state.working_dir,
                            voice_settings=voice_settings
                        )

                        # Stitch segments with proper timing
                        final_path = stitch_segments(
                            segments_to_use,
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
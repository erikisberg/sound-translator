# Swedish Audio Translator

A Streamlit app that transcribes Swedish audio, translates to English, and generates English TTS with proper timing.

## 60-Second Setup

### 1. Prerequisites
```bash
# Install FFmpeg (required for audio processing)
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html
```

### 2. Python Environment
```bash
# Clone/download this project
cd sound-translator-poc

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. API Keys Setup
Create a `.env` file in the project root:
```bash
# Translation (choose one)
OPENAI_API_KEY=sk-your-openai-key-here
# OR
DEEPL_API_KEY=your-deepl-key-here

# Text-to-Speech (required)
ELEVEN_API_KEY=your-elevenlabs-key-here
ELEVEN_VOICE_ID=your-voice-id-here
ELEVEN_SPEAKING_RATE=0.8  # Optional: 0.25=very slow, 1.0=normal, 4.0=very fast

# Audio Enhancement (optional)
DOLBY_API_KEY=your-dolby-api-key-here  # For professional audio enhancement
```

Alternatively, copy `.streamlit/secrets.toml` and fill in your keys.

### 4. Run the App

**macOS users (recommended):**
```bash
./run_app.sh
```

**Other platforms:**
```bash
streamlit run main.py
```

The app will open at `http://localhost:8501`

> **Note for macOS**: The `run_app.sh` script sets environment variables to prevent PyTorch hanging issues on macOS systems.

## Usage

1. **Upload** a Swedish audio file (WAV/MP3, ≤20 min)
2. **Transcribe** with Whisper (auto-detects GPU/CPU)
3. **Translate** segments to English (editable in the grid)
4. **Generate Voice** using ElevenLabs TTS
5. **Download** the final translated audio

## API Requirements

- **Translation**: OpenAI API (GPT-4o) OR DeepL API
- **TTS**: ElevenLabs API + Voice ID
- **Transcription**: Local Whisper (no API needed)
- **Audio Enhancement** (Optional): Dolby.io Media API

## 🆕 New Features

### Voice Speed Control
Control the speaking pace of generated audio:
- `ELEVEN_SPEAKING_RATE=0.8` (default - slightly slower)
- `0.25` = Very slow, deliberate speech
- `1.0` = Normal speaking pace  
- `4.0` = Very fast speech

### Automatic Audio Enhancement
Enable professional audio enhancement with Dolby.io:
- **Noise reduction**: Removes background noise
- **Dynamic range control**: Balances audio levels  
- **Speech clarity**: Enhances voice quality
- **Auto-enabled** when `DOLBY_API_KEY` is set

## File Structure

```
sound-translator-poc/
├── main.py                  # Streamlit web interface
├── audio_utils.py           # Core audio processing functions
├── run_app.sh              # macOS startup script (sets env variables)
├── requirements.txt         # Python dependencies
├── .env                     # API keys (create this file)
├── .streamlit/
│   └── secrets.toml        # Alternative API key storage
└── venv/                    # Virtual environment (auto-generated)
```

**Key Files:**
- `main.py` - Streamlit UI with audio upload, editing, and voice settings
- `audio_utils.py` - Transcription, translation, TTS, and audio stitching
- `run_app.sh` - macOS launcher with PyTorch compatibility fixes
- Working files are saved to temp directory (path shown in sidebar)

## GPU Support

The app automatically uses GPU for Whisper if available, with CPU fallback. For better performance, install PyTorch with CUDA support.

## Troubleshooting

### macOS: App Hangs on Startup

**Symptom**: App gets stuck loading or shows `ModuleNotFoundError: No module named 'torch'`

**Solutions**:

1. **Use the startup script** (recommended):
   ```bash
   ./run_app.sh
   ```

2. **Use Python 3.11** (not 3.12):
   ```bash
   # Install Python 3.11
   brew install python@3.11

   # Recreate virtual environment
   rm -rf venv
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ./run_app.sh
   ```

3. **Manual environment variables**:
   ```bash
   source venv/bin/activate
   export PYTORCH_JIT=0
   export KMP_DUPLICATE_LIB_OK=TRUE
   export OMP_NUM_THREADS=1
   streamlit run main.py
   ```

### General Issues

- **No audio playback**: Ensure FFmpeg is installed and on PATH
- **GPU not detected**: Install CUDA-compatible PyTorch (NVIDIA GPUs only)
- **API errors**: Check your API keys and quotas
- **Large files**: Keep audio under 20 minutes for best results
- **Import errors**: Ensure virtual environment is activated: `source venv/bin/activate` 
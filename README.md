# Whisper Audio Transcription

This tool transcribes audio files using OpenAI's Whisper API, breaking long audio into manageable chunks and providing both full transcription and timestamped segments.

## Prerequisites

- Python 3.9.10 (required for pydub compatibility)
- uv package manager
- OpenAI API key
- Audio file in supported format (mp3, wav, etc.)

## Installation

1. Clone the repository and cd into the root directory:
2. Install uv (if not already installed):
```bash
pip install uv
```
3. Create and activate virtual environment:
```bash
uv venv --python=python3.9.10
source .venv/bin/activate
```

4. Install required packages with uv:
```bash
uv sync
```


## Usage

1. Create `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_key_here
```

2. Modify audio file path in `whisper.py`:
```python
audio_file_path = "/path/to/audio/file.mp3"
```

3. Run the script:
```bash
python whisper.py
```

The script will:
- Split audio into 10-minute chunks
- Transcribe each chunk
- Save full transcription as `{audio_name}_transcription.txt`
- Save timestamped segments as `{audio_name}_segments.txt`

## Important Notes

- Python version 3.9.10 is required for pydub compatibility
- Large audio files will be processed in chunks
- Make sure your audio file is in a supported format (mp3, mp4, mpeg, mpga, m4a, wav, and webm.)
- See OpenAI's [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) for more information

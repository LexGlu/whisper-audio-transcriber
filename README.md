# Overview

This repository provides an asynchronous audio transcription utility using OpenAI’s Whisper model. The script reads audio files, splits them into 10-minute chunks (to keep each chunk under ~25 MB), and concurrently transcribes each chunk. The final transcription is then compiled and written to disk, along with an optional segments file containing time-stamped text segments.

### Features
Uses asyncio to handle file I/O and concurrent chunk transcriptions for efficiency.

### Configurable language
You can optionally pass a two-letter language code (ISO 3166-1 alpha-2) to the script (e.g., en, fr, es). The default is English (en).

### Chunk-based processing
The script splits the audio into 10-minute chunks (configurable via CHUNK_DURATION_MS), making it easier to handle large audio files and keep them under the size limits for the Whisper API.

### Transcribed segments with timestamps
The script can produce a separate segments file (_segments.txt), displaying each recognized text segment with start and end timestamps.


## Prerequisites

- Python 3.9.10 (required for pydub compatibility)
- uv package manager
- OpenAI API key
- Audio file in supported format (mp3, wav, etc.)

## Installation

1. Clone the repository and cd into the root directory:
2. Install uv (if not already installed). Read more here: [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
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

2. Run the script (with optional language code):
```bash
python whisper.py /path/to/audio.mp3 [language_code]
```


## Important notes

- Python version 3.9.10 is required for pydub compatibility
- Large audio files will be processed in chunks
- Make sure your audio file is in a supported format (mp3, mp4, mpeg, mpga, m4a, wav, and webm.)
- See OpenAI's [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) for more information

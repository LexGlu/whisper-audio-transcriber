import asyncio
import os
import sys

import aiofiles

from datetime import datetime
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.audio.transcription_segment import TranscriptionSegment
from openai.types.audio.transcription_verbose import TranscriptionVerbose
from pydub import AudioSegment


# 10 minutes in milliseconds (should ensure resulting file <25 MB, adjust as needed)
CHUNK_DURATION_MS = 10 * 60 * 1000


def parse_arguments() -> Tuple[str, str]:
    """
    Parse command-line arguments.
    Returns a tuple of (audio_file_path, language_code).
    """
    if len(sys.argv) < 2:
        print("Usage: python whisper.py <audio_file_path> [optional_language_code]")
        sys.exit(1)

    audio_file_path = sys.argv[1]
    language_code = "en"  # default language

    if len(sys.argv) >= 3:
        # Language code should not be empty and should be an ISO 3166-1 alpha-2 code
        user_lang = sys.argv[2].strip().lower()
        if user_lang and len(user_lang) == 2:
            language_code = user_lang
        else:
            raise ValueError(f"Invalid language code '{user_lang}'. Should be an ISO 3166-1 alpha-2 code.")

    return audio_file_path, language_code


async def load_environment_variables() -> None:
    """Loads environment variables from a .env file."""
    await asyncio.to_thread(load_dotenv)


async def get_openai_client() -> OpenAI:
    """Initializes and returns the OpenAI client."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    return OpenAI(api_key=openai_api_key)


async def read_audio_file(audio_file_path: str) -> AudioSegment:
    """
    Reads an audio file using pydub in a thread
    to avoid blocking the event loop.
    """
    return await asyncio.to_thread(AudioSegment.from_file, audio_file_path)


def split_audio_into_chunks(audio: AudioSegment) -> List[AudioSegment]:
    """
    Splits the AudioSegment into multiple smaller chunks
    based on the global CHUNK_DURATION_MS setting.
    """
    return audio[::CHUNK_DURATION_MS]


async def export_chunk(
    audio_chunk: AudioSegment, output_path: str, audio_format: str
) -> None:
    """
    Exports a single audio chunk to a file in the specified format
    (done in a thread to avoid blocking the event loop).
    """
    await asyncio.to_thread(audio_chunk.export, output_path, format=audio_format)


async def transcribe_chunk(
    client: OpenAI, chunk_file_path: str, language: str
) -> TranscriptionVerbose:
    """
    Calls OpenAI's Whisper API to transcribe a single audio chunk.
    Performed in a thread because the OpenAI client is synchronous.
    Ensures the file is opened with the correct extension so
    that the format is recognized.
    """

    def _sync_transcribe() -> dict:
        # Opening the file synchronously
        with open(chunk_file_path, "rb") as file:
            return client.audio.transcriptions.create(
                file=file,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                language=language,
            )

    return await asyncio.to_thread(_sync_transcribe)


async def save_file(content: str, file_path: str) -> None:
    """
    Saves content to a file asynchronously using aiofiles.
    """
    async with aiofiles.open(file_path, "w") as file:
        await file.write(content)


def format_segment_text(segment: TranscriptionSegment) -> str:
    """
    Formats a transcription segment (with timestamps) into a user-readable string.
    """
    start_time = datetime.fromtimestamp(segment.start).strftime("%H:%M:%S")
    end_time = datetime.fromtimestamp(segment.end).strftime("%H:%M:%S")
    return f"{start_time} - {end_time}:\n{segment.text}\n\n"


async def process_chunk(
    client: OpenAI,
    audio_chunk: AudioSegment,
    audio_format: str,
    chunk_index: int,
    audio_name: str,
    language: str
) -> TranscriptionVerbose:
    """
    Exports the chunk to disk, then transcribes it via the OpenAI Whisper API.
    Returns the transcription result as a TranscriptionVerbose object.
    """
    chunk_file_path = f"{audio_name}_{chunk_index}.{audio_format}"
    print(f"Processing chunk {chunk_index}. Duration: {audio_chunk.duration_seconds} seconds")

    # Export chunk
    await export_chunk(audio_chunk, chunk_file_path, audio_format)

    # Transcribe chunk
    transcription_data = await transcribe_chunk(client, chunk_file_path, language)
    return transcription_data


async def run_transcription(audio_file_path: str, language: str) -> None:
    """
    Main logic for transcribing the provided audio file:
      1. Load environment vars & create OpenAI client
      2. Read audio & split into chunks
      3. Process each chunk concurrently
      4. Save combined transcription and segments to disk
    """
    await load_environment_variables()
    client = await get_openai_client()

    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    audio_name, extension = os.path.splitext(audio_file_path)
    audio_format = extension.lstrip(".").lower()

    audio = await read_audio_file(audio_file_path)
    print(f"Full audio duration: {audio.duration_seconds} seconds")

    chunks = split_audio_into_chunks(audio)

    transcripts: List[str] = []
    segments: List[TranscriptionSegment] = []

    # Process all chunks concurrently
    tasks = [
        asyncio.create_task(
            process_chunk(client, chunk, audio_format, i + 1, audio_name, language)
        )
        for i, chunk in enumerate(chunks)
    ]
    transcriptions_data = await asyncio.gather(*tasks)

    # Combine text and segments
    for transcription in transcriptions_data:
        transcripts.append(transcription.text)
        segments.extend(transcription.segments)

    combined_text = " ".join(transcripts).strip()

    # Save final transcription
    transcription_file_path = f"{audio_name}_transcription.txt"
    if combined_text:
        await save_file(combined_text, transcription_file_path)
    else:
        print("No transcription data found.")

    # Save segment timestamps
    segments_file_path = f"{audio_name}_segments.txt"
    if segments:
        formatted_segments = "".join(format_segment_text(s) for s in segments if s)
        await save_file(formatted_segments, segments_file_path)
    else:
        print("No segments data found.")

    print("Transcription complete.")


def main() -> None:
    """
    Entry point for command-line usage.
    """
    audio_file_path, language_code = parse_arguments()
    print(f"Transcribing file '{audio_file_path}' using language '{language_code}'")
    asyncio.run(run_transcription(audio_file_path, language_code))


if __name__ == "__main__":
    main()

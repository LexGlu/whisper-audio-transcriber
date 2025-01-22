import os

from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.audio.transcription_segment import TranscriptionSegment

from pydub import AudioSegment
from typing import List


def main():
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)

    audio_file_path = "/path/to/audio/file.mp3"
    audio_name, audio_format = os.path.splitext(audio_file_path)
    audio_format = audio_format[
        1:
    ].lower()  # Remove the dot from the audio format and lower case

    audio = AudioSegment.from_file(audio_file_path)
    print(f"Audio duration: {audio.duration_seconds} seconds")

    chunk_duration = 10 * 60 * 1000  # 10 minutes
    chunks = audio[::chunk_duration]  # Split audio into 10 minutes chunks

    text = ""
    segments: List[TranscriptionSegment] = []
    for i, chunk in enumerate(chunks):
        chunk_index = i + 1
        print(f"Chunk No. {chunk_index} duration: {chunk.duration_seconds} seconds")
        chunk_file_path = f"{audio_name}_{chunk_index}.{audio_format}"
        chunk.export(chunk_file_path, format=audio_format)

        with open(chunk_file_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=f,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                language="en",
            )

            if i == 0:
                text = transcription.text
            else:
                text += " " + transcription.text

            segments.extend(transcription.segments)

    # save transcription to the same directory as the audio file
    transcription_file_path = f"{audio_name}_transcription.txt"
    try:
        if text:
            with open(transcription_file_path, "w") as f:
                f.write(text)
        else:
            print("No transcription found")

    except Exception as e:
        print(f"Error saving transcription: {e}")

    # save segments with timestamps
    try:
        with open(f"{audio_name}_segments.txt", "w") as f:
            segments_text = ""

            for segment in segments:
                if not segment:
                    continue

                start = datetime.fromtimestamp(segment.start).strftime("%H:%M:%S")
                end = datetime.fromtimestamp(segment.end).strftime("%H:%M:%S")

                segments_text += f"{start} - {end}: \n {segment.text} \n\n"

            if segments_text:
                f.write(segments_text)
    except Exception as e:
        print(f"Error saving segments: {e}")

    print("Transcription and segments saved successfully")


if __name__ == "__main__":
    main()

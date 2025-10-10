import subprocess
import os
from pydub import AudioSegment
import pyttsx3

#AudioSegment.from_mp3("XTTS_sample.mp3").export("XTTS_sample.wav", format="wav") MP3 -> WAV

def pytts_tts(text, output_path):
    engine = pyttsx3.init()
    engine.save_to_file(text, output_path)
    engine.runAndWait()

    print(f"PYTTS hang generálva: {output_path}")
    return output_path

def xtts_tts(text, voice_sample, output_path):
    cwd = os.getcwd()
    image = "ghcr.io/coqui-ai/tts:latest"

    command = [
        "E:/Docker/resources/bin/docker.exe", "run", "--rm", "-i",
        "-v", f"{cwd}/voice_files:/app",
        "-v", "tts_cache:/root/.local/share/tts",
        image,
        "--text", text,
        "--model_name", "tts_models/multilingual/multi-dataset/xtts_v2",
        "--speaker_wav", f"/app/{voice_sample}",
        "--out_path", f"/app/{output_path}",
        "--language_idx", "en"
    ]

    subprocess.run(
        command, 
        input=b"y\n",
        check=True
    )

    print(f"XTTS hang generálva: {output_path}")
    return f"voice_files/{output_path}"
import subprocess
import os
import pyttsx3
import librosa
import soundfile as sf

#MP3 -> WAV
#y, sr = librosa.load("voice_files/pytts_answer_1.mp3", sr=16000, mono=True)
#sf.write("voice_files/pytts_answer_1.wav", y, sr)

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
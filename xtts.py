import subprocess
import os
from pydub import AudioSegment

#AudioSegment.from_mp3("XTTS_sample.mp3").export("XTTS_sample.wav", format="wav") MP3 -> WAV

def xtts_tts(text, voice_sample, output_path="xtts_output.wav"):

    cwd = os.getcwd()
    image = "ghcr.io/coqui-ai/tts:latest"

    command = [
        "E:/Docker/resources/bin/docker.exe", "run", "--rm", "-i",
        "-v", f"{cwd}:/app",
        "-v", "tts_cache:/root/.local/share/tts",
        image,
        "--text", text,
        "--model_name", "tts_models/multilingual/multi-dataset/xtts_v2",
        "--speaker_wav", f"/app/{voice_sample}",
        "--out_path", f"/app/{output_path}",
        "--language_idx", "hu"
    ]

    subprocess.run(
        command, 
        input=b"y\n",
        check=True
    )
    print(f"XTTS hang generálva: {output_path}")
    return output_path

xtts_tts("Óriási péniszem van, de bocsánat ez egy jól eltervezett karaktergyilkosság, a Bibliából ismert Belzebub vád", "XTTS_sample.wav")
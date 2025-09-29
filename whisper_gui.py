import whisper
import gradio as gr
from datetime import datetime

model = whisper.load_model("tiny")

def transcribe(audio_path):
    if audio_path is None:
        return "Nincs bemenet."
    result = model.transcribe(audio_path, language="en")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./transcribeleirat_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(result['text'])
    print(f"A leirat mentve: {filename}")

    return result["text"]

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
    outputs="text",
    title="Whisper Speech-to-Text (lokális)",
    description="Válassz mikrofont vagy tölts fel hangfájlt. A Whisper modell a gépeden fut."
)

iface.launch()
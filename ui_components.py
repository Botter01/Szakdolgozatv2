import gradio as gr

audio_input = gr.Audio(
    type="filepath",
    label="Tölts fel egy hanganyagot"
)

transcript_output = gr.Textbox(
    label="Whisper leirat",
    lines=10,
    max_lines=20,
    interactive=False
)

rag_output = gr.Textbox(
    label="RAG eredmény",
    lines=10,
    max_lines=20,
    interactive=False
)

tts_output = gr.Audio(
    label="Válasz hangban",
    type="filepath"
)

interface_title = "Whisper + RAG + TTS demo"

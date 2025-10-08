import re
import pyttsx3

def strip_reasoning(text):
    return re.sub(r".*?</think>", "", text, flags=re.DOTALL).strip()

def text_to_speech(text, output_path="answer.mp3"):
    engine = pyttsx3.init()
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    return output_path
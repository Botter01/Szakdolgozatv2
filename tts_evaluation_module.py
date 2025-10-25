from jiwer import wer, cer, wer_standardize
from sentence_transformers import SentenceTransformer, util
import librosa 
from tts_module import *
from utils_module import whisper_model
from szakdoga import transcribe
import pandas as pd
from pesq import pesq

eval_dataset_tts_stt = [
    "The pharmaceutical industry has undergone significant transformations with the emergence of personalized medicine, biotechnology innovations, and artificial intelligence in drug discovery processes.",
    "Contemporary architecture and urban planning professionals are increasingly incorporating sustainable design principles such as green building certifications and rainwater harvesting technologies.",
    "Quantum computing represents a paradigm shift in computational technology with its potential to solve problems in cryptography, optimization, and artificial intelligence through quantum mechanical phenomena."
]

def stt_eval(eval_dataset_tts_stt, excel_path='evaluation_results/stt_eval.xlsx'):
    ref_voice = [
        "voice_files/bio.wav",
        "voice_files/arch.wav",
        "voice_files/quantum.wav"
    ]
    transcriptions = []
    for voice in ref_voice:
        transcriptions.append(transcribe(voice, whisper_model))

    stt_results = []
    model = SentenceTransformer('all-MiniLM-L6-v2')

    for transcription, eval_text in zip(transcriptions, eval_dataset_tts_stt):
        whisper_wer = wer(eval_text, transcription, wer_standardize, wer_standardize)
        similarity = util.cos_sim(model.encode(eval_text), model.encode(transcription))
        stt_results.append({
            'Whisper Transcription': transcription,
            'Evaluation Text': eval_text,
            "Semantic Similarity": float(similarity),
            'Transcription WER': whisper_wer
        })

    df = pd.DataFrame(stt_results)
    df.to_excel(excel_path, index=False, sheet_name='STT Results')

def wer_score_tts(eval_dataset_tts_stt, excel_path='evaluation_results/tts_eval.xlsx'):
    pytts_results = []
    xtts_results = []
    pytts_transcribe_results = []
    xtts_transcribe_results = []

    for i, text in enumerate(eval_dataset_tts_stt):
        pytts_results.append(pytts_tts(text, f"voice_files/pytts_answer_{i}.wav"))
        xtts_results.append(xtts_tts(text, "XTTS_Boti_sample.wav", f"xtts_answer_{i}.wav"))

    for pytts, xtts in zip(pytts_results, xtts_results):
        pytts_transcribe_results.append(transcribe(pytts, whisper_model))
        xtts_transcribe_results.append(transcribe(xtts, whisper_model))

    wer_results = []

    for eval, pytts, xtts in zip(eval_dataset_tts_stt, pytts_transcribe_results, xtts_transcribe_results):
        pytts_wer = wer(eval, pytts, wer_standardize, wer_standardize)
        xtts_wer = wer(eval, xtts, wer_standardize, wer_standardize)
        wer_results.append({
            'PyTTS Transcription': pytts,
            'PyTTS WER': pytts_wer,
            'XTTS Transcription': xtts,
            'XTTS WER': xtts_wer
        })

    df = pd.DataFrame(wer_results)
    df.to_excel(excel_path, index=False, sheet_name='TTS WER Results')

def cer_score_tts(eval_dataset_tts_stt, excel_path='evaluation_results/tts_eval.xlsx'):
    pytts_results = []
    xtts_results = []
    pytts_transcribe_results = []
    xtts_transcribe_results = []

    for i, text in enumerate(eval_dataset_tts_stt):
        pytts_results.append(pytts_tts(text, f"voice_files/pytts_answer_{i}.wav"))
        xtts_results.append(xtts_tts(text, "XTTS_Boti_sample.wav", f"xtts_answer_{i}.wav"))

    for pytts, xtts in zip(pytts_results, xtts_results):
        pytts_transcribe_results.append(transcribe(pytts, whisper_model))
        xtts_transcribe_results.append(transcribe(xtts, whisper_model))

    cer_results = []

    for eval, pytts, xtts in zip(eval_dataset_tts_stt, pytts_transcribe_results, xtts_transcribe_results):
        pytts_cer = cer(eval, pytts)
        xtts_cer = cer(eval, xtts)
        cer_results.append({
            'PyTTS Transcription': pytts,
            'PyTTS CER': pytts_cer,
            'XTTS Transcription': xtts,
            'XTTS CER': xtts_cer
        })

    df = pd.DataFrame(cer_results)
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, index=False, sheet_name='TTS CER Results')

def pesq_score_tts(sr=16000, excel_path='evaluation_results/tts_eval.xlsx'):
    pytts_voice = [
        "voice_files/pytts_answer_0.wav",
        "voice_files/pytts_answer_1.wav",
        "voice_files/pytts_answer_2.wav"
    ]
    xtts_voice = [
        "voice_files/xtts_answer_0.wav",
        "voice_files/xtts_answer_1.wav",
        "voice_files/xtts_answer_2.wav"
    ]
    ref_voice = [
        "voice_files/bio.wav",
        "voice_files/arch.wav",
        "voice_files/quantum.wav"
    ]

    results = []

    for ref_path, pytts_path, xtts_path in zip(ref_voice, pytts_voice, xtts_voice):
        ref, _ = librosa.load(ref_path, sr=sr, mono=True)
        pytts, _ = librosa.load(pytts_path, sr=sr, mono=True)
        xtts, _ = librosa.load(xtts_path, sr=sr, mono=True)

        pesq_pytts = pesq(sr, ref, pytts, 'wb')
        pesq_xtts = pesq(sr, ref, xtts, 'wb')

        results.append({
            "Reference file": ref_path,
            "PyTTS file": pytts_path,
            "PyTTS PESQ": pesq_pytts,
            "XTTS file": xtts_path,
            "XTTS PESQ": pesq_xtts
        })

    df = pd.DataFrame(results)
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, index=False, sheet_name='TTS PESQ Results')

#stt_eval(eval_dataset_tts_stt)
wer_score_tts(eval_dataset_tts_stt)
cer_score_tts(eval_dataset_tts_stt)
pesq_score_tts()
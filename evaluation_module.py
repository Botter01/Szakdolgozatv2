from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from jiwer import wer
import librosa 
import pystoi
from tts_module import *
from utils_module import whisper_model
from szakdoga import transcribe
import pandas as pd

eval_dataset_rag = []

sample_queries = [
    "Which CEO is widely recognized for democratizing AI education through platforms like Coursera?",
    "Who is Sam Altman?",
    "Who is Demis Hassabis and how did he gained prominence?",
    "Who is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's product ecosystem?",
    "How did Arvind Krishna transformed IBM?",
]

expected_responses = [
    "Andrew Ng is the CEO of Landing AI and is widely recognized for democratizing AI education through platforms like Coursera.",
    "Sam Altman is the CEO of OpenAI and has played a key role in advancing AI research and development. He strongly advocates for creating safe and beneficial AI technologies.",
    "Demis Hassabis is the CEO of DeepMind and is celebrated for his innovative approach to artificial intelligence. He gained prominence for developing systems like AlphaGo that can master complex games.",
    "Sundar Pichai is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's vast product ecosystem. His leadership has significantly enhanced user experiences globally.",
    "Arvind Krishna is the CEO of IBM and has transformed the company towards cloud computing and AI solutions. He focuses on delivering cutting-edge technologies to address modern business challenges.",
]

"""for query, reference in zip(sample_queries, expected_responses):
    relevant_docs = retriever.invoke(query)
    response = qa_chain.invoke({"context": format_docs(relevant_docs), "query": query})
    eval_dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
            "response": response,
            "reference": reference,
        }
    )"""

evaluation_dataset = EvaluationDataset.from_list(eval_dataset_rag)

eval_dataset_tts_stt = [
    "The pharmaceutical industry has undergone significant transformations over the past decade with the emergence of personalized medicine, biotechnology innovations, and the integration of artificial intelligence in drug discovery processes, which have accelerated the development of novel therapeutics and improved patient outcomes through tailored treatment strategies.",
    "Contemporary architecture and urban planning professionals are increasingly incorporating sustainable design principles such as green building certifications, passive solar heating systems, rainwater harvesting technologies, and permeable pavements to minimize environmental impact while simultaneously enhancing the quality of life for residents.",
    "Quantum computing represents a paradigm shift in computational technology with its potential to solve previously intractable problems in cryptography, optimization, drug simulation, and artificial intelligence through the manipulation of quantum mechanical phenomena such as superposition and entanglement, thereby enabling exponential increases."
]

pytts_results = []
xtts_results = []
pytts_transcribe_results = []
xtts_transcribe_results = []

for i, text in enumerate(eval_dataset_tts_stt):
    pytts_results.append(pytts_tts(text, f"voice_files/pytts_answer_{i}.mp3"))
    xtts_results.append(xtts_tts(text, "XTTS_Boti_sample.wav", f"xtts_answer_{i}.wav"))

for pytts, xtts in zip(pytts_results, xtts_results):
    pytts_transcribe_results.append(transcribe(pytts, whisper_model))
    xtts_transcribe_results.append(transcribe(xtts, whisper_model))

wer_results = []

for eval, pytts, xtts in zip(eval_dataset_tts_stt, pytts_transcribe_results, xtts_transcribe_results):
    pytts_wer = wer(eval, pytts)
    xtts_wer = wer(eval, xtts)
    wer_results.append({
        'PyTTS Transcription': pytts,
        'PyTTS WER': pytts_wer,
        'XTTS Transcription': xtts,
        'XTTS WER': xtts_wer
    })

df = pd.DataFrame(wer_results)
df.to_excel('evaluation_results/tts_eval.xlsx', index=False, sheet_name='WER Results')
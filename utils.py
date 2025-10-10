import whisper
import re
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter

whisper_model = whisper.load_model("small")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 32, "normalize_embeddings": True}
)

#embedding_model = OllamaEmbeddings(model="nomic-embed-text")

#fastmodel_name = "qwen3:0.6b"

model_name = "qwen3:4b"
evalmodel_name = "qwen3:8b"

local_llm = OllamaLLM(model=model_name, temperature=0.7) 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
)

def strip_reasoning(text):
    return re.sub(r".*?</think>", "", text, flags=re.DOTALL).strip()
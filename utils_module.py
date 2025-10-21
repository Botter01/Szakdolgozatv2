import whisper
import re
import inflect
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import WikipediaLoader

fastmodel_name = "qwen3:0.6b"
querymodel_name = "phi3:medium"
generationmodel_name = "qwen3:4b"
evalmodel_name = "qwen3:8b"
whisper_model = whisper.load_model("small")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 32, "normalize_embeddings": True}
)

#embedding_model = OllamaEmbeddings(model="nomic-embed-text")

local_llm = OllamaLLM(model=generationmodel_name, temperature=0.7)
query_model = OllamaLLM(model=querymodel_name)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
)

reranker = CrossEncoder("BAAI/bge-reranker-large")

def strip_reasoning(text):
    return re.sub(r".*?</think>", "", text, flags=re.DOTALL).strip()

def normalize_numbers_tts(query):
    p = inflect.engine()
    def replace_number(match):
        number = match.group()
        try:
            return p.number_to_words(int(number))
        except:
            return number 
    return re.sub(r'\b\d+\b', replace_number, query)
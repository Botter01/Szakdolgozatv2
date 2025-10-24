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
query_model = OllamaLLM(model=fastmodel_name)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
)

reranker = CrossEncoder("BAAI/bge-reranker-large")

def strip_reasoning(text):
    return re.sub(r".*?</think>", "", text, flags=re.DOTALL).strip()

def normalize_numbers_tts(query):
    engine = inflect.engine()

    def replace_number(match):
        number_str = match.group()
        clean_number = re.sub(r'[.,]', '', number_str)
        try:
            words = engine.number_to_words(int(clean_number))
            return words
        except:
            return number_str

    pattern = r'\b\d{1,3}(?:[.,]\d{3})*\b|\b\d+\b'
    return re.sub(pattern, replace_number, query)

def lol(query):
    loader = WikipediaLoader(query=query, lang="en", load_max_docs=2)
    raw_docs = loader.load()
    print(f"Vamos: {len(raw_docs)}")
    print(raw_docs)

#lol('Climate change economic impacts on food security')
#lol('Climate change effects on food access')
#lol('Climate change effects on agricultural yields')
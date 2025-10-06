import whisper
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter

whisper_model = whisper.load_model("small")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 32, "normalize_embeddings": True}
)

#embedding_model = OllamaEmbeddings(model="nomic-embed-text")

model_name = "qwen3:0.6b"

local_llm = Ollama(model=model_name) 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
)
import whisper
import time
import gradio as gr
import mlflow
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter


whisper_model = whisper.load_model("tiny")

#embedding_model = OllamaEmbeddings(model="nomic-embed-text")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 32, "normalize_embeddings": True}
)

model_name = "qwen3:0.6b"

local_llm = Ollama(model=model_name) 

def transcribe_and_rag(audio_path):
    with mlflow.start_run():
        total_start = time.time()
        whisper_start = time.time()

        result = whisper_model.transcribe(audio_path, language="en")
        whisper_time = time.time() - whisper_start
        mlflow.log_metric("whisper_time", whisper_time)
        query = result["text"]
        print(query)

        loader = WikipediaLoader(query=query, lang="en", load_max_docs=3)
        raw_docs = loader.load()
        print("Dokumentumok (nyers):", len(raw_docs))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
        )
        docs = text_splitter.split_documents(raw_docs)
        mlflow.log_param("chunk_size", 1500)
        mlflow.log_param("load_max_docs", 3)
        print("Chunkolt dokumentumok száma:", len(docs))

        embedding_start = time.time()
        vectorstore = FAISS.from_documents(docs, embedding_model)
        embedding_time = time.time() - embedding_start
        mlflow.log_metric("embedding_time", embedding_time)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        mlflow.log_param("retriever_k", 3)
        mlflow.log_param("search_type", "similarity")

        retriever_start = time.time()
        retrieved_docs = retriever.get_relevant_documents(query)
        retriever_time = time.time() - retriever_start
        mlflow.log_metric("retriever_time", retriever_time)

        llm_start = time.time()
        rag_chain = RetrievalQA.from_chain_type(
            llm=local_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=(
                        "Use the following context to answer the question. And only answer the question, don't mention what the question was.\n\n"
                        "Context:\n{context}\n\n"
                        "Question: {question}\n"
                        "Answer:"
                    ),
                    input_variables=["question", "context"], 
                )
            },
            return_source_documents=True,
        )

        response = rag_chain.invoke({"query": query})
        llm_time = time.time() - llm_start
        mlflow.log_metric("llm_time", llm_time)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("parameters", "752M")
        mlflow.log_param("quantization", "Q4_K_M")
        answer = response["result"]
        sources = response.get("source_documents", [])
        simplified_sources = [{"id": doc.id, "title": doc.metadata.get("title")} for doc in sources]

        mlflow.log_param("query", query)
        mlflow.log_param("num_docs", len(docs))
        mlflow.log_param("answer", answer)
        mlflow.log_metric("answer_length", len(answer))
        mlflow.log_param("sources", simplified_sources)

        total_time = time.time() - total_start
        mlflow.log_metric("total_time", total_time)

    return query, f"Válasz: {answer}"

ui = gr.Interface(
    fn=transcribe_and_rag,
    inputs=gr.Audio(type="filepath", label="Tölts fel egy hanganyagot"),
    outputs=[
        gr.Textbox(label="Whisper leirat", lines=10, max_lines=20, interactive=False),
        gr.Textbox(label="RAG eredmény", lines=10, max_lines=20, interactive=False)
    ],
    title="Whisper + RAG demo",
)


if __name__ == "__main__":
    ui.launch()
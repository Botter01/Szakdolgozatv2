import whisper
import gradio as gr
import mlflow
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter


whisper_model = whisper.load_model("small")

embedding_model = OllamaEmbeddings(model="nomic-embed-text")  

local_llm = Ollama(model="llama3") 


def transcribe_and_rag(audio_path):
    with mlflow.start_run():
        result = whisper_model.transcribe(audio_path, language="en")
        query = result["text"]
        print(query)

        loader = WikipediaLoader(query=query, lang="en", load_max_docs=3)
        raw_docs = loader.load()
        print("Dokumentumok (nyers):", len(raw_docs))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
        )
        docs = text_splitter.split_documents(raw_docs)
        print("Chunkolt dokumentumok száma:", len(docs))

        vectorstore = FAISS.from_documents(docs, embedding_model)
        retriever = vectorstore.as_retriever()

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
        answer = response["result"]
        sources = response.get("source_documents", [])

        mlflow.log_param("query", query)
        mlflow.log_param("num_docs", len(docs))
        mlflow.log_param("answer", answer)
        mlflow.log_metric("answer_length", len(answer))

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
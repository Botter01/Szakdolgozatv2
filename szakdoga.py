from ui_components_module import *
from utils_module import *
from tts_module import *
import time
import gradio as gr
import mlflow
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def transcribe(audio_path, whisper_model):
    start = time.time()
    result = whisper_model.transcribe(audio_path, language="en")
    duration = time.time() - start
    mlflow.log_metric("whisper_time", duration)
    query = result["text"]

    return query

def build_retriever(query, embedding_model, text_splitter):
    load_start = time.time()
    loader = WikipediaLoader(query=query, lang="en", load_max_docs=7)
    raw_docs = loader.load()
    docs = text_splitter.split_documents(raw_docs)
    mlflow.log_param("chunk_size", 1500)
    mlflow.log_param("num_chunks", len(docs))
    mlflow.log_param("load_max_docs", 7)
    mlflow.log_metric("load_time", time.time() - load_start)

    embed_start = time.time()
    vectorstore = FAISS.from_documents(docs, embedding_model)
    mlflow.log_metric("embedding_time", time.time() - embed_start)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    mlflow.log_param("retriever_k", 5)
    mlflow.log_param("search_type", "similarity")

    return retriever, docs

def generate_answer(query, retriever, llm, model_name):
    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Using ONLY the information in the context, "
            "answer the question in a full sentence."
            "Do not include your reasoning.\n\n"
            "Use ONLY the following context to answer. "
            "If the answer is not present, respond 'I don't know'.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        ),
        input_variables=["question", "context"],
    )

    start = time.time()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    response = rag_chain.invoke({"query": query})
    mlflow.log_metric("llm_time", time.time() - start)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("parameters", "752M")
    mlflow.log_param("quantization", "Q4_K_M")

    answer = strip_reasoning(response["result"])
    sources = response.get("source_documents", [])
    simplified_sources = [{"id": doc.id, "title": doc.metadata.get("title")} for doc in sources]

    mlflow.log_param("sources", simplified_sources)
    mlflow.log_param("reasoning", response["result"])
    mlflow.log_metric("answer_length", len(answer))

    return answer

def transcribe_and_rag(audio_path):
    with mlflow.start_run():
        total_start = time.time()

        query = transcribe(audio_path, whisper_model)
        yield gr.update(value=query), None, None

        retriever, docs = build_retriever(query, embedding_model, text_splitter)
        mlflow.log_metric("num_docs", len(docs))

        answer = generate_answer(query, retriever, local_llm, model_name)
        yield gr.update(value=query), gr.update(value=answer), None

        audio_path = pytts_tts(answer, "voice_files/pytts_answer.mp3")
        total_time = time.time() - total_start
        mlflow.log_metric("total_time", total_time)

        yield gr.update(value=query), gr.update(value=answer), gr.update(value=audio_path, autoplay=True)

    return f"Leirat: {query}", f"Válasz: {answer}", audio_path


with gr.Blocks() as ui:
    gr.Markdown(f"<u><h1 style='text-align: center;'>{interface_title}</h1></u>")
    audio_input.render()
    transcript_output.render()
    rag_output.render()
    tts_output.render()

    audio_input.change(
        fn=transcribe_and_rag,
        inputs=[audio_input],
        outputs=[transcript_output, rag_output, tts_output]
    )

if __name__ == "__main__":
    ui.launch()
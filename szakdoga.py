from ui_components_module import *
from utils_module import *
from tts_module import *
import time
import gradio as gr
import mlflow
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain

def transcribe(audio_path, whisper_model):
    start = time.time()
    result = whisper_model.transcribe(audio_path, language="en")
    duration = time.time() - start
    mlflow.log_metric("whisper_time", duration)
    query = result["text"]

    return query

def query_rewriting(query, llm):
    print("Query rewritingban vagyok\n")
    start = time.time()
    prompt = PromptTemplate(
        template=(
            "You are a query optimizer for a retrieval system.\n"
            "Rewrite this user query so that it is explicit, factual, and suitable for Wikipedia search.\n"
            "User query: \"{query}\"\n\nRewritten query:"
        ),
        input_variables=["query"],
    )

    formatted_prompt = prompt.format(query=query)
    rewritten_query = llm.invoke(formatted_prompt).strip()
    duration = time.time() - start

    mlflow.log_metric("rewrite_time", duration)
    mlflow.log_param("rewrite_input", query)
    mlflow.log_param("rewrite_output", rewritten_query)

    print("Query rewritingban végeztem\n")    
    return rewritten_query

def build_retriever(query, embedding_model, text_splitter):
    load_start = time.time()

    loader = WikipediaLoader(query=query, lang="en", load_max_docs=7)
    raw_docs = loader.load()

    docs = text_splitter.split_documents(raw_docs)
    original_sources = [{"id": doc.id, "title": doc.metadata.get("title")} for doc in docs]

    mlflow.log_param("original_sources", original_sources)
    mlflow.log_param("chunk_size", 2500)
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

def rerank_docs(query, docs):
    print("Rerankbanban vagyok\n")
    start = time.time()

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, _ in ranked[:5]]

    duration = time.time() - start
    avg_score = float(sum(scores) / len(scores)) if len(scores) > 0 else 0

    mlflow.log_metric("rerank_time", duration)
    mlflow.log_metric("rerank_avg_score", avg_score)
    mlflow.log_param("rerank_top_k", 5)
    print("Rerankbanban végeztem\n")

    return reranked_docs

def generate_answer(query, reranked_docs, llm, model_name):
    start = time.time()
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

    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})
    mlflow.log_metric("llm_time", time.time() - start)
    mlflow.log_param("model_name", model_name)

    answer = strip_reasoning(response)
    simplified_sources = [{"id": doc.id, "title": doc.metadata.get("title")} for doc in reranked_docs]

    mlflow.log_param("reranked_sources", simplified_sources)
    mlflow.log_param("reasoning", response)
    mlflow.log_metric("answer_length", len(answer))

    return answer

def fact_check(answer, sources, llm):
    start = time.time()
    context = "\n\n".join(doc.page_content for doc in sources[:3])
    prompt = PromptTemplate(
            template=("""
                You are a fact-checking assistant.
                Verify if the following answer is supported by the given context.

                Context:
                {context}

                Answer:
                {answer}

                Respond with:
                Verdict: "SUPPORTED" or "NOT SUPPORTED"
                Explanation: one short sentence."""
            ),
            input_variables=["context", "answer"],
        )
    
    formatted_prompt = prompt.format(context=context, answer=answer)
    verification = llm.invoke(formatted_prompt).strip()
    duration = time.time() - start

    mlflow.log_metric("fact_check_time", duration)

    return verification

def transcribe_and_rag(audio_path):
    with mlflow.start_run():
        total_start = time.time()

        query = transcribe(audio_path, whisper_model)
        yield gr.update(value=query), None, None

        rewritten_query = query_rewriting(query, query_model)
        yield gr.update(value=f"Original: {query}\n\nRewritten: {rewritten_query}"), None, None

        _, docs = build_retriever(query, embedding_model, text_splitter)
        mlflow.log_metric("num_docs", len(docs))

        reranked_docs = rerank_docs(rewritten_query, docs)
        mlflow.log_metric("num_docs_reranked", len(reranked_docs))

        answer = generate_answer(rewritten_query, reranked_docs, local_llm, model_name)
        yield gr.update(value=rewritten_query), gr.update(value=answer), None

        verification = fact_check(answer, reranked_docs, local_llm)
        mlflow.log_param("fact_check_verdict", verification)

        audio_path = pytts_tts(answer, "voice_files/pytts_rag_answer.mp3")
        total_time = time.time() - total_start
        mlflow.log_metric("total_time", total_time)

        yield gr.update(value=rewritten_query), gr.update(value=answer), gr.update(value=audio_path, autoplay=True)

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
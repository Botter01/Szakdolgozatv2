from ui_components_module import *
from utils_module import *
from tts_module import *
import time
import gradio as gr
import mlflow
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from langchain.prompts import PromptTemplate

def transcribe(audio_path, whisper_model):
    start = time.time()

    result = whisper_model.transcribe(audio_path, language="en")
    duration = time.time() - start
    mlflow.log_metric("whisper_time", duration)
    query = result["text"]

    return query

def query_rewriting(query, llm):
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

    return rewritten_query

def multi_query(query, llm, variant_number):
    prompt = PromptTemplate(
        template = (
        "Generate {n} different paraphrases of the following query.\n"
        "Each should be clear, factual, and optimized for Wikipedia search.\n\n"
        "Original query: \"{query}\"\n\n"
        "Paraphrases (one per line, and don't use numbering):"
        ),
        input_variables=["n", "query"]
    )

    formatted_prompt = prompt.format(n=variant_number, query=query)
    response = llm.invoke(formatted_prompt).strip()
    variants = [q.strip("-•1234567890. ") for q in response.split("\n") if q.strip()]

    return [query] + variants[:variant_number]

def faiss_retriever(query, embedding_model, text_splitter, top_k=4, n_docs=3):
    load_start = time.time()

    loader = WikipediaLoader(query=query, lang="en", load_max_docs=n_docs)
    wiki_docs = loader.load()

    wiki_chuncks = text_splitter.split_documents(wiki_docs)
    original_sources = [{"id": wiki_chunck.id, "title": wiki_chunck.metadata.get("title")} for wiki_chunck in wiki_chuncks]

    #mlflow.log_param("original_sources", original_sources)
    #mlflow.log_param("chunk_size", 1000)
    #mlflow.log_param("num_chunks", len(wiki_chuncks))
    #mlflow.log_param("load_max_docs", n_docs)
    #mlflow.log_metric("load_time", time.time() - load_start)

    embed_start = time.time()
    vectorstore = FAISS.from_documents(wiki_chuncks, embedding_model)
    #mlflow.log_metric("embedding_time", time.time() - embed_start)

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_chuncks = retriever.get_relevant_documents(query)
    #mlflow.log_param("retriever_k", top_k)

    return retrieved_chuncks

def bm25_retriever(query, text_splitter, top_k=4, n_docs=3):
    loader = WikipediaLoader(query=query, lang="en", load_max_docs=n_docs)
    raw_docs = loader.load()

    wiki_chuncks = text_splitter.split_documents(raw_docs)
    retriever = BM25Retriever.from_documents(wiki_chuncks)
    retriever.k = top_k
    retrieved_chuncks = retriever.get_relevant_documents(query)

    return retrieved_chuncks

def hybrid_retriever(query, embedding_model, text_splitter, top_k=4, n_docs=3):
    bm25_chuncks = bm25_retriever(query, text_splitter, top_k, n_docs)
    faiss_chuncks = faiss_retriever(query, embedding_model, text_splitter, top_k, n_docs)

    all_chuncks = list({chunck.page_content: chunck for chunck in bm25_chuncks + faiss_chuncks}.values())

    reranked_chuncks = rerank_chuncks(query, all_chuncks)
    return reranked_chuncks[:top_k]

def rerank_chuncks(query, chuncks):

    start = time.time()

    pairs = [[query, chunck.page_content] for chunck in chuncks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chuncks, scores), key=lambda x: x[1], reverse=True)
    reranked_chuncks = [chunck for chunck, _ in ranked[:4]]

    duration = time.time() - start
    avg_score = float(sum(scores) / len(scores)) if len(scores) > 0 else 0

    #mlflow.log_metric("rerank_time", duration)
    #mlflow.log_metric("rerank_avg_score", avg_score)

    return reranked_chuncks

def generate_answer(query, reranked_chuncks, llm, model_name):

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

    context = "\n\n".join([chunck.page_content for chunck in reranked_chuncks])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})
    #mlflow.log_metric("llm_time", time.time() - start)
    #mlflow.log_param("model_name", model_name)

    answer = strip_reasoning(response)
    #mlflow.log_param("reasoning", response)
    #mlflow.log_metric("answer_length", len(answer))

    return answer

def fact_check(answer, chuncks, llm):

    start = time.time()
    context = "\n\n".join(chunck.page_content for chunck in chuncks[:3])

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

def transcribe_and_rag(audio_path, multi_query_toggle: bool):
    with mlflow.start_run():
        total_start = time.time()

        query = transcribe(audio_path, whisper_model)
        yield gr.update(value=query), None, None

        if(multi_query_toggle):
            queries = multi_query(query, query_model, 3)
            all_docs = []

            for q in queries:
                retrieved_chuncks = faiss_retriever(q, embedding_model, text_splitter)
                all_docs.extend(retrieved_chuncks)

            unique_docs = {d.page_content: d for d in all_docs}.values()

            reranked_chuncks = rerank_chuncks(query, list(unique_docs))

        rewritten_query = query_rewriting(query, query_model)
        yield gr.update(value=f"Original: {query}\n\nRewritten: {rewritten_query}"), None, None

        retrieved_chuncks = faiss_retriever(query, embedding_model, text_splitter)

        reranked_chuncks = rerank_chuncks(rewritten_query, retrieved_chuncks)
        mlflow.log_metric("num_docs_reranked", len(reranked_chuncks))

        answer = generate_answer(rewritten_query, reranked_chuncks, local_llm, generationmodel_name)
        yield gr.update(value=rewritten_query), gr.update(value=answer), None

        verification = fact_check(answer, reranked_chuncks, local_llm)
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
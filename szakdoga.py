from ui_components_module import *
from utils_module import *
from tts_module import *
import time
import gradio as gr
import mlflow
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import PromptTemplate

def transcribe(audio_path, whisper_model):
    start = time.time()

    result = whisper_model.transcribe(audio_path, language="en")
    query = result["text"]

    mlflow.log_metric("whisper_time", time.time() - start)

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
    rewritten_query = strip_reasoning(rewritten_query)

    #mlflow.log_metric("rewrite_time", time.time() - start)
    #mlflow.log_param("original_query", query)
    #mlflow.log_param("rewritten_query", rewritten_query)

    return rewritten_query

def multi_aspect_query(query, llm, variant_number):
    multi_aspect_query_start = time.time()
    prompt = PromptTemplate(
        template=(
            "You are an expert research assistant creating concise search queries for Wikipedia.\n\n"
            "Your task is to take the user's question and expand it into {n} distinct, short sub-queries.\n"
            "Each sub-query should focus on a different aspect of the topic (for example: historical, social, political, economic, or cultural),\n"
            "and be phrased in a way that Wikipedia search would likely return relevant pages.\n\n"
            "Keep each sub-query under 10 words, avoid question marks or filler words.\n"
            "Example:\n"
            "Main question: 'Causes and effects of the 1956 Hungarian Revolution'\n"
            "Sub-queries:\n"
            "- 1956 Hungarian Revolution political causes\n"
            "- 1956 Hungarian Revolution social impact\n"
            "- 1956 Hungarian Revolution economic consequences\n"
            "- Soviet response to the 1956 Hungarian Revolution\n\n"
            "Now create {n} sub-queries for:\n\"{query}\"\n\n"
            "Output each sub-query on a new line, without numbering or bullet points."
        ),
        input_variables=["query", "n"]
    )
    formatted_prompt = prompt.format(query=query, n=variant_number)
    response = llm.invoke(formatted_prompt).strip()
    sub_queries = [q.lstrip("-• ").rstrip().strip() for q in response.split("\n") if q.strip()]

    #mlflow.log_param("aspect_sub_queries", sub_queries)
    #mlflow.log_metric("multi_aspect_query_time", time.time() - multi_aspect_query_start)
    print(sub_queries)

    return [query] + sub_queries[:variant_number]

def multi_paraphrase_query(query, llm, variant_number):
    multi_query_start = time.time()
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
    variants = [q.lstrip("-• ").rstrip().strip() for q in response.split("\n") if q.strip()]
    #mlflow.log_param("query_variants", variants)
    #mlflow.log_metric("multi_query_time", time.time() - multi_query_start)

    return [query] + variants[:variant_number]

def faiss_retriever(query, embedding_model, text_splitter, top_k=4, n_docs=2):
    loader = WikipediaLoader(query=query, lang="en", load_max_docs=n_docs)
    time.sleep(2)
    wiki_docs = loader.load()

    wiki_chuncks = text_splitter.split_documents(wiki_docs)
    original_sources = [{"id": wiki_chunck.id, "title": wiki_chunck.metadata.get("title")} for wiki_chunck in wiki_chuncks]

    #mlflow.log_param("original_sources_faiss", original_sources)
    #mlflow.log_param("num_chunks_faiss", len(wiki_chuncks))
    #mlflow.log_param("load_max_docs_faiss", n_docs)

    embed_start = time.time()
    vectorstore = FAISS.from_documents(wiki_chuncks, embedding_model)

    #mlflow.log_metric("embedding_time", time.time() - embed_start)

    retriev_start = time.time()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_chuncks = retriever.get_relevant_documents(query)
    retrieved_chuncks_titles = [{"id": retrieved_chunck.id, "title": retrieved_chunck.metadata.get("title")} for retrieved_chunck in retrieved_chuncks]

    #mlflow.log_param("retrieved_chuncks_faiss", retrieved_chuncks_titles)
    #mlflow.log_metric("retrieving_time_faiss", time.time() - retriev_start)
    #mlflow.log_param("retriever_k_faiss", top_k)

    return retrieved_chuncks

def bm25_retriever(query, text_splitter, top_k=4, n_docs=2):
    print(f"Query BM25-nél {query}")
    loader = WikipediaLoader(query=query, lang="en", load_max_docs=n_docs)
    time.sleep(2)
    raw_docs = loader.load()
    print(f"Vamos: {len(raw_docs)}")

    wiki_chuncks = text_splitter.split_documents(raw_docs)
    original_sources = [{"id": wiki_chunck.id, "title": wiki_chunck.metadata.get("title")} for wiki_chunck in wiki_chuncks]
    print(f"Lol: {original_sources}")

    #mlflow.log_param("original_sources_bm25", original_sources)
    #mlflow.log_param("num_chunks_bm25", len(wiki_chuncks))
    #mlflow.log_param("load_max_docs_bm25", n_docs)

    retriever = BM25Retriever.from_documents(wiki_chuncks)
    retriever.k = top_k

    retriev_start = time.time()
    retrieved_chuncks = retriever.get_relevant_documents(query)
    print(len(retrieved_chuncks))
    retrieved_chuncks_titles = [{"id": retrieved_chunck.id, "title": retrieved_chunck.metadata.get("title")} for retrieved_chunck in retrieved_chuncks]

    #mlflow.log_param("retrieved_chuncks_bm25", retrieved_chuncks_titles)
    #mlflow.log_metric("retrieving_time_bm25", time.time() - retriev_start)
    #mlflow.log_param("retriever_k_bm25", top_k)

    return retrieved_chuncks

def hybrid_retriever(query, embedding_model, text_splitter, top_k=4, n_docs=2):
    retriev_start = time.time()
    
    bm25_chuncks = bm25_retriever(query, text_splitter, top_k, n_docs)
    faiss_chuncks = faiss_retriever(query, embedding_model, text_splitter, top_k, n_docs)

    all_chuncks = list({chunck.page_content: chunck for chunck in bm25_chuncks + faiss_chuncks}.values())

    reranked_chuncks = rerank_chuncks(query, all_chuncks)

    #mlflow.log_metric("retrieving_time_hybrid", time.time() - retriev_start)

    return reranked_chuncks[:top_k]

def rerank_chuncks(query, chuncks):
    start = time.time()

    pairs = [[query, chunck.page_content] for chunck in chuncks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chuncks, scores), key=lambda x: x[1], reverse=True)
    reranked_chuncks = [chunck for chunck, _ in ranked[:4]]

    avg_score = float(sum(scores) / len(scores)) if len(scores) > 0 else 0

    #mlflow.log_metric("rerank_time", time.time() - start)
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
    #mlflow.log_param("answer_generator_model", model_name)

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
    verdict = strip_reasoning(verification)
    mlflow.log_param("fact_check_verification_result", verdict)
    mlflow.log_metric("fact_check_time", time.time() - start)

    return verdict

def transcribe_and_rag(audio_path, use_query_rewriting, multi_query_choice, retriever_choice, use_fact_check, tts_choice, chunk_rerank, query_model=query_model, generator_model=local_llm):
    with mlflow.start_run():
        total_start = time.time()

        query = transcribe(audio_path, whisper_model)
        yield gr.update(value=f"Transcribed query: {query}"), None, None

        if (use_query_rewriting):
            rewritten_query = query_rewriting(query, generator_model)
            query_to_use = rewritten_query
            yield gr.update(value=f"Original query: {query}\n\nRewritten query: {query_to_use}"), None, None
        else:
            query_to_use = query
        

        if (multi_query_choice == "paraphrase"):
            queries = multi_paraphrase_query(query_to_use, generator_model, 3)
            all_chuncks = []

            for query in queries:
                if (retriever_choice == "faiss"):
                    retrieved_chuncks = faiss_retriever(query, embedding_model, text_splitter)
                    all_chuncks.extend(retrieved_chuncks)
                elif (retriever_choice == "bm25"):
                    retrieved_chuncks = bm25_retriever(query, text_splitter)
                    all_chuncks.extend(retrieved_chuncks)
                elif (retriever_choice == "hybrid"):
                    retrieved_chuncks_hybrid = hybrid_retriever(query, embedding_model, text_splitter)
                    all_chuncks.extend(retrieved_chuncks_hybrid)

            unique_chuncks = {chunck.page_content: chunck for chunck in all_chuncks}.values()

        elif (multi_query_choice == "aspect"):
            queries = multi_aspect_query(query_to_use, generator_model, 3)
            all_chuncks = []

            for query in queries:
                if (retriever_choice == "faiss"):
                    retrieved_chuncks = faiss_retriever(query, embedding_model, text_splitter)
                    all_chuncks.extend(retrieved_chuncks)
                elif (retriever_choice == "bm25"):
                    retrieved_chuncks = bm25_retriever(query, text_splitter)
                    all_chuncks.extend(retrieved_chuncks)
                elif (retriever_choice == "hybrid"):
                    retrieved_chuncks_hybrid = hybrid_retriever(query, embedding_model, text_splitter)
                    all_chuncks.extend(retrieved_chuncks_hybrid)

            unique_chuncks = {chunck.page_content: chunck for chunck in all_chuncks}.values()

        else:
            all_chuncks = []
            if (retriever_choice == "faiss"):
                retrieved_chuncks = faiss_retriever(query, embedding_model, text_splitter)
                all_chuncks.extend(retrieved_chuncks)
            elif (retriever_choice == "bm25"):
                retrieved_chuncks = bm25_retriever(query, text_splitter)
                all_chuncks.extend(retrieved_chuncks)
            elif (retriever_choice == "hybrid"):
                retrieved_chuncks_hybrid = hybrid_retriever(query, embedding_model, text_splitter)
                all_chuncks.extend(retrieved_chuncks_hybrid)

            unique_chuncks = {chunck.page_content: chunck for chunck in all_chuncks}.values()

        mlflow.log_param("unranked_chuncks", unique_chuncks)
        if chunk_rerank:
            reranked_chuncks = rerank_chuncks(query_to_use, list(unique_chuncks))
        else:
            reranked_chuncks = list(unique_chuncks)[:4]
        mlflow.log_metric("num_docs_reranked", len(reranked_chuncks))
        mlflow.log_param("reranked_chuncks", reranked_chuncks)

        answer = generate_answer(query_to_use, reranked_chuncks, local_llm, generationmodel_name)
        yield gr.update(value=query_to_use), gr.update(value=f"Generated answer: {answer}"), None


        if use_fact_check:
            verdict = fact_check(answer, reranked_chuncks, query_model)
            yield gr.update(value=query_to_use), gr.update(value=f"Generated answer: {answer}\n\nFact-checked: {verdict}"), None


        if tts_choice == "pytts":
            norm_answer = normalize_numbers_tts(answer)
            audio_path = pytts_tts(norm_answer, "voice_files/pytts_rag_answer.mp3")
        elif tts_choice == "xtts":
            norm_answer = normalize_numbers_tts(answer)
            audio_path = xtts_tts(norm_answer, "XTTS_Boti_sample.wav", "xtts_rag_answer_boti_complex.mp3")


        mlflow.log_metric("total_time", time.time() - total_start)

        yield gr.update(value=query_to_use), gr.update(value=answer), gr.update(value=audio_path, autoplay=True)

    return query, answer, audio_path

with gr.Blocks() as ui:
    gr.Markdown(f"<u><h1 style='text-align: center;'>{interface_title}</h1></u>")
    with gr.Row():
        with gr.Column(scale=1):
            audio_input.render()
            query_rewrite_toggle = gr.Checkbox(label="Use Query Rewriting", value=True)
            multi_query_choice = gr.Radio(
                choices=["paraphrase", "aspect", "None"],
                value="None",
                label="Multi-Query Type"
            )
            retriever_choice = gr.Radio(
                choices=["faiss", "bm25", "hybrid"],
                value="faiss",
                label="Retriever Type"
            )
            chunk_rerank_toggle = gr.Checkbox(label="Enable Chunk Reranking", value=True)
            fact_check_toggle = gr.Checkbox(label="Enable Fact Checking", value=False)
            tts_choice = gr.Radio(
                choices=["pytts", "xtts"],
                value="pytts",
                label="TTS Engine"
            )

        with gr.Column(scale=2):
            transcript_output.render()
            rag_output.render()
            tts_output.render()

    audio_input.change(
        fn=transcribe_and_rag,
        inputs=[
            audio_input,
            query_rewrite_toggle,
            multi_query_choice,
            retriever_choice,
            fact_check_toggle,
            tts_choice,
            chunk_rerank_toggle,
        ],
        outputs=[transcript_output, rag_output, tts_output]
    )

if __name__ == "__main__":
    ui.launch()
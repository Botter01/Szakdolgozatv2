from utils_module import embedding_model, text_splitter, generationmodel_name, evalmodel_name, fastmodel_name, local_llm, query_model
from szakdoga import *
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import json
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOllama
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from langchain.schema import Document
import random

generator_llm = OllamaLLM(model=evalmodel_name, temperature=0.7)
critic_llm = LangchainLLMWrapper(ChatOllama(model=fastmodel_name, temperature=0.0))
ragas_emb = LangchainEmbeddingsWrapper(embedding_model)

topics = ["Hungarian cuisine", "Budapest history", "Climate change", "1956 Hungarian revolution"]

rag_configs = [
    {
        "name": "faiss_paraphrase",
        "retriever": "faiss",
        "reranking": True,
        "multi_query_paraphrase": True,
        "multi_query_aspect": False,
        "query_rewriting": False,
    },
    {
        "name": "bm25_paraphrase",
        "retriever": "bm25",
        "reranking": True,
        "multi_query_paraphrase": True,
        "multi_query_aspect": False,
        "query_rewriting": False,
    },
    {
        "name": "hybrid_complex",
        "retriever": "hybrid",
        "reranking": False,
        "multi_query_paraphrase": False,
        "multi_query_aspect": True,
        "query_rewriting": True,
    },
]

def create_chunk_corpus(topics):
    all_chunks = []

    for topic in topics:
        docs = WikipediaLoader(query=topic, load_max_docs=2).load()
        chunks = text_splitter.split_documents(docs)
        for ch in chunks:
            all_chunks.append({
                "topic": topic,
                "page_content": ch.page_content,
                "metadata": ch.metadata
            })

    with open("chunk_corpus.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

def generate_test_dataset_from_corpus(llm, max_questions_per_chunk=2, output_file="question_dataset.json"):
    with open("chunk_corpus.json", "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    test_dataset = []

    for chunk in all_chunks:
        text = chunk["page_content"].strip()
        topic = chunk["topic"]

        prompt = PromptTemplate(
            template=("""
            Based on the following text, write {max_questions_per_chunk} English question-answer pairs.
            Keep the questions concise and factual, and the answers short and directly supported by the text.

            Text:
            ---
            {text}
            ---

            Return the result in pure JSON format like this:
            [
            {{"question": "...", "answer": "..."}} ,
            ...
            ]
            """),
            input_variables=["max_questions_per_chunk", "text"],
        )

        formatted_prompt = prompt.format(max_questions_per_chunk=max_questions_per_chunk, text=text)

        content = llm.invoke(formatted_prompt).strip()
        qa_pairs = json.loads(content)

        for qa in qa_pairs:
            test_dataset.append({
                "topic": topic,
                "question": qa["question"],
                "expected": qa["answer"],
                "reference": text
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=2)

def generate_complex_questions_from_corpus(llm=generator_llm, max_questions_per_topic=2, chunks_per_topic=3):
    with open("chunk_corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)

    topics = {}
    for item in corpus:
        topic = item["topic"]
        topics.setdefault(topic, []).append(item)

    complex_dataset = []

    for topic, chunks in topics.items():
        if len(chunks) < chunks_per_topic:
            continue

        selected_chunks = random.sample(chunks, chunks_per_topic)
        combined_text = "\n\n".join([c["page_content"] for c in selected_chunks])

        prompt = PromptTemplate(
            template=(
                """
                You are an expert question generator.
                Based on the following combined text (from multiple paragraphs on the topic of {topic}),
                write {max_questions_per_topic} **complex** English question–answer pairs.
                
                Each question should:
                - require information from more than one paragraph
                - be factual, not opinion-based
                - have a short, direct answer supported by the text
                
                Return pure JSON like this:
                [
                  {{"question": "...", "answer": "..."}},
                  ...
                ]

                Text:
                ---
                {text}
                ---
                """
            ),
            input_variables=["topic", "max_questions_per_topic", "text"],
        )

        formatted_prompt = prompt.format(topic=topic, max_questions_per_topic=max_questions_per_topic, text=combined_text)

        response = llm.invoke(formatted_prompt).strip()
        qa_pairs = json.loads(response)

        for qa in qa_pairs:
            if "question" in qa and "answer" in qa:
                complex_dataset.append({
                    "topic": topic,
                    "question": qa["question"].strip(),
                    "expected": qa["answer"].strip(),
                    "reference": combined_text
                })

    with open("hard_questions.json", "w", encoding="utf-8") as f:
        json.dump(complex_dataset, f, ensure_ascii=False, indent=2)

def get_retrievers_from_corpus():
    with open("chunk_corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
    documents = [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in corpus]

    faiss = FAISS.from_documents(documents, embedding_model)
    bm25 = BM25Retriever.from_documents(documents)

    return faiss.as_retriever(), bm25

def evaluate_rag_configs(rag_configs):
    faiss_retriever, bm25_retriever = get_retrievers_from_corpus()

    for config in rag_configs:
        with mlflow.start_run(run_name=f"Dataset_eval_{config['name']}"):
            all_results = []
            print(f"\nRunning config: {config['name']}")
            
            with open("hard_questions.json", "r", encoding="utf-8") as f:
                json_dataset = json.load(f)

            total_start = time.time()

            for i, item in enumerate(json_dataset):
                query = item["question"]
                expected = item["expected"]
                reference = item["reference"]
                print(f"Kérdés: {query} és ennyi van meg {i+1}/{len(json_dataset)}")

                query_to_use = query_rewriting(query, query_model) if config.get("query_rewriting") else query

                if config.get("multi_query_paraphrase"):
                    query_to_use = multi_paraphrase_query(query_to_use, generator_llm, 2)

                if config.get("multi_query_aspect"):
                    query_to_use = multi_aspect_query(query_to_use, generator_llm, 2)

                if isinstance(query_to_use, str):
                    query_list = [query_to_use]
                else:
                    query_list = query_to_use

                retrieved_chunks = []

                for q in query_list:
                    if config["retriever"] == "faiss":
                        retrieved_chunks.extend(faiss_retriever.get_relevant_documents(q))
                    elif config["retriever"] == "bm25":
                        retrieved_chunks.extend(bm25_retriever.get_relevant_documents(q))
                    elif config["retriever"] == "hybrid":
                        faiss_results = faiss_retriever.get_relevant_documents(q)
                        bm25_results = bm25_retriever.get_relevant_documents(q)
                        retrieved_chunks.extend(faiss_results + bm25_results)

                retrieved_chunks = list({c.page_content: c for c in retrieved_chunks}.values())

                if config.get("reranking"):
                    reranked_chunks = rerank_chuncks(query_list[0], retrieved_chunks)
                else:
                    reranked_chunks = retrieved_chunks[:4]

                answer = generate_answer(query_to_use, reranked_chunks, local_llm, generationmodel_name)

                all_results.append({
                    "config": config["name"],
                    "topic": item["topic"],
                    "question": query_to_use if isinstance(query_to_use, list) else [query_to_use],
                    "expected": expected,
                    "answer": answer,
                    "reference": reference,
                    "retrieved_contexts": [c.page_content for c in reranked_chunks]
                })

            with open("rag_comparison_hard_results.json", "a", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            mlflow.log_metric("total_time", time.time() - total_start)

def rag_evaluation(results_path="rag_comparison_hard_results_last_two.json"):
    with open(results_path, "r", encoding="utf-8") as f:
        json_dataset = json.load(f)

    eval_rows = []
    for item in json_dataset:
        config = item.get("config")
        question = item.get("question")
        answer = item.get("answer")
        expected = item.get("expected")
        reference = item.get("reference", [])
        contexts = item.get("retrieved_contexts", [])

        if isinstance(question, list):
            user_input = " | ".join(question)
        else:
            user_input = question


        eval_rows.append({
            "config": config,
            "user_input": user_input,
            "response": answer,
            "retrieved_contexts": contexts,
            "ground_truths": expected,
            "reference": reference
        })

    samples = [
        SingleTurnSample(
            user_input=row["user_input"],
            response=row["response"],
            retrieved_contexts=row["retrieved_contexts"],
            ground_truths=row["ground_truths"],
            reference=row["reference"]
        )
        for row in eval_rows
    ]

    eval_ds = EvaluationDataset(samples=samples)

    results = evaluate(
        eval_ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=critic_llm,
        embeddings=ragas_emb,
        batch_size=1
    )

    df = results.to_pandas()
    print(df)
    df.to_csv("ragas_evaluation_results_ultimate_complex_last_two.csv", index=False)

#create_chunk_corpus(topics)
#generate_test_dataset_from_corpus(generator_llm)
#evaluate_rag_configs(rag_configs)
#generate_complex_questions_from_corpus()
rag_evaluation()
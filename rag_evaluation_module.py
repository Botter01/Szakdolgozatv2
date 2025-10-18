from utils_module import embedding_model, text_splitter, generationmodel_name, evalmodel_name, fastmodel_name
from szakdoga import build_faiss_retriever, rerank_docs, generate_answer
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import json
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOllama
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

generator_llm = OllamaLLM(model=evalmodel_name, temperature=0.7)
critic_llm = LangchainLLMWrapper(ChatOllama(model=fastmodel_name, temperature=0.0))
ragas_emb = LangchainEmbeddingsWrapper(embedding_model)

topics = ["Hungarian cuisine", "Budapest history", "Lake Balaton"]
all_results = []

def generate_test_dataset_from_docs(docs, llm, max_questions_per_doc=2):

    test_dataset = []

    for i, doc in enumerate(docs):
        text = doc.page_content.strip()

        prompt = PromptTemplate(
            template=("""
            Based on the following text, write {max_questions_per_doc} English question-answer pairs.
            Keep the questions concise and factual, and the answers short and directly supported by the text.

            Text:
            ---
            {text}
            ---

            Return the result in pure JSON format like this:
            [
            {{"question": "...", "answer": "..."}},
            ...
            ]
            """
            ),
            input_variables=["max_questions_per_doc", "text"],
        )

        formatted_prompt = prompt.format(max_questions_per_doc=max_questions_per_doc, text=text)
        content = llm.invoke(formatted_prompt).strip()

        qa_pairs = json.loads(content)

        for qa in qa_pairs:
            if "question" in qa and "answer" in qa:
                test_dataset.append({
                    "question": qa["question"],
                    "expected": qa["answer"]
                })

        print(f"Generated {len(qa_pairs)} questions from document {i+1}/{len(docs)}")

    return test_dataset
"""
for topic in topics:

    retriever, docs = build_faiss_retriever(topic, embedding_model, text_splitter)
    print(len(docs))

    test_dataset = generate_test_dataset_from_docs(docs, generator_llm, max_questions_per_doc=2)

    topic_results = []
    for i, item in enumerate(test_dataset):
        question = item["question"]
        expected = item["expected"]

        print(f"\n [{topic}] Q{i+1}: {question}")
        reranked_docs = rerank_docs(question, docs)
        answer = generate_answer(question, reranked_docs, generator_llm, model_name=generation_model)

        topic_results.append({
            "topic": topic,
            "question": question,
            "expected": expected,
            "answer": answer
        })

    all_results.extend(topic_results)

with open("test_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
"""

with open("test_results.json", "r", encoding="utf-8") as f:
    json_dataset = json.load(f)

def rag_evaluation(json_dataset):

    eval_rows = []
    for item in json_dataset:
        topic = item.get("topic") 
        question = item.get("question")
        answer = item.get("answer")
        expected = item.get("expected")

        retriever, docs = build_faiss_retriever(topic, embedding_model, text_splitter)

        retrieved_docs = retriever.get_relevant_documents(question)

        reranked = rerank_docs(question, retrieved_docs)

        contexts = [doc.page_content for doc in reranked]

        eval_rows.append({
            "user_input": question,  
            "response": answer,            
            "retrieved_contexts": contexts,    
            "ground_truths": expected,         
            "reference": "\n\n".join(contexts)  
        })

    samples = []
    for item in eval_rows:
        samples.append(
            SingleTurnSample(
                user_input=item["user_input"],
                response=item["response"],
                retrieved_contexts=item["retrieved_contexts"],
                ground_truths=item.get("ground_truths", ""),
                reference=item.get("reference", "")
            )
        )

    eval_ds = EvaluationDataset(samples=samples)

    results = evaluate(
        eval_ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=critic_llm,
        embeddings=ragas_emb,
        batch_size=1,
        column_map={
            "question": "question",
            "answer": "answer",
            "contexts": "contexts",
            "ground_truths": "ground_truths",
            "reference":"reference"
        },
    )

    df = results.to_pandas()
    print(df)
    df.to_csv("ragas_evaluation_results.csv", index=False)
    print("Saved ragas_evaluation_results.csv")

rag_evaluation(json_dataset)
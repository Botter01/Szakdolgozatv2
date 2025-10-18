from utils_module import embedding_model, text_splitter, generationmodel_name, evalmodel_name, fastmodel_name
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

generator_llm = OllamaLLM(model=evalmodel_name, temperature=0.7)
critic_llm = LangchainLLMWrapper(ChatOllama(model=fastmodel_name, temperature=0.0))
ragas_emb = LangchainEmbeddingsWrapper(embedding_model)

topics = ["Hungarian cuisine", "Budapest history", "Lake Balaton"]
all_results = []

def generate_test_dataset_from_chuncks(chuncks, llm, max_questions_per_chunck=2):

    test_dataset = []

    for i, chunck in enumerate(chuncks):
        text = chunck.page_content.strip()

        prompt = PromptTemplate(
            template=("""
            Based on the following text, write {max_questions_per_chunck} English question-answer pairs.
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
            input_variables=["max_questions_per_chunck", "text"],
        )

        formatted_prompt = prompt.format(max_questions_per_chunck=max_questions_per_chunck, text=text)
        content = llm.invoke(formatted_prompt).strip()

        qa_pairs = json.loads(content)

        for qa in qa_pairs:
            if "question" in qa and "answer" in qa:
                test_dataset.append({
                    "question": qa["question"],
                    "expected": qa["answer"]
                })

        print(f"Generated {len(qa_pairs)} questions from document {i+1}/{len(chuncks)}")

    return test_dataset
"""
for topic in topics:

    chuncks = faiss_retriever(topic, embedding_model, text_splitter)

    test_dataset = generate_test_dataset_from_chuncks(chuncks, generator_llm, max_questions_per_doc=2)

    topic_results = []
    for i, item in enumerate(test_dataset):
        question = item["question"]
        expected = item["expected"]

        print(f"\n [{topic}] Q{i+1}: {question}")
        reranked_chuncks = rerank_chuncks(question, chuncks)
        answer = generate_answer(question, reranked_chuncks, generator_llm, model_name=generation_model)

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

        retrieved_chuncks = faiss_retriever(topic, embedding_model, text_splitter)

        reranked_chuncks = rerank_chuncks(question, retrieved_chuncks)

        contexts = [chunck.page_content for chunck in reranked_chuncks]

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
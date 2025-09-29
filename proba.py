import whisper
import gradio as gr
import mlflow
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA


whisper_model = whisper.load_model("small")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

llm_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,
)

local_llm = HuggingFacePipeline(pipeline=llm_pipeline)


def transcribe_and_rag(audio_path):
    with mlflow.start_run():
        result = whisper_model.transcribe(audio_path, language="en")
        query = result["text"]
        print(query)

        loader = WikipediaLoader(query=query, lang="en", load_max_docs=5)
        docs = loader.load()
        print(len(docs))

        vectorstore = FAISS.from_documents(docs, embedding_model)
        retriever = vectorstore.as_retriever()

        rag_chain = RetrievalQA.from_chain_type(
            llm=local_llm,
            chain_type="refine",
            retriever=retriever,
            chain_type_kwargs={
                "question_prompt": PromptTemplate(
                    template=(
                        "Use the following context to answer the question.\n\n"
                        "Context: {context_str}\n"
                        "Question: {question}\n"
                        "Answer:"
                    ),
                    input_variables=["question", "context_str"], 
                ),
                "refine_prompt": PromptTemplate(
                    template=(
                        "The original answer was:\n{existing_answer}\n\n"
                        "Refine it using the new context below. "
                        "If it doesn't add anything useful, keep the original answer.\n\n"
                        "Context: {context_str}\n"
                        "Question: {question}\n"
                        "Refined Answer:"
                    ),
                    input_variables=["question", "context_str", "existing_answer"],
                ),
            },
            return_source_documents=True,
        )
        
        print("Expected input keys:", rag_chain.input_keys)
        print("Query variable:", query)
        print("Type of query:", type(query))
        response = rag_chain.invoke({"query": query})
        answer = response["answer"]
        sources = response.get("source_documents", [])

        mlflow.log_param("query", query)
        mlflow.log_param("num_docs", len(docs))
        mlflow.log_param("answer", answer)
        mlflow.log_metric("answer_length", len(answer))
        mlflow.log_param("sources", sources)

    return query, f"Leirat: {query}\n\nVálasz: {answer}"

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
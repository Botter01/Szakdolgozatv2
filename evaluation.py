from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

eval_dataset = []

sample_queries = [
    "Which CEO is widely recognized for democratizing AI education through platforms like Coursera?",
    "Who is Sam Altman?",
    "Who is Demis Hassabis and how did he gained prominence?",
    "Who is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's product ecosystem?",
    "How did Arvind Krishna transformed IBM?",
]

expected_responses = [
    "Andrew Ng is the CEO of Landing AI and is widely recognized for democratizing AI education through platforms like Coursera.",
    "Sam Altman is the CEO of OpenAI and has played a key role in advancing AI research and development. He strongly advocates for creating safe and beneficial AI technologies.",
    "Demis Hassabis is the CEO of DeepMind and is celebrated for his innovative approach to artificial intelligence. He gained prominence for developing systems like AlphaGo that can master complex games.",
    "Sundar Pichai is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's vast product ecosystem. His leadership has significantly enhanced user experiences globally.",
    "Arvind Krishna is the CEO of IBM and has transformed the company towards cloud computing and AI solutions. He focuses on delivering cutting-edge technologies to address modern business challenges.",
]

for query, reference in zip(sample_queries, expected_responses):
    relevant_docs = retriever.invoke(query)
    response = qa_chain.invoke({"context": format_docs(relevant_docs), "query": query})
    eval_dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
            "response": response,
            "reference": reference,
        }
    )

evaluation_dataset = EvaluationDataset.from_list(eval_dataset)
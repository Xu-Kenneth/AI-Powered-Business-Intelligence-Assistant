"""QAEvalChain-based evaluation of the RAG system."""

import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain.evaluation.qa import QAEvalChain
from langchain_core.prompts import PromptTemplate

load_dotenv()

EVAL_QA_PAIRS: List[Dict[str, str]] = [
    {
        "query": "What is the total sales amount across all orders?",
        "answer": "The total sales can be found in the overall business performance summary.",
    },
    {
        "query": "Which product has the highest average customer satisfaction?",
        "answer": "The product with the highest satisfaction score can be found in the product analysis section.",
    },
    {
        "query": "Which region generates the most sales?",
        "answer": "The regional analysis shows which region leads in total sales.",
    },
    {
        "query": "What age group places the most orders?",
        "answer": "The customer demographics section shows orders broken down by age group.",
    },
    {
        "query": "How do male and female customers compare in total sales?",
        "answer": "The gender analysis section shows sales and order counts split by gender.",
    },
]


def run_evaluation(rag_chain) -> List[Dict]:
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )

    eval_chain = QAEvalChain.from_llm(llm)

    predictions = []
    for pair in EVAL_QA_PAIRS:
        result = rag_chain.invoke({"question": pair["query"]})
        predictions.append({"query": pair["query"], "result": result["answer"]})

    graded = eval_chain.evaluate(
        EVAL_QA_PAIRS,
        predictions,
        question_key="query",
        answer_key="answer",
        prediction_key="result",
    )

    results = []
    for pair, pred, grade in zip(EVAL_QA_PAIRS, predictions, graded):
        results.append({
            "question": pair["query"],
            "expected": pair["answer"],
            "actual": pred["result"],
            "grade": grade.get("results", "N/A"),
        })
    return results

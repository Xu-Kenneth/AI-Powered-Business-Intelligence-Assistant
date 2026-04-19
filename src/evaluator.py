"""LLM-as-judge evaluation of the RAG system."""

import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

load_dotenv()

EVAL_QA_PAIRS: List[Dict[str, str]] = [
    {
        "query": "What is the total sales amount across all orders?",
        "answer": "The total sales figure is available in the overall business performance summary.",
    },
    {
        "query": "Which product has the highest average customer satisfaction?",
        "answer": "The product with the highest satisfaction score is listed in the product analysis section.",
    },
    {
        "query": "Which region generates the most sales?",
        "answer": "The regional analysis shows which region leads in total sales revenue.",
    },
    {
        "query": "What age group places the most orders?",
        "answer": "The customer demographics section shows order counts broken down by age group.",
    },
    {
        "query": "How do male and female customers compare in total sales?",
        "answer": "The gender analysis section shows sales and order counts split by gender.",
    },
]

JUDGE_PROMPT = """You are grading an AI assistant's answer to a business intelligence question.

Question: {question}
Reference answer: {reference}
AI answer: {prediction}

Does the AI answer correctly address the question using specific data? Reply with exactly one word: CORRECT or INCORRECT."""


def run_evaluation(rag_chain) -> List[Dict]:
    judge = ChatAnthropic(
        model="claude-sonnet-4-6",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )

    results = []
    for pair in EVAL_QA_PAIRS:
        prediction = rag_chain.invoke(
            {"question": pair["query"]},
            config={"configurable": {"session_id": f"eval_{pair['query'][:20]}"}},
        )

        verdict = judge.invoke([HumanMessage(content=JUDGE_PROMPT.format(
            question=pair["query"],
            reference=pair["answer"],
            prediction=prediction,
        ))]).content.strip()

        results.append({
            "question": pair["query"],
            "expected": pair["answer"],
            "actual": prediction,
            "grade": verdict,
        })

    return results

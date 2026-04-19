# InsightForge — AI-Powered Business Intelligence Assistant

A conversational BI assistant powered by RAG — ask plain-English questions about sales data and get accurate, data-grounded answers backed by FAISS vector search, LangChain, and Anthropic Claude.

## Highlights

- **Conversational Q&A with memory** — multi-turn dialogue via `RunnableWithMessageHistory`; follow-up questions retain full context
- **RAG over structured data** — sales statistics pre-summarised into semantic documents, embedded with `all-MiniLM-L6-v2`, and retrieved via FAISS
- **7 interactive dashboards** — Plotly charts covering sales trends, product performance, regional breakdown, customer demographics, and satisfaction scores
- **LLM-as-judge evaluation** — automated benchmarking using Claude to grade RAG responses against reference QA pairs

## Stack

LangChain · FAISS · Anthropic Claude · HuggingFace Embeddings · Streamlit · Plotly · pandas

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your Anthropic API key
streamlit run app.py
```

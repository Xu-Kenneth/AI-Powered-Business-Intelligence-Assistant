# AI-Powered Business Intelligence Assistant

A Retrieval-Augmented Generation (RAG) BI assistant that lets you have a natural-language conversation with your sales data, backed by FAISS vector search, LangChain, and Anthropic Claude.

---

## Features

- **Conversational Q&A** — ask plain-English questions about sales trends, product performance, regional analysis, and customer demographics
- **Persistent memory** — follow-up questions maintain conversation context via `ConversationBufferMemory`
- **Interactive dashboards** — 7 Plotly charts covering time-series, product, regional, demographic, and satisfaction analytics
- **Automated evaluation** — QAEvalChain grades the RAG chain against 5 reference QA pairs using Claude as the judge

---

## Architecture

```
sales_data.csv
      │
      ▼
data_loader.py          — load & compute statistical summary
      │
      ▼
knowledge_base.py       — convert summary → Documents → FAISS index
      │
      ▼
rag_system.py           — FAISS retriever + Claude LLM + ConversationBufferMemory
      │
      ▼
app.py (Streamlit)      — Chat / Visualizations / Evaluation tabs
```

---

## Dataset

`data/sales_data.csv` — 3 000 sales records (2022–2028)

| Column | Type | Description |
|---|---|---|
| Date | date | Transaction date |
| Product | str | Widget A / B / C / D |
| Region | str | North / South / East / West |
| Sales | int | Order value (100–999) |
| Customer_Age | int | Customer age (18–69) |
| Customer_Gender | str | Male / Female |
| Customer_Satisfaction | float | Rating 1.0–5.0 |

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Xu-Kenneth/AI-Powered-Business-Intelligence-Assistant.git
cd AI-Powered-Business-Intelligence-Assistant

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env and paste your Anthropic API key

# 5. Run the app
streamlit run app.py
```

---

## Usage

1. Open the **Chat** tab and ask questions such as:
   - "Which product has the highest customer satisfaction?"
   - "How did sales trend year-over-year?"
   - "Compare North and South region performance."

2. Explore the **Visualizations** tab for pre-built charts.

3. Click **Run Evaluation** in the **Evaluation** tab to benchmark the RAG chain.

---

## Tech Stack

| Component | Library |
|---|---|
| LLM | Anthropic Claude (`claude-sonnet-4-6`) |
| Orchestration | LangChain (v0.3+) |
| Vector store | FAISS (`faiss-cpu`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| UI | Streamlit |
| Data | pandas, numpy |
| Charts | Plotly |
| Evaluation | LangChain `QAEvalChain` |

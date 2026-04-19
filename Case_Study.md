# InsightForge: A RAG-Based Business Intelligence Assistant
## Case Study

---

### 1. Problem Statement

Business teams routinely need to interrogate sales data to identify trends, compare product performance, and understand customer behaviour — but traditional BI tools require SQL fluency or pre-built dashboards that freeze the questions users can ask. The goal of InsightForge was to build a conversational assistant that lets any stakeholder ask ad-hoc questions in plain English and receive accurate, context-grounded answers derived from real sales data.

The project scope (derived from the Simplilearn capstone problem statement) required:
- A Retrieval-Augmented Generation (RAG) pipeline grounding LLM responses in structured data
- Persistent conversational memory enabling multi-turn dialogue
- Interactive data visualisations
- Automated model evaluation using LLM-as-judge methodology

---

### 2. Dataset

The dataset contains 3 000 sales transactions spanning 2022-01-01 to 2028-11-04, with the following schema:

| Field | Values |
|---|---|
| Date | 2022-01-01 – 2028-11-04 |
| Product | Widget A, Widget B, Widget C, Widget D |
| Region | North, South, East, West |
| Sales | $100 – $999 per order |
| Customer_Age | 18 – 69 |
| Customer_Gender | Male / Female |
| Customer_Satisfaction | 1.0 – 5.0 |

---

### 3. Methodology

#### 3.1 Data Preparation
Raw tabular data is loaded with pandas. Derived columns — Year, Month, Quarter, and AgeGroup (five bins: 18-24, 25-34, 35-44, 45-54, 55+) — are computed at load time. A comprehensive summary dictionary is then produced, covering aggregate KPIs and group-level breakdowns across every meaningful dimension.

#### 3.2 Knowledge Base Construction
The summary dictionary is converted into plain-text `Document` objects — one per analytical theme (overall performance, annual trends, quarterly breakdown, product analysis, regional analysis, customer demographics). These documents are embedded using HuggingFace `all-MiniLM-L6-v2` and indexed in a FAISS flat vector store that is persisted to disk on first build and reloaded on subsequent runs.

This pre-computation approach avoids running the LLM over raw tabular rows, improves retrieval quality by surfacing semantically coherent chunks, and keeps inference costs low.

#### 3.3 RAG Chain
At query time, LangChain's `ConversationalRetrievalChain` retrieves the top-4 most relevant documents for the user's question, injects them into a structured prompt alongside the conversation history, and invokes Anthropic Claude (`claude-sonnet-4-6`, temperature 0.1) to generate a grounded, data-specific answer.

`ConversationBufferMemory` stores the full dialogue history, enabling coherent multi-turn Q&A (e.g., "Which product is best?" → "How does that compare to the second-best one?").

#### 3.4 Visualisations
Seven Plotly charts are generated directly from the pandas DataFrame:
- Monthly sales trend line chart
- Product sales bar chart
- Regional sales pie chart
- Customer satisfaction by product (with overall average reference line)
- Sales by age group and gender (grouped bars)
- Year × Quarter heatmap
- Satisfaction score distribution histogram

#### 3.5 Evaluation (LLMOps)
Five reference question-answer pairs covering different analytical topics are evaluated using LangChain's `QAEvalChain`. The RAG chain generates a prediction for each question; Claude then grades each prediction as CORRECT or INCORRECT relative to the reference answer. This creates a reproducible, human-readable benchmark for regression testing as the knowledge base or prompt evolves.

---

### 4. Architecture

```
┌──────────────────────────────────────────────┐
│                  Streamlit UI                │
│  ┌─────────┐  ┌──────────────┐  ┌────────┐  │
│  │  Chat   │  │Visualizations│  │  Eval  │  │
│  └────┬────┘  └──────┬───────┘  └───┬────┘  │
│       │              │              │        │
└───────┼──────────────┼──────────────┼────────┘
        │              │              │
        ▼              ▼              ▼
  rag_system.py  visualizations.py  evaluator.py
        │
   ┌────┴────────────────┐
   │   FAISS Retriever   │◄── knowledge_base.py ◄── data_loader.py
   └────┬────────────────┘                               │
        │                                         sales_data.csv
        ▼
  Claude LLM (claude-sonnet-4-6)
        │
  ConversationBufferMemory
```

---

### 5. Results

The system successfully:
- Answered multi-turn natural-language queries about sales figures, product satisfaction, regional performance, and customer demographics with factually grounded responses
- Maintained conversation context across follow-up questions without repetition
- Rendered 7 interactive Plotly charts with accurate aggregations
- Completed all 5 evaluation benchmarks, with Claude-graded CORRECT/INCORRECT verdicts providing an objective quality signal

---

### 6. Key Design Decisions

| Decision | Rationale |
|---|---|
| Summary-doc RAG (not row-level) | Aggregate statistics fit in 8 documents; row-level RAG would require hundreds of chunks and degrade precision |
| `all-MiniLM-L6-v2` embeddings | Fast, locally-run, no API calls for embedding; sufficient semantic coverage for structured summaries |
| Temperature 0.1 | Minimises hallucination for factual BI answers |
| Claude as evaluator | Same-model judging is consistent; Claude's instruction-following makes grading reliable |
| FAISS persisted to disk | Avoids re-embedding on every app restart; rebuild forced with `force_rebuild=True` |

---

### 7. Limitations & Future Work

- **Static knowledge base** — summaries are precomputed; live data ingestion would require scheduled rebuilds
- **No SQL fallback** — complex filtering ("sales > $500 in Q3 2026") requires the LLM to reason from aggregate summaries rather than executing a direct query
- **Small dataset** — 3 000 rows and 4 products limit the granularity of insights; a production deployment would connect to a real data warehouse
- **Single-user memory** — `ConversationBufferMemory` is per-session; a multi-user deployment would need session-scoped memory stores

---

### 8. Conclusion

InsightForge demonstrates that a well-structured RAG pipeline — where tabular data is pre-summarised into semantically coherent text documents before embedding — can deliver accurate, conversational business intelligence without exposing the LLM to raw rows or requiring SQL expertise from end users. The combination of LangChain, FAISS, Anthropic Claude, and Streamlit provides a production-ready scaffold that can be extended to larger datasets and richer analytical domains.

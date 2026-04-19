# InsightForge — Build Changelog

All notable changes made during the construction of this project, in chronological order.

---

## Session 1 — Initial Setup

### Added
- `.env.example` — API key template; user pastes real key into `.env`
- `.gitignore` — excludes `.env`, `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `faiss_index/`, `.DS_Store`, `*.ipynb_checkpoints`
- `requirements.txt` — pinned dependencies: langchain, langchain-anthropic, langchain-community, langchain-core, anthropic, faiss-cpu, sentence-transformers, streamlit, pandas, numpy, matplotlib, plotly, python-dotenv, scikit-learn
- `src/__init__.py` — empty package marker

### Added (later deleted/replaced)
- `data/generate_data.py` — synthetic 5 000-row CSV generator using old schema (OrderID, TotalSales, Profit, Category, City, PaymentMethod, …)
- `data/sales_data.csv` (generated) — 5 000-row synthetic dataset from above script
- `src/data_loader.py` (v1) — computed TotalSales, Profit, Category, Region, City, AgeGroup, Gender, PaymentMethod stats for old schema
- `src/knowledge_base.py` (v1) — FAISS-backed knowledge base documents for old schema

---

## Session 1 — Schema Pivot (user-provided dataset)

User provided their own `sales_data.csv` and five research PDFs. Instruction: "Use this dataset. Remove what you have generated before and replace it with what I just provided."

### Deleted
- `data/generate_data.py`
- `data/sales_data.csv` (old generated version)

### Changed — `data/sales_data.csv`
- **New schema**: `Date, Product, Region, Sales, Customer_Age, Customer_Gender, Customer_Satisfaction`
- Products: Widget A, Widget B, Widget C, Widget D
- Regions: North, South, East, West
- Date range: 2022-01-01 to 2028-11-04
- 3 000 rows, no OrderID/Profit/Category/City/PaymentMethod columns

### Changed — `src/data_loader.py` (v2)
- Removed all references to TotalSales, Profit, Category, City, AgeGroup (hard-coded), PaymentMethod
- Added derived columns: Year, Month, Quarter (from `Date`), AgeGroup (pd.cut on Customer_Age with bins 18-25-35-45-55-70)
- `compute_summary()` now computes: total_sales, total_orders, avg/median/std order value, avg/median satisfaction, sales/orders by year/quarter/month/product/region/age_group/gender, avg satisfaction by product/region/gender

### Changed — `src/knowledge_base.py` (v2)
- Removed old document sections: annual profit, category, top-products, top-cities, payment-methods, statistical profit measures
- Added new document sections: overall performance (with satisfaction), annual sales, quarterly, monthly avg, product analysis (sales + orders + satisfaction), regional analysis (sales + orders + satisfaction), customer demographics (age + gender with satisfaction)

---

## Session 1 — New Modules

### Added — `src/rag_system.py`
- `build_rag_chain()` — assembles `ConversationalRetrievalChain` with:
  - Claude `claude-sonnet-4-6` as LLM (temperature 0.1)
  - FAISS retriever (k=4 documents)
  - `ConversationBufferMemory` keyed on `chat_history`
  - Custom system prompt instructing InsightForge persona with data-driven, unit-aware answers

### Added — `src/visualizations.py`
- `sales_over_time(df)` — Plotly line chart, monthly aggregation
- `sales_by_product(df)` — Plotly bar chart by product
- `sales_by_region(df)` — Plotly pie chart by region
- `satisfaction_by_product(df)` — Plotly bar with overall-average dashed line
- `sales_by_age_gender(df)` — Grouped bar by age group + gender
- `quarterly_heatmap(df)` — Imshow heatmap of Quarter × Year
- `satisfaction_distribution(df)` — Histogram of satisfaction scores

### Added — `src/evaluator.py`
- 5 hand-crafted QA pairs covering overall sales, product satisfaction, region, age group, gender
- `run_evaluation(rag_chain)` — invokes chain on each pair, grades with `QAEvalChain` (Claude as judge)

### Added — `app.py`
- Streamlit 3-tab layout: **Chat**, **Visualizations**, **Evaluation**
- Chat tab: streaming-style conversational Q&A with source-document expander
- Visualizations tab: 4 KPI metrics + 7 Plotly charts
- Evaluation tab: button-triggered `run_evaluation`, colour-coded CORRECT/INCORRECT grades

---

## Session 1 — Documentation & Repository

### Added
- `CHANGELOG.md` — this file
- `README.md` — project overview, setup instructions, usage guide, architecture diagram (text)
- `Case_Study.md` — narrative case study: problem statement, methodology, architecture, results, conclusions
- `notebooks/InsightForge_Demo.ipynb` — end-to-end demo: data loading → summary → knowledge base → RAG chain → sample queries → evaluation

### Repository
- GitHub remote: https://github.com/Xu-Kenneth/AI-Powered-Business-Intelligence-Assistant
- Pushed all tracked files; `.env`, `faiss_index/`, and `__pycache__/` excluded via `.gitignore`

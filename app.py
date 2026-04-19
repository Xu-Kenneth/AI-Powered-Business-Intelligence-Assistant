"""InsightForge — AI-Powered Business Intelligence Assistant (Streamlit UI)."""

import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.rag_system import build_rag_chain
from src.visualizations import (
    sales_over_time,
    sales_by_product,
    sales_by_region,
    satisfaction_by_product,
    sales_by_age_gender,
    quarterly_heatmap,
    satisfaction_distribution,
)
from src.evaluator import run_evaluation

st.set_page_config(
    page_title="InsightForge BI Assistant",
    page_icon="📊",
    layout="wide",
)

st.title("📊 InsightForge — AI-Powered Business Intelligence Assistant")


@st.cache_resource(show_spinner="Building knowledge base…")
def get_chain():
    return build_rag_chain()


@st.cache_data(show_spinner="Loading data…")
def get_data():
    return load_data()


tab_chat, tab_viz, tab_eval = st.tabs(["💬 Chat", "📈 Visualizations", "🧪 Evaluation"])

# ── Chat tab ──────────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Ask InsightForge about your sales data")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chain = get_chain()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("e.g. Which product has the highest satisfaction score?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = chain.invoke({"question": prompt})
                answer = response["answer"]
            st.markdown(answer)

            with st.expander("Retrieved context"):
                for doc in response.get("source_documents", []):
                    st.caption(f"Topic: {doc.metadata.get('topic', 'unknown')}")
                    st.text(doc.page_content[:400])

        st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Visualizations tab ────────────────────────────────────────────────────────
with tab_viz:
    df = get_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    col2.metric("Total Orders", f"{len(df):,}")
    col3.metric("Avg Order Value", f"${df['Sales'].mean():,.0f}")
    col4.metric("Avg Satisfaction", f"{df['Customer_Satisfaction'].mean():.2f}/5.0")

    st.plotly_chart(sales_over_time(df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(sales_by_product(df), use_container_width=True)
    with c2:
        st.plotly_chart(sales_by_region(df), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(satisfaction_by_product(df), use_container_width=True)
    with c4:
        st.plotly_chart(sales_by_age_gender(df), use_container_width=True)

    st.plotly_chart(quarterly_heatmap(df), use_container_width=True)
    st.plotly_chart(satisfaction_distribution(df), use_container_width=True)

# ── Evaluation tab ────────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("RAG System Evaluation (QAEvalChain)")
    st.info(
        "Runs 5 predefined question-answer pairs through the RAG chain and grades "
        "the responses using Claude as the judge."
    )

    if st.button("▶ Run Evaluation", type="primary"):
        chain = get_chain()
        with st.spinner("Evaluating…"):
            results = run_evaluation(chain)

        for r in results:
            with st.expander(r["question"]):
                st.markdown(f"**Expected:** {r['expected']}")
                st.markdown(f"**Actual:** {r['actual']}")
                grade = r["grade"]
                color = "green" if "CORRECT" in str(grade).upper() else "red"
                st.markdown(f"**Grade:** :{color}[{grade}]")

"""Builds and persists a FAISS vector store from data summary documents."""

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.data_loader import load_data, compute_summary

FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
EMBED_MODEL = "all-MiniLM-L6-v2"


def _summary_to_documents(summary: dict) -> List[Document]:
    docs = []

    docs.append(Document(
        page_content=(
            f"Overall Business Performance:\n"
            f"- Total Sales: ${summary['total_sales']:,.2f}\n"
            f"- Total Orders: {summary['total_orders']:,}\n"
            f"- Average Order Value: ${summary['avg_order_value']:,.2f}\n"
            f"- Median Order Value: ${summary['median_order_value']:,.2f}\n"
            f"- Std Dev of Order Value: ${summary['std_order_value']:,.2f}\n"
            f"- Average Customer Satisfaction: {summary['avg_satisfaction']}/5.0\n"
            f"- Median Customer Satisfaction: {summary['median_satisfaction']}/5.0\n"
            f"- Data Range: {summary['date_range']}"
        ),
        metadata={"topic": "overall_performance"}
    ))

    yr_lines = "\n".join(
        f"  {yr}: ${val:,.2f}" for yr, val in summary["sales_by_year"].items()
    )
    docs.append(Document(
        page_content=f"Annual Sales Performance:\n{yr_lines}",
        metadata={"topic": "annual_trends"}
    ))

    q_lines = "\n".join(
        f"  {yr} Q{q}: ${val:,.2f}" for (yr, q), val in summary["sales_by_quarter"].items()
    )
    docs.append(Document(
        page_content=f"Quarterly Sales Breakdown:\n{q_lines}",
        metadata={"topic": "quarterly_trends"}
    ))

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    m_lines = "\n".join(
        f"  {month_names[m]}: ${val:,.2f}" for m, val in summary["avg_monthly_sales"].items()
    )
    docs.append(Document(
        page_content=f"Average Monthly Sales (across all years):\n{m_lines}",
        metadata={"topic": "monthly_trends"}
    ))

    prod_lines = "\n".join(
        f"  {prod}: Sales=${summary['sales_by_product'][prod]:,.2f}, "
        f"Orders={summary['orders_by_product'][prod]:,}, "
        f"Avg Satisfaction={summary['avg_satisfaction_by_product'][prod]}/5.0"
        for prod in summary["sales_by_product"]
    )
    docs.append(Document(
        page_content=f"Sales by Product:\n{prod_lines}",
        metadata={"topic": "product_analysis"}
    ))

    reg_lines = "\n".join(
        f"  {reg}: Sales=${summary['sales_by_region'][reg]:,.2f}, "
        f"Orders={summary['orders_by_region'][reg]:,}, "
        f"Avg Satisfaction={summary['avg_satisfaction_by_region'][reg]}/5.0"
        for reg in summary["sales_by_region"]
    )
    docs.append(Document(
        page_content=f"Sales by Region:\n{reg_lines}",
        metadata={"topic": "regional_analysis"}
    ))

    age_lines = "\n".join(
        f"  {ag}: Sales=${summary['sales_by_age_group'][ag]:,.2f}, "
        f"Orders={summary['orders_by_age_group'][ag]:,}"
        for ag in summary["sales_by_age_group"]
    )
    gender_lines = "\n".join(
        f"  {g}: Sales=${summary['sales_by_gender'][g]:,.2f}, "
        f"Orders={summary['orders_by_gender'][g]:,}, "
        f"Avg Satisfaction={summary['avg_satisfaction_by_gender'][g]}/5.0"
        for g in summary["sales_by_gender"]
    )
    docs.append(Document(
        page_content=(
            f"Customer Demographics - Age Group Analysis:\n{age_lines}\n\n"
            f"Customer Demographics - Gender Analysis:\n{gender_lines}"
        ),
        metadata={"topic": "customer_demographics"}
    ))

    return docs


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_knowledge_base(force_rebuild: bool = False) -> FAISS:
    """Build or load the FAISS vector store."""
    embeddings = get_embeddings()

    if not force_rebuild and os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )

    df = load_data()
    summary = compute_summary(df)
    documents = _summary_to_documents(summary)

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

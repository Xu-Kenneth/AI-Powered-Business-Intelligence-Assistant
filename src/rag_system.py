"""RAG chain with conversation memory backed by FAISS retriever and Claude LLM."""

import os
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

from src.knowledge_base import build_knowledge_base

load_dotenv()

LLM_MODEL = "claude-sonnet-4-6"
RETRIEVER_K = 4

_session_histories: dict[str, ChatMessageHistory] = {}


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_histories:
        _session_histories[session_id] = ChatMessageHistory()
    return _session_histories[session_id]


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are InsightForge, an expert business intelligence assistant. "
     "Answer questions about sales performance, product trends, regional analysis, "
     "and customer demographics using the retrieved business data context below. "
     "Be concise, data-driven, and specific. Quote figures with units ($ for sales, /5.0 for satisfaction). "
     "If the context does not contain enough information to answer, say so clearly.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def build_rag_chain():
    vectorstore = build_knowledge_base()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    llm = ChatAnthropic(
        model=LLM_MODEL,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.1,
    )

    chain = (
        RunnablePassthrough.assign(context=lambda x: _format_docs(retriever.invoke(x["question"])))
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

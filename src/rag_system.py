"""RAG chain with conversation memory backed by FAISS retriever and Claude LLM."""

import os
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from src.knowledge_base import build_knowledge_base

load_dotenv()

LLM_MODEL = "claude-sonnet-4-6"
RETRIEVER_K = 4

SYSTEM_PROMPT = """You are InsightForge, an expert business intelligence assistant.
Answer questions about sales performance, product trends, regional analysis, and customer demographics
using the retrieved business data context below.

Be concise, data-driven, and specific. When quoting figures always include units ($ for sales, /5.0 for satisfaction scores).
If the context does not contain enough information to answer, say so clearly.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""


def build_rag_chain() -> ConversationalRetrievalChain:
    vectorstore = build_knowledge_base()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    llm = ChatAnthropic(
        model=LLM_MODEL,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.1,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=SYSTEM_PROMPT,
            )
        },
    )
    return chain

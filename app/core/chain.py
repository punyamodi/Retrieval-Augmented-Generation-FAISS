from __future__ import annotations

from typing import List, Optional

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableSerializable
from langchain_community.vectorstores import FAISS

from app.core.llm import get_llm
from app.core.retriever import MultiQueryContextualRetriever, SimpleRetriever


RAG_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on the provided context documents.

Rules:
- Answer only based on the context provided below
- If the context does not contain enough information, say so clearly
- Cite relevant sources when possible
- Be concise and accurate

Context:
{context}"""

RAG_HUMAN_PROMPT = """Question: {question}

Answer:"""

STANDALONE_QUESTION_PROMPT = """Given the conversation history and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "unknown")
        parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class RAGChain:
    def __init__(self, store: FAISS, use_multi_query: bool = True):
        self._store = store
        self._llm = get_llm()
        self._use_multi_query = use_multi_query
        self._retriever = MultiQueryContextualRetriever(store) if use_multi_query else SimpleRetriever(store)

    def _build_chain(self) -> RunnableSerializable:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT),
                ("human", RAG_HUMAN_PROMPT),
            ]
        )
        return (
            {
                "context": lambda x: _format_docs(
                    self._retriever.retrieve(x["question"])
                    if self._use_multi_query
                    else self._retriever.retrieve(x["question"])
                ),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )

    def run(self, question: str) -> dict:
        if self._use_multi_query:
            source_docs = self._retriever.retrieve(question)
        else:
            source_docs = self._retriever.retrieve(question)

        context = _format_docs(source_docs)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT),
                ("human", RAG_HUMAN_PROMPT),
            ]
        )
        chain = prompt | self._llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return {
            "answer": answer,
            "source_documents": source_docs,
            "num_sources": len(source_docs),
        }

    def run_with_history(self, question: str, chat_history: List[dict]) -> dict:
        if chat_history:
            history_text = "\n".join(
                f"Human: {m['question']}\nAssistant: {m['answer']}"
                for m in chat_history[-4:]
            )
            standalone_prompt = PromptTemplate.from_template(STANDALONE_QUESTION_PROMPT)
            chain = standalone_prompt | self._llm | StrOutputParser()
            question = chain.invoke({"chat_history": history_text, "question": question})

        return self.run(question)


def build_rag_chain(store: FAISS, use_multi_query: bool = True) -> RAGChain:
    return RAGChain(store=store, use_multi_query=use_multi_query)

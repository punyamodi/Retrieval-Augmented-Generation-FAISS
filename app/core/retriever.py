from __future__ import annotations

import logging
from typing import List, Optional

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import BaseRetriever, Document
from langchain_community.vectorstores import FAISS

from app.config import settings
from app.core.llm import get_llm

logger = logging.getLogger(__name__)


class MultiQueryContextualRetriever:
    def __init__(self, store: FAISS):
        self._store = store
        self._llm = get_llm()

    def _base_retriever(self) -> BaseRetriever:
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retrieval_k},
        )

    def _multi_query_retriever(self) -> MultiQueryRetriever:
        return MultiQueryRetriever.from_llm(
            retriever=self._base_retriever(),
            llm=self._llm,
        )

    def _compression_retriever(self, base: BaseRetriever) -> ContextualCompressionRetriever:
        compressor = LLMChainExtractor.from_llm(self._llm)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base,
        )

    def retrieve(
        self,
        question: str,
        use_multi_query: bool = True,
        use_compression: bool | None = None,
    ) -> List[Document]:
        compress = settings.compression_enabled if use_compression is None else use_compression
        retriever: BaseRetriever = self._base_retriever()

        if use_multi_query:
            retriever = self._multi_query_retriever()

        if compress:
            retriever = self._compression_retriever(retriever)

        docs = retriever.get_relevant_documents(question)
        seen = set()
        unique = []
        for doc in docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique


class SimpleRetriever:
    def __init__(self, store: FAISS):
        self._store = store

    def retrieve(self, question: str, k: Optional[int] = None) -> List[Document]:
        k = k or settings.retrieval_k
        return self._store.similarity_search(question, k=k)

    def retrieve_with_scores(self, question: str, k: Optional[int] = None) -> List[tuple[Document, float]]:
        k = k or settings.retrieval_k
        return self._store.similarity_search_with_score(question, k=k)

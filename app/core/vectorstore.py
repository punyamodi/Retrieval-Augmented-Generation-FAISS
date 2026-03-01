from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from app.config import settings
from app.core.embeddings import EmbeddingsWrapper


class VectorStoreManager:
    def __init__(self):
        self._embeddings = EmbeddingsWrapper()
        self._stores: Dict[str, FAISS] = {}
        self._meta_path = settings.indexes_dir / "metadata.json"
        self._metadata: Dict[str, dict] = self._load_metadata()

    def _load_metadata(self) -> Dict[str, dict]:
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                return json.load(f)
        return {}

    def _save_metadata(self) -> None:
        with open(self._meta_path, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def _index_path(self, doc_id: str) -> Path:
        return settings.indexes_dir / doc_id

    def add_documents(self, doc_id: str, chunks: List[Document], doc_meta: dict) -> int:
        index_path = self._index_path(doc_id)
        if index_path.exists():
            store = FAISS.load_local(
                str(index_path),
                self._embeddings.as_langchain(),
                allow_dangerous_deserialization=True,
            )
            store.add_documents(chunks)
        else:
            store = FAISS.from_documents(chunks, self._embeddings.as_langchain())

        store.save_local(str(index_path))
        self._stores[doc_id] = store
        self._metadata[doc_id] = doc_meta
        self._save_metadata()
        return len(chunks)

    def get_store(self, doc_id: str) -> Optional[FAISS]:
        if doc_id in self._stores:
            return self._stores[doc_id]
        index_path = self._index_path(doc_id)
        if index_path.exists():
            store = FAISS.load_local(
                str(index_path),
                self._embeddings.as_langchain(),
                allow_dangerous_deserialization=True,
            )
            self._stores[doc_id] = store
            return store
        return None

    def get_merged_store(self, doc_ids: Optional[List[str]] = None) -> Optional[FAISS]:
        ids = doc_ids or list(self._metadata.keys())
        stores = [self.get_store(did) for did in ids if self.get_store(did) is not None]
        if not stores:
            return None
        merged = stores[0]
        for store in stores[1:]:
            merged.merge_from(store)
        return merged

    def delete_document(self, doc_id: str) -> bool:
        index_path = self._index_path(doc_id)
        if index_path.exists():
            shutil.rmtree(index_path)
        self._stores.pop(doc_id, None)
        existed = doc_id in self._metadata
        self._metadata.pop(doc_id, None)
        self._save_metadata()
        return existed

    def list_documents(self) -> List[dict]:
        return [{"doc_id": k, **v} for k, v in self._metadata.items()]

    def document_exists(self, doc_id: str) -> bool:
        return doc_id in self._metadata

    def get_document_meta(self, doc_id: str) -> Optional[dict]:
        return self._metadata.get(doc_id)

    def similarity_search(
        self,
        query: str,
        k: int = 6,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Document]:
        store = self.get_merged_store(doc_ids)
        if store is None:
            return []
        return store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 6,
        doc_ids: Optional[List[str]] = None,
    ) -> List[tuple[Document, float]]:
        store = self.get_merged_store(doc_ids)
        if store is None:
            return []
        return store.similarity_search_with_score(query, k=k)

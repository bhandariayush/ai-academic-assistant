from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from nltk.tokenize import sent_tokenize  # sentence-aware splitter [web:322]

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = INDEX_DIR / "faiss.index"
META_FILE = INDEX_DIR / "metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim


@dataclass
class ChunkMeta:
    chunk_id: int
    doc_id: str
    text: str


class RAGStore:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index: faiss.IndexFlatL2 | None = None
        self.metadata: Dict[int, ChunkMeta] = {}
        self.next_chunk_id: int = 0

        self._load_or_init_index()

    # ---------- index persistence ----------

    def _load_or_init_index(self) -> None:
        if INDEX_FILE.exists() and META_FILE.exists():
            self.index = faiss.read_index(str(INDEX_FILE))  # [web:360]
            with open(META_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)

            self.metadata = {
                int(k): ChunkMeta(
                    chunk_id=int(k),
                    doc_id=v["doc_id"],
                    text=v["text"],
                )
                for k, v in raw.items()
            }
            self.next_chunk_id = max(self.metadata.keys()) + 1 if self.metadata else 0
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.metadata = {}
            self.next_chunk_id = 0
            self._save()

    def _save(self) -> None:
        faiss.write_index(self.index, str(INDEX_FILE))  # [web:360]
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    str(cid): {"doc_id": meta.doc_id, "text": meta.text}
                    for cid, meta in self.metadata.items()
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    # ---------- sentence-aware text splitter ----------

    @staticmethod
    def split_text_sentence_aware(
        text: str,
        max_chars: int = 1200,
        overlap_sents: int = 2,
    ) -> List[str]:
        """
        Sentence-aware chunking:
        - Overlaps last N sentences between consecutive chunks.
        """
        sents = sent_tokenize(text)  # [web:322]
        chunks: List[str] = []

        cur_sents: List[str] = []
        cur_len = 0

        for s in sents:
            s = s.strip()
            if not s:
                continue

            # if adding sentence would exceed max_chars -> flush current chunk
            if cur_sents and (cur_len + len(s) + 1 > max_chars):
                chunks.append(" ".join(cur_sents).strip())

                # overlap last overlap_sents sentences
                if overlap_sents > 0:
                    cur_sents = cur_sents[-overlap_sents:]
                    cur_len = len(" ".join(cur_sents))
                else:
                    cur_sents = []
                    cur_len = 0

            cur_sents.append(s)
            cur_len += len(s) + 1

        if cur_sents:
            chunks.append(" ".join(cur_sents).strip())

        return chunks

    # ---------- ingest API ----------

    def ingest_text(self, doc_id: str, text: str) -> int:
        """
        Add a new document's text to the vector store.
        Returns number of chunks added.
        """
        chunks = self.split_text_sentence_aware(text, max_chars=1200, overlap_sents=2)
        if not chunks:
            return 0

        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        embeddings = embeddings.astype("float32")

        # assign global chunk IDs
        num_new = embeddings.shape[0]
        ids = np.arange(self.next_chunk_id, self.next_chunk_id + num_new)
        self.next_chunk_id += num_new

        # add to FAISS index
        self.index.add(embeddings)

        # update metadata
        for cid, chunk_text in zip(ids, chunks):
            self.metadata[int(cid)] = ChunkMeta(
                chunk_id=int(cid),
                doc_id=doc_id,
                text=chunk_text,
            )

        self._save()
        return int(num_new)

    # ---------- query API ----------

    def query(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        q_emb = self.model.encode([question], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(q_emb, k)

        results: List[Dict[str, Any]] = []
        for dist, cid in zip(D[0], I[0]):
            if cid == -1:
                continue
            meta = self.metadata[int(cid)]
            results.append(
                {
                    "chunk_id": int(cid),
                    "doc_id": meta.doc_id,
                    "score": float(dist),
                    "text": meta.text,
                }
            )
        return results
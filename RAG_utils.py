import os
import re
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from preprocessing import *
from llm_utils import *



# ----------------------------
# 3) Chunking
# ----------------------------
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Fixed-length character chunks with overlap.
    Overlap helps avoid splitting definitions across boundaries.
    """
    text = clean_text(text)
    if not text:
        return []
    out = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        out.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return out

@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str

def build_chunks(docs: List[Dict], chunk_size=900, overlap=150) -> List[Chunk]:
    chunks = []
    for d in docs:
        parts = chunk_text(d["text"], chunk_size=chunk_size, overlap=overlap)
        for i, p in enumerate(parts):
            chunks.append(Chunk(d["doc_id"], i, p))
    if not chunks:
        raise ValueError("No chunks produced. Check PDF extraction.")
    return chunks


# ----------------------------
# 4) Embeddings + FAISS retrieval
# ----------------------------
def embed_texts(embed_model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        e = embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embs.append(e)
    embs = np.vstack(embs).astype("float32")
    return l2_normalize(embs)

def build_index(embs: np.ndarray) -> faiss.Index:
    """
    IndexFlatIP + normalized vectors => cosine similarity search.
    """
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index

def retrieve(embed_model, index, chunks, question: str, top_k: int = 6) -> List[Tuple[Chunk, float]]:
    q = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    q = l2_normalize(q)
    scores, ids = index.search(q, top_k)
    out = []
    for cid, sc in zip(ids[0], scores[0]):
        if cid >= 0:
            out.append((chunks[cid], float(sc)))
    return out

def should_refuse(retrieved: List[Tuple[Chunk, float]], min_score: float = 0.25) -> bool:
    """If retrieval is weak, don't guess."""
    return (not retrieved) or (retrieved[0][1] < min_score)


# ----------------------------
# 5) Prompt
# ----------------------------
def build_prompt(question: str, retrieved: List[Tuple[Chunk, float]], max_context_chars: int = 6000):
    """
    Very strict prompt (helps small models):
    - answer only from context
    - cite sources (doc_id, chunk_id)
    """
    blocks, used = [], 0
    for c, sc in retrieved:
        b = f"[SOURCE: {c.doc_id} | chunk:{c.chunk_id} | score:{sc:.3f}]\n{c.text}\n"
        if used + len(b) > max_context_chars:
            break
        blocks.append(b)
        used += len(b)

    context = "\n".join(blocks).strip()

    system = (
        "You are a careful assistant. Answer strictly from CONTEXT.\n"
        "If the answer is not in the context, say: \"I don't know based on the provided documents.\".\n"
        "Cite sources like (doc_id, chunk_id)."
    )
    user = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer with citations."
    return system, user


# ----------------------------
# 7) Build RAG once
# ----------------------------
def build_rag(docs_dir: str,
              embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
              chunk_size: int = 900,
              overlap: int = 150):
    t = Timer()

    t.mark("load")
    docs = load_documents(docs_dir)
    t.mark("load_done")

    t.mark("chunk")
    chunks = build_chunks(docs, chunk_size=chunk_size, overlap=overlap)
    t.mark("chunk_done")

    t.mark("embed_model")
    embed_model = SentenceTransformer(embed_model_name)
    t.mark("embed_model_done")

    t.mark("embed")
    embs = embed_texts(embed_model, [c.text for c in chunks], batch_size=64)
    t.mark("embed_done")

    t.mark("index")
    index = build_index(embs)
    t.mark("index_done")

    print(f"[build] load={t.dt('load','load_done'):.2f}s | chunk={t.dt('chunk','chunk_done'):.2f}s | "
          f"embed={t.dt('embed','embed_done'):.2f}s | index={t.dt('index','index_done'):.2f}s")
    print("Docs:", len(docs), "| Chunks:", len(chunks), "| EmbDim:", embs.shape[1])

    return embed_model, index, chunks


def rag_answer(embed_model, index, chunks, llm: HFLocalLLM,
               question: str, k: int = 6, min_score: float = 0.25):
    """
    retrieve -> prompt -> generate
    Returns answer + retrieved chunks + timing + tokens
    """
    t = Timer()

    t.mark("retrieve")
    retrieved = retrieve(embed_model, index, chunks, question, top_k=k)
    t.mark("retrieve_done")

    retrieved_rows = [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": sc, "text": c.text} for (c, sc) in retrieved]

    if should_refuse(retrieved, min_score=min_score):
        return {
            "answer": "I don't know based on the provided documents.",
            "retrieved": retrieved_rows,
            "timing": {"retrieve_s": t.dt("retrieve", "retrieve_done"), "llm_s": 0.0, "total_s": t.dt("retrieve", "retrieve_done")},
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }

    system, user = build_prompt(question, retrieved)

    t.mark("llm")
    out = llm.generate(system, user)
    t.mark("llm_done")

    return {
        "answer": out["text"],
        "retrieved": retrieved_rows,
        "timing": {
            "retrieve_s": t.dt("retrieve", "retrieve_done"),
            "llm_s": t.dt("llm", "llm_done"),
            "total_s": t.dt("retrieve", "retrieve_done") + t.dt("llm", "llm_done"),
        },
        "usage": out.get("usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
    }



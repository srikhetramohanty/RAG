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


# ----------------------------
# 1) Small utilities
# ----------------------------
class Timer:
    """Tiny stage timer."""
    def __init__(self): self.t = {}
    def mark(self, k: str): self.t[k] = time.perf_counter()
    def dt(self, a: str, b: str) -> float: return self.t[b] - self.t[a]

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def _contains_all(text: str, phrases: List[str]) -> bool:
    t = _norm(text)
    return all(_norm(p) in t for p in phrases)


# ----------------------------
# 2) Load docs (PDF/TXT)
# ----------------------------
def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return clean_text(f.read())

def load_pdf_text(path: str) -> str:
    """
    Simple PDF text extraction.
    Works for text-based PDFs. (No OCR here.)
    """
    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    return clean_text("\n".join(pages))

def load_documents(docs_dir: str) -> List[Dict]:
    """
    Returns: [{"doc_id": filename, "text": text}, ...]
    """
    docs = []
    for fname in os.listdir(docs_dir):
        path = os.path.join(docs_dir, fname)
        if os.path.isdir(path):
            continue
        ext = os.path.splitext(fname)[1].lower()

        try:
            if ext == ".pdf":
                text = load_pdf_text(path)
            elif ext == ".txt":
                text = load_txt(path)
            else:
                continue
        except Exception as e:
            print("SKIP:", fname, "|", repr(e))
            continue

        if not text:
            print("WARN empty:", fname)
            continue

        docs.append({"doc_id": fname, "text": text})

    if not docs:
        raise ValueError(f"No documents loaded from {docs_dir}. Check path/extensions.")
    return docs

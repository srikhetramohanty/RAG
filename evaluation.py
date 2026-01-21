
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
from RAG_utils import *
from llm_utils import *

# ----------------------------
# 8) Gold evaluation set (10 Qs) — writes qa_set.jsonl
# ----------------------------
def write_gold_set(path: str):
    """
    Gold set uses:
      - evidence: doc_id + must_contain_phrases (for retrieval_hit)
      - required_answer_phrases (for phrase recall)
      - gold_answer (for eyeballing / optional exact match)
    """
    gold = [
        {
          "id":"q1_scope_bsa",
          "question":"Where does the Bharatiya Sakshya Adhiniyam, 2023 apply, and what does it exclude?",
          "evidence":[{"doc_id":"250882_english_01042024_0.pdf","must_contain_phrases":["judicial proceedings","not to affidavits","arbitrator"]}],
          "required_answer_phrases":["judicial proceedings","courts-martial","not to affidavits","arbitrator"],
          "gold_answer":"It applies to all judicial proceedings in or before any Court (including courts-martial), but not to affidavits and not to proceedings before an arbitrator."
        },
        {
          "id":"q2_commencement_bsa",
          "question":"How does the Bharatiya Sakshya Adhiniyam, 2023 come into force?",
          "evidence":[{"doc_id":"250882_english_01042024_0.pdf","must_contain_phrases":["Central Government","notification","Official Gazette"]}],
          "required_answer_phrases":["Central Government","notification","Official Gazette"],
          "gold_answer":"It comes into force on a date appointed by the Central Government by notification in the Official Gazette."
        },
        {
          "id":"q3_document_definition",
          "question":"How is a “document” defined, and does it include electronic/digital records? Give examples.",
          "evidence":[{"doc_id":"250882_english_01042024_0.pdf","must_contain_phrases":["document","electronic","digital","emails"]}],
          "required_answer_phrases":["includes electronic","digital records","emails","server logs"],
          "gold_answer":"A document includes electronic/digital records; examples include emails and server logs (and other digital records on devices)."
        },
        {
          "id":"q4_evidence_definition",
          "question":"What is “evidence” under the Bharatiya Sakshya Adhiniyam, 2023?",
          "evidence":[{"doc_id":"250882_english_01042024_0.pdf","must_contain_phrases":["evidence","oral evidence","documentary evidence"]}],
          "required_answer_phrases":["oral evidence","documentary evidence","electronic","digital"],
          "gold_answer":"Evidence includes oral evidence (statements permitted/required from witnesses, including electronic statements) and documentary evidence (documents including electronic/digital records)."
        },
        {
          "id":"q5_conclusive_proof",
          "question":"What does “conclusive proof” mean in the Bharatiya Sakshya Adhiniyam, 2023?",
          "evidence":[{"doc_id":"250882_english_01042024_0.pdf","must_contain_phrases":["conclusive proof","shall regard","shall not allow evidence"]}],
          "required_answer_phrases":["shall regard","shall not allow evidence"],
          "gold_answer":"If one fact is declared conclusive proof of another, once the first is proved the Court must regard the second as proved and cannot allow evidence to disprove it."
        },
        {
          "id":"q6_bnss_audio_video_electronic",
          "question":"Under BNSS 2023, what does “audio-video electronic” mean?",
          "evidence":[{"doc_id":"Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf","must_contain_phrases":["audio-video electronic","video conferencing","recording"]}],
          "required_answer_phrases":["video conferencing","recording","search","seizure"],
          "gold_answer":"It includes using communication devices for video conferencing and for recording processes like identification, search and seizure, or evidence, as provided by rules."
        },
        {
          "id":"q7_bnss_electronic_communication",
          "question":"Under BNSS 2023, what is “electronic communication”?",
          "evidence":[{"doc_id":"Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf","must_contain_phrases":["electronic communication","written","verbal","pictorial","video"]}],
          "required_answer_phrases":["written","verbal","pictorial","video"],
          "gold_answer":"Electronic communication is transmission of written, verbal, pictorial or video content between persons/devices via electronic devices."
        },
        {
          "id":"q8_bnss_bailable_nonbailable",
          "question":"Define “bailable offence” and “non-bailable offence” under BNSS 2023.",
          "evidence":[{"doc_id":"Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf","must_contain_phrases":["bailable offence","First Schedule","non-bailable offence"]}],
          "required_answer_phrases":["First Schedule","bailable","non-bailable"],
          "gold_answer":"Bailable offence: shown as bailable in the First Schedule or made bailable by other law. Non-bailable offence: any other offence."
        },
        {
          "id":"q9_goi1935_lists_counts",
          "question":"As per the notes, what were the three subject lists under the Government of India Act 1935 and how many items did each have?",
          "evidence":[{"doc_id":"UNIT III.pdf","must_contain_phrases":["Federal List","59","Provincial List","54","Concurrent List","36"]}],
          "required_answer_phrases":["Federal List","59","Provincial List","54","Concurrent List","36"],
          "gold_answer":"Federal List (59), Provincial List (54), Concurrent List (36)."
        },
        {
          "id":"q10_goi1935_provincial_autonomy",
          "question":"According to the notes, what did “Provincial Autonomy” mean under the Government of India Act 1935, and what timeline is mentioned?",
          "evidence":[{"doc_id":"UNIT III.pdf","must_contain_phrases":["Provincial Autonomy","diarchy","1937","1939"]}],
          "required_answer_phrases":["diarchy","1937","1939"],
          "gold_answer":"Provincial autonomy abolished diarchy in provinces; governors acted with advice of responsible ministers. Notes mention it came into effect in 1937 and was discontinued in 1939."
        },
    ]

    with open(path, "w", encoding="utf-8") as f:
        for ex in gold:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print("Wrote:", path)


#####################

# ----------------------------
# 9) Gold evaluation (objective scoring)
# ----------------------------
def answer_phrase_recall(answer: str, required_phrases: List[str]) -> float:
    """Fraction of required phrases present in answer (0..1)."""
    if not required_phrases:
        return 1.0
    a = _norm(answer)
    hit = sum(1 for p in required_phrases if _norm(p) in a)
    return hit / len(required_phrases)

def answer_exact_match(answer: str, gold_answer: str) -> int:
    """Exact match is harsh; mostly useful for short answers."""
    return 1 if _norm(answer) == _norm(gold_answer) else 0

def retrieval_hit(retrieved_rows: List[Dict], evidence_spec: List[Dict], k: int) -> int:
    """
    Hit if within top-k:
      doc_id matches AND chunk text contains all must_contain_phrases
    """
    top = retrieved_rows[:k]
    for ev in evidence_spec:
        ev_doc = ev["doc_id"]
        must = ev.get("must_contain_phrases", [])
        for r in top:
            if r.get("doc_id") == ev_doc and _contains_all(r.get("text",""), must):
                return 1
    return 0

def load_qa_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def run_gold_eval(run_name: str, qa_jsonl_path: str,
                  embed_model, index, chunks, llm,
                  k: int, min_score: float = 0.25) -> pd.DataFrame:
    """
    Runs all gold questions and returns a per-question DataFrame.
    """
    qa = load_qa_jsonl(qa_jsonl_path)
    rows = []

    for ex in qa:
        qid = ex["id"]
        q = ex["question"]
        gold = ex.get("gold_answer", "")
        evidence = ex.get("evidence", [])
        req_phrases = ex.get("required_answer_phrases", [])

        out = rag_answer(embed_model, index, chunks, llm, q, k=k, min_score=min_score)

        hit = retrieval_hit(out["retrieved"], evidence, k=k)
        phr = answer_phrase_recall(out["answer"], req_phrases)
        em = answer_exact_match(out["answer"], gold) if gold else None

        top1_doc = out["retrieved"][0]["doc_id"] if out["retrieved"] else None
        top1_score = out["retrieved"][0]["score"] if out["retrieved"] else None

        rows.append({
            "run_name": run_name,
            "qid": qid,
            "question": q,
            "k": k,

            # objective metrics
            "retrieval_hit": hit,                  # 0/1
            "answer_phrase_recall": phr,           # 0..1
            "answer_exact_match": em,              # 0/1 (optional)

            # measured
            "latency_s": out["timing"]["total_s"],
            "tokens_total": out["usage"]["total_tokens"],

            # helpful debug
            "top1_doc": top1_doc,
            "top1_score": top1_score,
            "pred_answer": out["answer"],
            "gold_answer": gold,
        })

    return pd.DataFrame(rows)

def summarize_gold_eval(df: pd.DataFrame) -> pd.DataFrame:
    """One-row summary for quick comparisons."""
    return pd.DataFrame([{
        "run_name": df["run_name"].iloc[0],
        "k": int(df["k"].iloc[0]),
        "RetrievalHit@k": float(df["retrieval_hit"].mean()),
        "AnswerPhraseRecall": float(df["answer_phrase_recall"].mean()),
        "ExactMatchRate": float(df["answer_exact_match"].dropna().mean()) if df["answer_exact_match"].notna().any() else None,
        "AvgLatency_s": float(df["latency_s"].mean()),
        "AvgTokensTotal": float(df["tokens_total"].mean()),
    }])



# ============================================================
# 11) ADDITION 1: Precision metric (retrieval precision@k)
# ============================================================

def retrieval_precision_at_k(retrieved_rows: List[Dict], evidence_spec: List[Dict], k: int) -> float:
    """
    Precision@k (retrieval):
      Among top-k retrieved chunks, how many are "relevant" (match gold evidence spec)?

    Here, a chunk is considered relevant if:
      - doc_id matches a gold evidence doc_id AND
      - it contains all gold must_contain_phrases for that evidence item

    precision@k = (# relevant chunks in top-k) / k
    """
    top = retrieved_rows[:k]
    if k <= 0:
        return 0.0
    rel = 0
    for r in top:
        for ev in evidence_spec:
            if r.get("doc_id") == ev["doc_id"] and _contains_all(r.get("text",""), ev.get("must_contain_phrases", [])):
                rel += 1
                break
    return rel / k


def run_gold_eval_with_precision(run_name: str, qa_jsonl_path: str,
                                 embed_model, index, chunks, llm,
                                 k: int, min_score: float = 0.25) -> pd.DataFrame:
    """
    Same as run_gold_eval, but also adds retrieval_precision@k.
    (Does not change existing code; this is an added version.)
    """
    qa = load_qa_jsonl(qa_jsonl_path)
    rows = []

    for ex in qa:
        qid = ex["id"]
        q = ex["question"]
        gold = ex.get("gold_answer", "")
        evidence = ex.get("evidence", [])
        req_phrases = ex.get("required_answer_phrases", [])

        out = rag_answer(embed_model, index, chunks, llm, q, k=k, min_score=min_score)

        hit = retrieval_hit(out["retrieved"], evidence, k=k)
        prec = retrieval_precision_at_k(out["retrieved"], evidence, k=k)
        phr = answer_phrase_recall(out["answer"], req_phrases)
        em = answer_exact_match(out["answer"], gold) if gold else None

        top1_doc = out["retrieved"][0]["doc_id"] if out["retrieved"] else None
        top1_score = out["retrieved"][0]["score"] if out["retrieved"] else None

        rows.append({
            "run_name": run_name,
            "qid": qid,
            "question": q,
            "k": k,

            # objective metrics
            "retrieval_hit": hit,                  # 0/1
            "retrieval_precision@k": prec,         # 0..1
            "answer_phrase_recall": phr,           # 0..1
            "answer_exact_match": em,              # 0/1 (optional)

            # measured
            "latency_s": out["timing"]["total_s"],
            "tokens_total": out["usage"]["total_tokens"],

            # helpful debug
            "top1_doc": top1_doc,
            "top1_score": top1_score,
            "pred_answer": out["answer"],
            "gold_answer": gold,
        })

    return pd.DataFrame(rows)


def summarize_gold_eval_with_precision(df: pd.DataFrame) -> pd.DataFrame:
    """One-row summary with Precision@k added."""
    return pd.DataFrame([{
        "run_name": df["run_name"].iloc[0],
        "k": int(df["k"].iloc[0]),
        "RetrievalHit@k": float(df["retrieval_hit"].mean()),
        "RetrievalPrecision@k": float(df["retrieval_precision@k"].mean()),
        "AnswerPhraseRecall": float(df["answer_phrase_recall"].mean()),
        "ExactMatchRate": float(df["answer_exact_match"].dropna().mean()) if df["answer_exact_match"].notna().any() else None,
        "AvgLatency_s": float(df["latency_s"].mean()),
        "AvgTokensTotal": float(df["tokens_total"].mean()),
    }])


# Example usage (single run with precision):
# df_results_p = run_gold_eval_with_precision(run_name, QA_PATH, embed_model, index, chunks, llm, k=K, min_score=0.25)
# df_summary_p = summarize_gold_eval_with_precision(df_results_p)
# df_summary_p


# ============================================================
# 12) ADDITION 2: Grid run over multiple configs (LLM, embed, k, retriever_type)
# ============================================================

def run_grid_configs(
    docs_dir: str,
    qa_path: str,
    llm_models: List[str],
    embed_models: List[str],
    k_values: List[int],
    retriever_types: List[str],
    use_4bit: bool = True,
    min_score: float = 0.25,
    chunk_size: int = 900,
    overlap: int = 150,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loops over config grid and returns:
      - df_all_summaries: one row per run config with metrics
      - df_all_details: concatenated per-question results for all runs (optional but useful)

    Note:
      retriever_types included for your experiment tracking.
      In this baseline code, only 'dense' is implemented.
    """

    # Cache RAG indexes per (embed_model, retriever_type) to avoid rebuilding repeatedly
    rag_cache = {}  # key -> (embed_model_obj, index, chunks)

    all_summaries = []
    all_details = []

    for embed_name in embed_models:
        for rtype in retriever_types:
            if rtype != "dense":
                print(f"SKIP retriever_type={rtype} (not implemented in this baseline)")
                continue

            cache_key = (embed_name, rtype)
            if cache_key not in rag_cache:
                em, ix, ch = build_rag(
                    docs_dir=docs_dir,
                    embed_model_name=embed_name,
                    chunk_size=chunk_size,
                    overlap=overlap
                )
                rag_cache[cache_key] = (em, ix, ch)

    for llm_name in llm_models:
        print(f"----------------- {llm_name} --------------------")
        llm = HFLocalLLM(llm_name, use_4bit=use_4bit)

        for embed_name in embed_models:
            for rtype in retriever_types:
                if rtype != "dense":
                    continue

                em, ix, ch = rag_cache[(embed_name, rtype)]

                for k in k_values:
                    run_name = f"{llm_name} | {embed_name} | {rtype} | k={k}"

                    df_run = run_gold_eval_with_precision(
                        run_name=run_name,
                        qa_jsonl_path=qa_path,
                        embed_model=em,
                        index=ix,
                        chunks=ch,
                        llm=llm,
                        k=k,
                        min_score=min_score
                    )

                    df_sum = summarize_gold_eval_with_precision(df_run)
                    df_sum.insert(1, "llm_model", llm_name)
                    df_sum.insert(2, "embed_model", embed_name)
                    df_sum.insert(3, "retriever_type", rtype)

                    all_summaries.append(df_sum)
                    all_details.append(df_run)

    df_all_summaries = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    df_all_details = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()

    # Sort by the most important objective metrics first
    if not df_all_summaries.empty:
        df_all_summaries = df_all_summaries.sort_values(
            ["RetrievalHit@k", "AnswerPhraseRecall", "RetrievalPrecision@k"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    return df_all_summaries, df_all_details





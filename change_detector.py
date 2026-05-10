from dotenv import load_dotenv
load_dotenv()

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_MATCH_THRESHOLD = 0.85
_UNCERTAIN_THRESHOLD = 0.50

_STANDARD_TEMPLATE = (
    "You are analyzing year-over-year changes in SEC 10-K risk factor disclosures.\n\n"
    "Risk factor from {year_a}:\n{summary_a}\n\n"
    "Risk factor from {year_b}:\n{summary_b}\n\n"
    "Classify the change as exactly one of: ESCALATED, DE-ESCALATED, UNCHANGED.\n"
    "- ESCALATED: the risk grew more severe, broader, or more prominently disclosed\n"
    "- DE-ESCALATED: the risk shrank, was mitigated, or received less emphasis\n"
    "- UNCHANGED: the risk language and severity are essentially the same\n\n"
    "Respond with only the label."
)

_COT_TEMPLATE = (
    "You are analyzing year-over-year changes in SEC 10-K risk factor disclosures.\n\n"
    "Risk factor from {year_a}:\n{summary_a}\n\n"
    "Risk factor from {year_b}:\n{summary_b}\n\n"
    "Think step by step:\n"
    "1. What is the core risk described in {year_a}?\n"
    "2. What changed in {year_b}? Is the severity higher, lower, or the same?\n"
    "3. Are there new dimensions to the risk, or have existing concerns been addressed?\n\n"
    "After reasoning, output your final answer on a new line in this exact format:\n"
    "LABEL: <ESCALATED|DE-ESCALATED|UNCHANGED>"
)


def _get_year_embeddings(ticker: str, year: int, vectorstore: Chroma) -> dict:
    results = vectorstore.get(
        where={"$and": [{"ticker": ticker}, {"year": year}]},
        include=["embeddings", "metadatas"],
    )
    if not results["documents"] and not results.get("metadatas"):
        return {}
    return {
        meta["chunk_id"]: {
            "metadata": meta,
            "embedding": np.array(emb),
        }
        for meta, emb in zip(results["metadatas"], results["embeddings"])
    }


def _classify_standard(summary_a: str, summary_b: str, year_a: int, year_b: int) -> str:
    prompt = _STANDARD_TEMPLATE.format(
        year_a=year_a, summary_a=summary_a, year_b=year_b, summary_b=summary_b
    )
    response = _LLM.invoke(prompt).content.strip().upper()
    for label in ("ESCALATED", "DE-ESCALATED", "UNCHANGED"):
        if label in response:
            return label
    return "UNCHANGED"


def _classify_cot(summary_a: str, summary_b: str, year_a: int, year_b: int) -> tuple[str, str]:
    prompt = _COT_TEMPLATE.format(
        year_a=year_a, summary_a=summary_a, year_b=year_b, summary_b=summary_b
    )
    reasoning = _LLM.invoke(prompt).content.strip()
    match = re.search(r"LABEL:\s*(ESCALATED|DE-ESCALATED|UNCHANGED)", reasoning, re.IGNORECASE)
    label = match.group(1).upper() if match else "UNCHANGED"
    return label, reasoning


def detect_changes(
    ticker: str,
    year_a: int,
    year_b: int,
    vectorstore: Chroma,
    summaries_by_chunk_id: dict[str, dict],
) -> list[dict]:
    data_a = _get_year_embeddings(ticker, year_a, vectorstore)
    data_b = _get_year_embeddings(ticker, year_b, vectorstore)

    if not data_a or not data_b:
        print(f"  No embeddings found for {ticker} {year_a} or {year_b}, skipping.")
        return []

    ids_a = list(data_a.keys())
    ids_b = list(data_b.keys())
    embs_a = np.stack([data_a[i]["embedding"] for i in ids_a])
    embs_b = np.stack([data_b[i]["embedding"] for i in ids_b])

    sim_matrix = cosine_similarity(embs_a, embs_b)

    matched_b: set[int] = set()
    results: list[dict] = []

    total_pairs = len(ids_a)
    print(f"  Matching {len(ids_a)} chunks from {year_a} → {len(ids_b)} chunks in {year_b}...")

    for i, chunk_id_a in enumerate(ids_a):
        best_j = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_j])
        chunk_id_b = ids_b[best_j]

        summary_a = summaries_by_chunk_id.get(chunk_id_a, {}).get("zero_shot_summary", "")
        summary_b = summaries_by_chunk_id.get(chunk_id_b, {}).get("zero_shot_summary", "")

        if best_sim >= _MATCH_THRESHOLD:
            matched_b.add(best_j)
            print(f"    [{i+1}/{total_pairs}] Matched {chunk_id_a} → {chunk_id_b} (sim={best_sim:.3f})", flush=True)

            if summary_a and summary_b:
                standard_label = _classify_standard(summary_a, summary_b, year_a, year_b)
                cot_label, cot_reasoning = _classify_cot(summary_a, summary_b, year_a, year_b)
            else:
                standard_label = cot_label = "UNCHANGED"
                cot_reasoning = "(no summaries available)"

            results.append({
                "status": standard_label,
                "cot_status": cot_label,
                "cot_reasoning": cot_reasoning,
                "similarity": round(best_sim, 4),
                "chunk_id_a": chunk_id_a,
                "chunk_id_b": chunk_id_b,
                "header_a": data_a[chunk_id_a]["metadata"].get("header", ""),
                "header_b": data_b[chunk_id_b]["metadata"].get("header", ""),
                "summary_a": summary_a,
                "summary_b": summary_b,
                "year_a": year_a,
                "year_b": year_b,
            })
        elif best_sim < _UNCERTAIN_THRESHOLD:
            print(f"    [{i+1}/{total_pairs}] REMOVED {chunk_id_a} (best sim={best_sim:.3f})", flush=True)
            results.append({
                "status": "REMOVED",
                "cot_status": "REMOVED",
                "cot_reasoning": "",
                "similarity": round(best_sim, 4),
                "chunk_id_a": chunk_id_a,
                "chunk_id_b": None,
                "header_a": data_a[chunk_id_a]["metadata"].get("header", ""),
                "header_b": None,
                "summary_a": summary_a,
                "summary_b": None,
                "year_a": year_a,
                "year_b": year_b,
            })
        else:
            print(f"    [{i+1}/{total_pairs}] UNCERTAIN {chunk_id_a} (sim={best_sim:.3f}), treating as REMOVED", flush=True)
            results.append({
                "status": "REMOVED",
                "cot_status": "REMOVED",
                "cot_reasoning": "(similarity in uncertain range)",
                "similarity": round(best_sim, 4),
                "chunk_id_a": chunk_id_a,
                "chunk_id_b": None,
                "header_a": data_a[chunk_id_a]["metadata"].get("header", ""),
                "header_b": None,
                "summary_a": summary_a,
                "summary_b": None,
                "year_a": year_a,
                "year_b": year_b,
            })

    for j, chunk_id_b in enumerate(ids_b):
        if j not in matched_b:
            summary_b = summaries_by_chunk_id.get(chunk_id_b, {}).get("zero_shot_summary", "")
            results.append({
                "status": "NEW",
                "cot_status": "NEW",
                "cot_reasoning": "",
                "similarity": None,
                "chunk_id_a": None,
                "chunk_id_b": chunk_id_b,
                "header_a": None,
                "header_b": data_b[chunk_id_b]["metadata"].get("header", ""),
                "summary_a": None,
                "summary_b": summary_b,
                "year_a": year_a,
                "year_b": year_b,
            })

    return results

from dotenv import load_dotenv
load_dotenv()

import transformers
transformers.logging.set_verbosity_error()

from langchain_openai import ChatOpenAI
from bert_score import BERTScorer

_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_SCORER = BERTScorer(lang="en")

_ZERO_SHOT_TEMPLATE = (
    "Summarize the following 10-K risk factor in plain English "
    "for a non-expert investor in 1-2 sentences.\n\n{chunk}"
)

_FEW_SHOT_EXAMPLES = [
    (
        "We are subject to risks associated with fluctuations in interest rates, "
        "which may adversely affect our borrowing costs and access to capital markets.",
        "Rising interest rates could make it more expensive for the company to borrow "
        "money, which may reduce profits and limit its ability to raise funds.",
    ),
    (
        "Our business depends on the continued services of key personnel, including our "
        "Chief Executive Officer and other senior management. Competition for these "
        "individuals is intense, and we may not be able to retain or replace them.",
        "If the company loses key executives or cannot hire qualified replacements, "
        "business operations and strategy could be disrupted.",
    ),
    (
        "We face intense competition in all aspects of our business from companies that "
        "have significantly greater financial, technical, manufacturing, marketing, "
        "distribution and other resources than we do.",
        "Strong competition from larger, better-resourced companies could reduce the "
        "company's market share and put pressure on its prices and profits.",
    ),
]


def _build_few_shot_prompt(chunk: str) -> str:
    examples = "\n\n".join(
        f"Example {i + 1}:\nSOURCE: {src}\nSUMMARY: {summ}"
        for i, (src, summ) in enumerate(_FEW_SHOT_EXAMPLES)
    )
    return (
        "Summarize risk factors in plain English for a non-expert investor "
        f"in 1-2 sentences.\n\nHere are some examples:\n\n{examples}\n\n"
        f"Now summarize this:\nSOURCE: {chunk}\nSUMMARY:"
    )


def summarize(chunk_text: str, method: str = "zero_shot") -> str:
    prompt = (
        _ZERO_SHOT_TEMPLATE.format(chunk=chunk_text)
        if method == "zero_shot"
        else _build_few_shot_prompt(chunk_text)
    )
    return _LLM.invoke(prompt).content.strip()


def bertscore(summary: str, source: str) -> dict[str, float]:
    P, R, F1 = _SCORER.score([summary], [source])
    return {
        "precision": round(P.item(), 4),
        "recall": round(R.item(), 4),
        "f1": round(F1.item(), 4),
    }


def summarize_chunks(
    chunks: list[tuple[dict, str]],
    methods: tuple[str, ...] = ("zero_shot", "few_shot"),
    flag_threshold: float = 0.85,
) -> list[dict]:
    total = len(chunks)
    print(f"Summarizing {total} chunks ({len(methods)} methods each)...")

    results = []
    for i, (meta, body) in enumerate(chunks, 1):
        chunk_id = meta.get("chunk_id", f"chunk_{i}")
        print(f"  [{i}/{total}] {chunk_id}", flush=True)

        entry: dict = {
            "ticker": meta.get("ticker"),
            "year": meta.get("year"),
            "chunk_id": chunk_id,
            "category": meta.get("category"),
            "header": meta.get("header"),
            "word_count": meta.get("word_count"),
            "body": body,
        }
        for method in methods:
            summary = summarize(body, method)
            scores = bertscore(summary, body)
            entry[f"{method}_summary"] = summary
            entry[f"{method}_bertscore_f1"] = scores["f1"]
            entry[f"{method}_flagged"] = scores["f1"] < flag_threshold

        results.append(entry)

    return results

import csv
import os
import textstat


_CHANGE_FIELDS = [
    "ticker", "year_a", "year_b",
    "chunk_id_a", "chunk_id_b", "header_a", "header_b",
    "standard_label", "cot_label", "similarity", "true_label",
]

_SKIP_FIELDS = {"body", "header"}


def _build_summary_rows(results: list[dict]) -> tuple[list[str], list[dict]]:
    """Enrich results with Flesch scores and return (fieldnames, rows)."""
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in _SKIP_FIELDS}
        row["source_flesch"] = round(textstat.flesch_reading_ease(r.get("body", "")), 2)
        # Add Flesch for every *_summary column present
        for key in list(r.keys()):
            if key.endswith("_summary"):
                prefix = key[: -len("_summary")]
                row[f"{prefix}_flesch"] = round(textstat.flesch_reading_ease(r.get(key, "")), 2)
        rows.append(row)

    # Collect ordered fieldnames (preserving insertion order across all rows)
    seen: set[str] = set()
    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    return fieldnames, rows


def log_summaries(results: list[dict], path: str = "summarization_results.csv") -> None:
    fieldnames, rows = _build_summary_rows(results)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def log_changes(
    ticker: str,
    changes_by_pair: dict[tuple[int, int], list[dict]],
    path: str = "change_detection_results.csv",
) -> None:
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CHANGE_FIELDS)
        if write_header:
            writer.writeheader()
        for (year_a, year_b), changes in changes_by_pair.items():
            for c in changes:
                writer.writerow({
                    "ticker": ticker,
                    "year_a": year_a,
                    "year_b": year_b,
                    "chunk_id_a": c.get("chunk_id_a", ""),
                    "chunk_id_b": c.get("chunk_id_b", ""),
                    "header_a": (c.get("header_a") or "")[:300],
                    "header_b": (c.get("header_b") or "")[:300],
                    "standard_label": c.get("status", ""),
                    "cot_label": c.get("cot_status", ""),
                    "similarity": c.get("similarity", ""),
                    "true_label": "",
                })

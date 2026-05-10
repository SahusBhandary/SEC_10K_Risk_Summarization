import csv
import os
import statistics


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


_MODEL_DISPLAY = {
    "": "GPT-4o-mini",
    "llama": "Llama 3.1 8B",
    "mistral": "Mistral 7B v0.3",
}


def _detect_model_method_prefixes(rows: list[dict]) -> list[tuple[str, str]]:
    """Return ordered (prefix, display_label) for all *_bertscore_f1 columns found."""
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    fieldnames = list(rows[0].keys()) if rows else []
    for col in fieldnames:
        if not col.endswith("_bertscore_f1"):
            continue
        prefix = col[: -len("_bertscore_f1")]  # e.g. "zero_shot", "llama_few_shot"
        if prefix in seen:
            continue
        seen.add(prefix)
        # Determine model and method from prefix
        if "_" in prefix:
            parts = prefix.rsplit("_", 1)
            # Check if last two parts form a known method
            for method_suffix in ("zero_shot", "few_shot"):
                if prefix.endswith(method_suffix):
                    model_key = prefix[: -(len(method_suffix) + 1)] if prefix != method_suffix else ""
                    model_name = _MODEL_DISPLAY.get(model_key, model_key)
                    method_name = "Zero-Shot" if "zero_shot" in method_suffix else "Few-Shot"
                    result.append((prefix, f"{method_name} ({model_name})"))
                    break
        else:
            # bare prefix like "zero_shot" or "few_shot" → GPT-4o-mini
            method_name = "Zero-Shot" if prefix == "zero_shot" else "Few-Shot"
            result.append((prefix, f"{method_name} (GPT-4o-mini)"))
    return result


def print_table1(path: str = "summarization_results.csv") -> None:
    rows = _read_csv(path)
    if not rows:
        print("Table 1: summarization_results.csv not found — run the pipeline first.")
        return

    def col_mean(col: str) -> float | None:
        vals = [float(r[col]) for r in rows if r.get(col, "").strip()]
        return statistics.mean(vals) if vals else None

    prefixes = _detect_model_method_prefixes(rows)
    src_flesch = col_mean("source_flesch")

    n = len(rows)
    hdr = f"{'Method':<32} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Src Flesch':>11} {'Sum Flesch':>11}"
    print(f"\nTable 1: Summarization Quality (BERTScore + Readability)  [n={n} chunks]")
    print(hdr)
    print("-" * len(hdr))
    for prefix, label in prefixes:
        p = col_mean(f"{prefix}_bertscore_precision")
        r = col_mean(f"{prefix}_bertscore_recall")
        f = col_mean(f"{prefix}_bertscore_f1")
        fl = col_mean(f"{prefix}_flesch")
        if f is None:
            continue
        src = f"{src_flesch:>11.1f}" if src_flesch is not None else "        N/A"
        sum_fl = f"{fl:>11.1f}" if fl is not None else "        N/A"
        print(f"{label:<32} {p:>10.4f} {r:>8.4f} {f:>8.4f} {src} {sum_fl}")


def print_table2_table3(path: str = "change_detection_results.csv") -> None:
    rows = _read_csv(path)
    labeled = [r for r in rows if r.get("true_label", "").strip()]

    if not labeled:
        print(
            "\nTables 2 & 3: No ground-truth labels found.\n"
            f"  Open {path}, fill in the 'true_label' column\n"
            "  (values: NEW / ESCALATED / DE-ESCALATED / UNCHANGED / REMOVED),\n"
            "  then re-run: python evaluator.py"
        )
        return

    from sklearn.metrics import classification_report, precision_recall_fscore_support

    true = [r["true_label"].strip().upper() for r in labeled]
    std = [r["standard_label"].strip().upper() for r in labeled]
    cot = [r["cot_label"].strip().upper() for r in labeled]
    labels = sorted(set(true))

    def macro(y_pred):
        p, r, f, _ = precision_recall_fscore_support(true, y_pred, average="macro", zero_division=0)
        return p, r, f

    std_p, std_r, std_f = macro(std)
    cot_p, cot_r, cot_f = macro(cot)

    n = len(labeled)
    print(f"\nTable 2: Change Detection Accuracy (macro avg)  [n={n} labeled pairs]")
    print(f"{'Method':<20} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 48)
    print(f"{'Standard Prompt':<20} {std_p:>10.4f} {std_r:>8.4f} {std_f:>8.4f}")
    print(f"{'Chain-of-Thought':<20} {cot_p:>10.4f} {cot_r:>8.4f} {cot_f:>8.4f}")

    print(f"\nTable 3: Per-Label Breakdown (Standard Prompt)")
    print(classification_report(true, std, labels=labels, zero_division=0))
    print(f"Table 3b: Per-Label Breakdown (Chain-of-Thought)")
    print(classification_report(true, cot, labels=labels, zero_division=0))


def print_table4(path: str = "correlation_results.csv") -> None:
    rows = _read_csv(path)
    if not rows:
        print("\nTable 4: correlation_results.csv not found — run the pipeline first.")
        return

    def direction_match(net_risk, return_30d):
        if return_30d == "" or return_30d is None:
            return "N/A"
        net = float(net_risk)
        ret = float(return_30d)
        if net > 0 and ret < 0:
            return "Yes"
        if net < 0 and ret > 0:
            return "Yes"
        if net == 0:
            return "N/A"
        return "No"

    print(f"\nTable 4: Stock Price Correlation")
    header = f"  {'Company':<8} {'Year Pair':<12} {'Net Risk':>9} {'30d Return':>11} {'Direction Match?':>17}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        ret_raw = r.get("return_30d", "")
        ret_str = f"{float(ret_raw)*100:+.2f}%" if ret_raw not in ("", None) else "N/A"
        dm = direction_match(r.get("net_risk_score", 0), ret_raw)
        pair = f"{r['year_a']}→{r['year_b']}"
        print(f"  {r['ticker']:<8} {pair:<12} {int(r['net_risk_score']):>9} {ret_str:>11} {dm:>17}")


if __name__ == "__main__":
    print_table1()
    print_table2_table3()
    print_table4()

import csv
import os
import random
import statistics

_INPUT = "summarization_results.csv"
_OUTPUT = "human_eval_results.csv"


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rate(prompt: str) -> int:
    while True:
        try:
            val = int(input(prompt))
            if 1 <= val <= 5:
                return val
            print("  Enter a number from 1 to 5.")
        except ValueError:
            print("  Enter a number from 1 to 5.")


def main():
    if not os.path.exists(_INPUT):
        print(f"Error: {_INPUT} not found. Run the pipeline first.")
        return

    rows = _read_csv(_INPUT)
    sample_size = min(30, len(rows))
    sample = random.sample(rows, sample_size)

    print(f"Human Evaluation — rating {sample_size} summaries (1=poor, 5=excellent)\n")
    print("For each chunk you will rate clarity and usefulness of zero-shot and few-shot summaries.")
    print("Press Ctrl+C at any time to save partial results.\n")

    eval_rows = []
    try:
        for i, r in enumerate(sample, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{sample_size}] {r.get('chunk_id', '')} | {r.get('category', '')}")
            print(f"\nSOURCE (first 400 chars):\n  {r.get('body', r.get('zero_shot_summary', ''))[:400]}...")
            print(f"\nZERO-SHOT SUMMARY:\n  {r.get('zero_shot_summary', '')}")
            print(f"\nFEW-SHOT SUMMARY:\n  {r.get('few_shot_summary', '')}")
            print()

            zs_clarity = _rate("  Zero-Shot clarity    (1-5): ")
            zs_useful = _rate("  Zero-Shot usefulness (1-5): ")
            fs_clarity = _rate("  Few-Shot  clarity    (1-5): ")
            fs_useful = _rate("  Few-Shot  usefulness (1-5): ")

            eval_rows.append({
                "chunk_id": r.get("chunk_id", ""),
                "ticker": r.get("ticker", ""),
                "year": r.get("year", ""),
                "zero_shot_bertscore_f1": r.get("zero_shot_bertscore_f1", ""),
                "few_shot_bertscore_f1": r.get("few_shot_bertscore_f1", ""),
                "zero_shot_clarity": zs_clarity,
                "zero_shot_usefulness": zs_useful,
                "few_shot_clarity": fs_clarity,
                "few_shot_usefulness": fs_useful,
            })
    except KeyboardInterrupt:
        print("\n\nInterrupted — saving partial results.")

    if not eval_rows:
        print("No ratings recorded.")
        return

    with open(_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(eval_rows)

    zs_c = statistics.mean(r["zero_shot_clarity"] for r in eval_rows)
    zs_u = statistics.mean(r["zero_shot_usefulness"] for r in eval_rows)
    fs_c = statistics.mean(r["few_shot_clarity"] for r in eval_rows)
    fs_u = statistics.mean(r["few_shot_usefulness"] for r in eval_rows)

    print(f"\nResults saved to {_OUTPUT}")
    print(f"\n{'Method':<25} {'Clarity':>8} {'Usefulness':>11}")
    print("-" * 46)
    print(f"{'Zero-Shot (GPT-4o-mini)':<25} {zs_c:>8.2f} {zs_u:>11.2f}")
    print(f"{'Few-Shot  (GPT-4o-mini)':<25} {fs_c:>8.2f} {fs_u:>11.2f}")


if __name__ == "__main__":
    main()

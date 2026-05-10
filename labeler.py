"""
Interactive ground-truth labeler for change_detection_results.csv.

Controls:
  n  → NEW
  e  → ESCALATED
  d  → DE-ESCALATED
  u  → UNCHANGED
  r  → REMOVED
  s  → skip (leave blank, come back later)
  q  → quit and save
"""

import csv
import os
import sys

_PATH = "change_detection_results.csv"

_KEYS = {
    "n": "NEW",
    "e": "ESCALATED",
    "d": "DE-ESCALATED",
    "u": "UNCHANGED",
    "r": "REMOVED",
    "s": None,
    "q": "quit",
}

_LEGEND = "  [n]NEW  [e]ESCALATED  [d]DE-ESCALATED  [u]UNCHANGED  [r]REMOVED  [s]skip  [q]quit"


def _read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _wrap(text: str, width: int = 90, indent: str = "    ") -> str:
    words = text.split()
    lines, current = [], []
    for w in words:
        if sum(len(x) + 1 for x in current) + len(w) > width:
            lines.append(indent + " ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        lines.append(indent + " ".join(current))
    return "\n".join(lines)


def _get_key() -> str:
    if sys.platform == "win32":
        import msvcrt
        return msvcrt.getwch().lower()
    else:
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1).lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    if not os.path.exists(_PATH):
        print(f"Error: {_PATH} not found. Run the pipeline first.")
        return

    rows = _read_rows(_PATH)
    unlabeled = [i for i, r in enumerate(rows) if not r.get("true_label", "").strip()]
    total = len(rows)
    labeled_before = total - len(unlabeled)

    if not unlabeled:
        print(f"All {total} rows are already labeled. Run: python evaluator.py")
        return

    print(f"Labeling {len(unlabeled)} unlabeled rows ({labeled_before} already done).")
    print(_LEGEND)

    labeled_now = 0
    try:
        for pos, idx in enumerate(unlabeled, 1):
            r = rows[idx]
            year_a, year_b = r.get("year_a", "?"), r.get("year_b", "?")
            ticker = r.get("ticker", "")

            print(f"\n{'='*70}")
            print(f"  [{pos}/{len(unlabeled)}]  {ticker} {year_a} → {year_b}")
            print(f"  Model says: standard={r.get('standard_label','')}  cot={r.get('cot_label','')}  sim={r.get('similarity','N/A')}")

            header_a = r.get("header_a", "").strip()
            header_b = r.get("header_b", "").strip()
            summary_a = r.get("summary_a", r.get("zero_shot_summary_a", "")).strip()
            summary_b = r.get("summary_b", r.get("zero_shot_summary_b", "")).strip()

            if header_a or summary_a:
                print(f"\n  {year_a} HEADER:  {header_a[:120]}")
                if summary_a:
                    print(f"  {year_a} SUMMARY:\n{_wrap(summary_a)}")
            else:
                print(f"\n  (no {year_a} data — this may be a NEW risk)")

            if header_b or summary_b:
                print(f"\n  {year_b} HEADER:  {header_b[:120]}")
                if summary_b:
                    print(f"  {year_b} SUMMARY:\n{_wrap(summary_b)}")
            else:
                print(f"\n  (no {year_b} data — this may be a REMOVED risk)")

            print(f"\n{_LEGEND}")
            print("  Your label: ", end="", flush=True)

            while True:
                key = _get_key()
                if key in _KEYS:
                    break

            action = _KEYS[key]
            if action == "quit":
                print("q  (quit)")
                break
            elif action is None:
                print("s  (skipped)")
                continue
            else:
                print(action)
                rows[idx]["true_label"] = action
                _write_rows(rows, _PATH)
                labeled_now += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted — progress saved.")

    total_labeled = labeled_before + labeled_now
    print(f"\nDone. Labeled {labeled_now} new rows this session ({total_labeled}/{total} total).")
    if total_labeled > 0:
        print("Run: python evaluator.py")


if __name__ == "__main__":
    main()

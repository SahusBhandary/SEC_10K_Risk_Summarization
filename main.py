from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from api import get_risk_factors, get_filing_date
from vector import store_filing
from summarizer import summarize_chunks
from change_detector import detect_changes
from stock_correlation import analyze_stock_correlation
from results_logger import log_summaries, log_changes
from evaluator import print_table1

# Set to ("llama", "mistral") once Ollama is running with those models pulled.
# Set to () to use only GPT-4o-mini.
EXTRA_MODELS: tuple[str, ...] = ()

VECTORSTORE = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(),
)

def get_cached_chunks(ticker: str, year: int):
    results = VECTORSTORE.get(where={"$and": [{"ticker": ticker}, {"year": year}]})
    if not results["documents"]:
        return None
    return list(zip(results["metadatas"], results["documents"]))

def main():

    ticker = ""
    year_range = ""

    # Validate Args
    while True:
        # Input ticker and year range
        ticker = input("Please enter a ticker (ex: AAPL): ")
        year_range = input("Please enter a year range (ex: 2020-2024) [The earliest year is 2001 and the latest year is 2024]: ")

        # Check if the ticker contains numbers and if it has a length of 4
        if not (1 <= len(ticker) <= 5 and ticker.isalpha()):
            print("Invalid Ticker")
            continue

        start_year, end_year = "", ""
        # Split year range
        try:
            tokens = year_range.split("-")
            start_year, end_year = tokens[0], tokens[1]
        except Exception:
            print("Invalid Year Range Format (ex: 2020-2024)")
            continue

        # Check if the years entered are valid
        if not (start_year and end_year and int(start_year) >= 2001 and int(end_year) <= 2024):
            print("Invalid year range, please enter a year from 2001 to 2024!")
            continue

        break

    # Check cache and fetch missing years
    all_chunks: list[tuple[dict, str]] = []
    filing_dates_by_year: dict[int, str] = {}

    for year in range(int(start_year), int(end_year) + 1):
        chunks = get_cached_chunks(ticker, year)

        if chunks is None:
            print(f"[{ticker} {year}] Not in cache, fetching from SEC EDGAR...")
            result = get_risk_factors(ticker, year)

            if result is None:
                print(f"[{ticker} {year}] No filing found, skipping.")
                continue

            filing_text, filing_date = result
            filing_dates_by_year[year] = filing_date
            store_filing(ticker, year, filing_text, VECTORSTORE)
            chunks = get_cached_chunks(ticker, year)
            print(f"[{ticker} {year}] Stored {len(chunks)} chunks.")
        else:
            print(f"[{ticker} {year}] Loaded {len(chunks)} chunks from cache.")
            filing_date = get_filing_date(ticker, year)
            if filing_date:
                filing_dates_by_year[year] = filing_date

        # for i, (meta, body) in enumerate(chunks):
        #     print(f"\n--- [{ticker} {year}] Chunk {i + 1} | {meta.get('category', '')} | {meta.get('word_count', '')} words ---")
        #     print(f"HEADER: {meta.get('header', '')}")
        #     print(f"BODY:   {body}")

        all_chunks.extend(chunks)

    # Summarize all chunks with zero-shot and few-shot prompting
    if not all_chunks:
        return

    print(f"\n{'='*60}")
    print(f"SUMMARIZATION — {ticker} {start_year}–{end_year}")
    print(f"{'='*60}")

    results = summarize_chunks(all_chunks, extra_models=EXTRA_MODELS)

    for r in results:
        flagged_zs = " ⚠ LOW" if r["zero_shot_flagged"] else ""
        flagged_fs = " ⚠ LOW" if r["few_shot_flagged"] else ""
        print(f"\n[{r['ticker']} {r['year']}] {r['chunk_id']} | {r['category']}")
        print(f"  Header:     {r['header']}")
        print(f"  Zero-Shot:  {r['zero_shot_summary']}")
        print(f"              BERTScore F1: {r['zero_shot_bertscore_f1']}{flagged_zs}")
        print(f"  Few-Shot:   {r['few_shot_summary']}")
        print(f"              BERTScore F1: {r['few_shot_bertscore_f1']}{flagged_fs}")

    log_summaries(results)
    print(f"\n  Saved summarization_results.csv")
    print_table1()

    # Phase 4: Year-over-Year Change Detection
    years = list(range(int(start_year), int(end_year) + 1))
    if len(years) < 2:
        return

    summaries_by_chunk_id = {r["chunk_id"]: r for r in results}
    changes_by_pair: dict[tuple[int, int], list[dict]] = {}

    for year_a, year_b in zip(years, years[1:]):
        print(f"\n{'='*60}")
        print(f"YEAR-OVER-YEAR CHANGES — {ticker} {year_a} → {year_b}")
        print(f"{'='*60}")

        changes = detect_changes(ticker, year_a, year_b, VECTORSTORE, summaries_by_chunk_id)
        changes_by_pair[(year_a, year_b)] = changes

        if not changes:
            print("  No change data available.")
            continue

        counts = {"NEW": 0, "REMOVED": 0, "ESCALATED": 0, "DE-ESCALATED": 0, "UNCHANGED": 0}
        for c in changes:
            counts[c["status"]] = counts.get(c["status"], 0) + 1

        print(f"\n  Summary: {counts['NEW']} NEW | {counts['REMOVED']} REMOVED | "
              f"{counts['ESCALATED']} ESCALATED | {counts['DE-ESCALATED']} DE-ESCALATED | "
              f"{counts['UNCHANGED']} UNCHANGED")

        for c in changes:
            status = c["status"]
            if status == "NEW":
                print(f"\n  [NEW] {c['chunk_id_b']}")
                print(f"    Header:  {c['header_b']}")
                if c["summary_b"]:
                    print(f"    Summary: {c['summary_b']}")
            elif status == "REMOVED":
                print(f"\n  [REMOVED] {c['chunk_id_a']}")
                print(f"    Header:  {c['header_a']}")
                if c["summary_a"]:
                    print(f"    Summary: {c['summary_a']}")
            else:
                cot_note = f" (CoT: {c['cot_status']})" if c["cot_status"] != status else ""
                print(f"\n  [{status}{cot_note}] {c['chunk_id_a']} → {c['chunk_id_b']} (sim={c['similarity']:.3f})")
                print(f"    {year_a}: {c['summary_a']}")
                print(f"    {year_b}: {c['summary_b']}")

    log_changes(ticker, changes_by_pair)
    print(f"\n  Saved change_detection_results.csv")
    print(f"  Fill in 'true_label' column, then run: python evaluator.py")

    # Phase 5: Stock Price Correlation
    year_pairs = list(zip(years, years[1:]))
    if not year_pairs:
        return

    print(f"\n{'='*60}")
    print(f"STOCK PRICE CORRELATION — {ticker}")
    print(f"{'='*60}")

    stats = analyze_stock_correlation(ticker, year_pairs, changes_by_pair, filing_dates_by_year)

    print(f"\n  {'Year Pair':<12} {'Net Risk':>9} {'Escalated':>10} {'De-esc':>7} {'Filing Date':>12} {'30d Return':>11}")
    print(f"  {'-'*63}")
    for r in stats["records"]:
        ret = f"{r['return_30d']*100:+.2f}%" if r["return_30d"] is not None else "   N/A"
        print(f"  {r['year_a']}→{r['year_b']:<6} {r['net_risk_score']:>9} {r['escalated']:>10} {r['de_escalated']:>7} {r['filing_date']:>12} {ret:>11}")

    corr = stats["correlation"]
    if "note" in corr:
        print(f"\n  {corr['note']}")
    else:
        print(f"\n  n={corr['n']} pairs")
        print(f"  Pearson  r={corr['pearson_r']:+.4f}  (p={corr['pearson_p']:.4f})")
        print(f"  Spearman ρ={corr['spearman_rho']:+.4f}  (p={corr['spearman_p']:.4f})")

    if stats["records"]:
        print(f"\n  Saved correlation_results.csv")


if __name__ == "__main__":
    main()
        


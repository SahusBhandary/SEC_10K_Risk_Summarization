from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from api import get_risk_factors
from vector import store_filing
from summarizer import summarize_chunks

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
        except Exception as e:
            print("Invalid Year Range Format (ex: 2020-2024)")
            continue

        # Check if the years entered are valid
        if not (start_year and end_year and int(start_year) >= 2001 and int(end_year) <= 2024):
            print("Invalid year range, please enter a year from 2001 to 2024!")
            continue

        break

    # Check cache and fetch missing years
    all_chunks: list[tuple[dict, str]] = []

    for year in range(int(start_year), int(end_year) + 1):
        chunks = get_cached_chunks(ticker, year)

        if chunks is None:
            print(f"[{ticker} {year}] Not in cache, fetching from SEC EDGAR...")
            filing_text = get_risk_factors(ticker, year)

            if filing_text is None:
                print(f"[{ticker} {year}] No filing found, skipping.")
                continue

            store_filing(ticker, year, filing_text, VECTORSTORE)
            chunks = get_cached_chunks(ticker, year)
            print(f"[{ticker} {year}] Stored {len(chunks)} chunks.")
        else:
            print(f"[{ticker} {year}] Loaded {len(chunks)} chunks from cache.")

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

    results = summarize_chunks(all_chunks)

    for r in results:
        flagged_zs = " ⚠ LOW" if r["zero_shot_flagged"] else ""
        flagged_fs = " ⚠ LOW" if r["few_shot_flagged"] else ""
        print(f"\n[{r['ticker']} {r['year']}] {r['chunk_id']} | {r['category']}")
        print(f"  Header:     {r['header']}")
        print(f"  Zero-Shot:  {r['zero_shot_summary']}")
        print(f"              BERTScore F1: {r['zero_shot_bertscore_f1']}{flagged_zs}")
        print(f"  Few-Shot:   {r['few_shot_summary']}")
        print(f"              BERTScore F1: {r['few_shot_bertscore_f1']}{flagged_fs}")


if __name__ == "__main__":
    main()
        


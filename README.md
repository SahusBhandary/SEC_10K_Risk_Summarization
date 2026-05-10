# SEC 10-K Risk Factor Summarizer & Year-over-Year Change Tracker

An NLP pipeline that extracts the Risk Factors section (Item 1A) from SEC 10-K annual filings, summarizes each risk in plain English using LLM prompting, detects year-over-year changes, and correlates risk-language shifts with stock price movements.

---

## 1. Original Source

This project is an original implementation. All code was written for this project.

External APIs and data sources used:
- **SEC EDGAR Submissions API** — `https://data.sec.gov/submissions/CIK{cik}.json`
- **SEC EDGAR Full-Text Filing Archive** — `https://www.sec.gov/Archives/edgar/data/`
- **Company Ticker → CIK Mapping** — `https://www.sec.gov/files/company_tickers.json`
- **Yahoo Finance (yfinance)** — stock price data pulled via the `yfinance` Python library

---

## 2. Modified Files and Functions

### `api.py`
| Function | Description |
|---|---|
| `get_risk_factors(ticker, year)` | Resolves ticker to CIK, fetches the 10-K primary document from EDGAR, extracts Item 1A via regex. Handles paginated older filings via the `files` array in the submissions JSON. Returns `(risk_text, filing_date)`. |
| `get_filing_date(ticker, year)` | Lightweight variant — CIK lookup + submissions JSON only, no document download. Used for years already cached in ChromaDB. Also handles paginated older filings. |

### `vector.py`
| Function | Description |
|---|---|
| `clean_text(text)` | Strips page markers and rejoins lines split at inline HTML elements (apostrophes, ampersands) produced by BeautifulSoup. |
| `_extract_header(para)` | Normalizes whitespace and extracts the first complete sentence (≥30 chars) from a paragraph as the chunk header. Prevents sentence-fragment headers from mid-window splits. |
| `_is_section_header(line)` | Detects top-level section headers (e.g., "Cybersecurity Risks") using a regex for 2–8 word title-case phrases ending in risk-related terms. |
| `_is_topic_header(line)` | Detects individual risk topic openers tuned for Apple-format 10-Ks: 10–60 word sentences starting with "The Company" or other specific starters, ending with ".". |
| `chunk_filing(filing_text)` | Full chunking pipeline: clean → detect section/topic headers → flush chunks → merge short chunks → remove boilerplate. Falls back to paragraph splitting (double-newline), then sliding-window (500-word windows, 450-word step) for companies whose format does not match the topic-header rules. |
| `store_filing(ticker, year, filing_text, vectorstore)` | Embeds all chunks via OpenAI embeddings and stores them in ChromaDB with metadata: ticker, year, category, header, word\_count, chunk\_id. |

### `summarizer.py`
| Function | Description |
|---|---|
| `summarize(chunk_text, method, llm)` | Calls an LLM with either the zero-shot or few-shot prompt template. Defaults to GPT-4o-mini; accepts an optional local Ollama LLM. |
| `bertscore(summary, source)` | Computes BERTScore precision, recall, and F1 for a generated summary against the source chunk using `bert-score`. |
| `summarize_chunks(chunks, methods, flag_threshold, extra_models)` | Iterates all chunks, runs each prompt method × each model, stores BERTScore P/R/F1, and flags summaries below the threshold. Supports optional Ollama-hosted local models (Llama 3.1 8B, Mistral 7B). |

### `change_detector.py`
| Function | Description |
|---|---|
| `_get_year_embeddings(ticker, year, vectorstore)` | Retrieves stored chunk embeddings and metadata from ChromaDB for a given ticker/year. |
| `_classify_standard(summary_a, summary_b, year_a, year_b)` | Single-step LLM classification: given two summaries, returns ESCALATED / DE-ESCALATED / UNCHANGED. |
| `_classify_cot(summary_a, summary_b, year_a, year_b)` | Chain-of-Thought classification: LLM reasons step-by-step then outputs a `LABEL: <X>` line; returns `(label, full_reasoning)`. |
| `detect_changes(ticker, year_a, year_b, vectorstore, summaries_by_chunk_id)` | Cosine similarity matching between Year A and Year B chunk embeddings. Thresholds: ≥0.85 = matched pair (sent to classifier), 0.50–0.85 = uncertain (treated as REMOVED), <0.50 = REMOVED. Unmatched Year B chunks = NEW. |

### `stock_correlation.py`
| Function | Description |
|---|---|
| `get_filing_return(ticker, filing_date, days)` | Pulls closing prices via `yf.Ticker.history()` and computes the 30-day return after the filing date, finding the closest available trading day. |
| `analyze_stock_correlation(ticker, year_pairs, changes_by_pair, filing_dates_by_year)` | Computes net\_risk\_score = ESCALATED − DE-ESCALATED per year pair, fetches 30-day returns, runs Pearson and Spearman correlation across all pairs, and saves `correlation_results.csv`. |

### `results_logger.py`
| Function | Description |
|---|---|
| `log_summaries(results, path)` | Enriches summarization results with Flesch Reading Ease scores (source + each summary method) and appends to `summarization_results.csv`. Writes header only on first call. |
| `log_changes(ticker, changes_by_pair, path)` | Flattens change-detection results into rows and appends to `change_detection_results.csv`, leaving `true_label` blank for manual annotation. |

### `evaluator.py`
| Function | Description |
|---|---|
| `print_table1(path)` | Reads `summarization_results.csv`; detects all model+method column combos dynamically; prints mean BERTScore P/R/F1 and Flesch scores per method. |
| `print_table2_table3(path)` | Reads `change_detection_results.csv`; skips unlabeled rows; computes macro-average and per-label P/R/F1 for standard vs. CoT prompting using `sklearn.metrics.classification_report`. |
| `print_table4(path)` | Reads `correlation_results.csv`; computes direction-match (whether net risk score and 30-day return moved in the predicted direction); prints formatted table. |

### `main.py`
| Function | Description |
|---|---|
| `get_cached_chunks(ticker, year)` | Queries ChromaDB for already-processed chunks by ticker and year. |
| `main()` | Full pipeline orchestration: input validation → cache-or-fetch loop → summarization → change detection → stock correlation → CSV logging → table printing. |

### `labeler.py`
| Function | Description |
|---|---|
| `main()` | Interactive ground-truth labeler. Shows model predictions as hints, reads a single keypress per label (n/e/d/u/r/s/q), and writes to CSV after every label using raw terminal mode (`tty`/`termios`). Resumes from where it left off. |

### `human_eval.py`
Standalone script. Samples up to 30 rows from `summarization_results.csv`, shows source text and both summaries, collects 1–5 ratings for clarity and usefulness, and saves `human_eval_results.csv`.

---

## 3. Commands

### Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

### Run the full pipeline
```bash
python main.py
# Prompts for: ticker (e.g. AAPL) and year range (e.g. 2019-2023)
# Runs all 5 phases: fetch → chunk → summarize → change detect → stock correlation
```

### Label change-detection pairs (required for Tables 2 & 3)
```bash
python labeler.py
# Keys: [n]NEW  [e]ESCALATED  [d]DE-ESCALATED  [u]UNCHANGED  [r]REMOVED  [s]skip  [q]quit
```

### Generate evaluation tables
```bash
python evaluator.py
# Prints Table 1 (BERTScore + Flesch), Table 2 (change detection macro F1),
# Table 3 (per-label breakdown), Table 4 (stock correlation)
```

### Human evaluation of summaries
```bash
python human_eval.py
# Samples 30 chunks, prompts for 1-5 clarity/usefulness ratings, saves human_eval_results.csv
```

### Recommended run order for full evidence collection
```bash
# Clear cache before a fresh full run to avoid duplicate CSV rows
rm -rf chroma_db/ summarization_results.csv change_detection_results.csv correlation_results.csv

python main.py    # Ticker: AAPL, Years: 2018-2023
python main.py    # Ticker: MSFT, Years: 2019-2023
python main.py    # Ticker: SNAP, Years: 2020-2023

python labeler.py
python evaluator.py
python human_eval.py
```

---

## 4. Models and Data

No models were trained. The pipeline uses pre-trained models accessed via API or auto-downloaded from HuggingFace.

| Component | Model | Access |
|---|---|---|
| Summarization & classification | GPT-4o-mini (`gpt-4o-mini`) | OpenAI API |
| Embeddings | `text-embedding-ada-002` | OpenAI API via LangChain |
| BERTScore evaluation | `microsoft/deberta-xlarge-mnli` | HuggingFace (auto-downloaded by `bert-score`) |
| Optional local models | Llama 3.1 8B, Mistral 7B v0.3 | Ollama (`http://localhost:11434`) |

**Data:** SEC EDGAR public 10-K filings, fetched live — no local dataset download required.
Evaluated on: AAPL (2018–2023), MSFT (2019–2023), SNAP (2020–2023).

To use optional local models:
```bash
ollama pull llama3.1:8b
ollama pull mistral:v0.3
# Then in main.py set:
EXTRA_MODELS: tuple[str, ...] = ("llama", "mistral")
```

---

## 5. Prompts

All four prompts are in [`prompts.txt`](prompts.txt):

| Prompt | Used in |
|---|---|
| Zero-Shot Summarization | Phase 3 — `summarizer.py` |
| Few-Shot Summarization (3 examples) | Phase 3 — `summarizer.py` |
| Standard Change-Detection Classification | Phase 4 — `change_detector.py` |
| Chain-of-Thought Change-Detection Classification | Phase 4 — `change_detector.py` |

---

## 6. Software Requirements

| Package | Version | Purpose |
|---|---|---|
| Python | ≥ 3.11 | Runtime |
| `openai` | 2.33.0 | GPT-4o-mini API |
| `langchain` | 1.2.17 | LLM orchestration |
| `langchain-openai` | 1.2.1 | OpenAI embeddings + chat |
| `langchain-community` | 0.4.1 | ChromaDB vector store integration |
| `chromadb` | 1.5.8 | Vector store (chunk cache + embeddings) |
| `bert-score` | — | BERTScore P/R/F1 evaluation |
| `scikit-learn` | — | Cosine similarity, classification report |
| `scipy` | — | Pearson and Spearman correlation |
| `yfinance` | — | Stock price data |
| `textstat` | — | Flesch Reading Ease scores |
| `beautifulsoup4` | — | HTML parsing of SEC filings |
| `requests` | 2.33.1 | SEC EDGAR API calls |
| `python-dotenv` | 1.2.2 | `.env` API key loading |

Full pinned dependency list: [`requirements.txt`](requirements.txt)

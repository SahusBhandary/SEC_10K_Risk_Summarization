import requests
import re
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "sahusbhandary04@gmail.com"}

def get_risk_factors(ticker: str, year: int) -> str | None:
    ticker = ticker.upper()

    # Step 1: Resolve ticker to CIK
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(tickers_url, headers=HEADERS)
    if response.status_code != 200:
        return None

    tickers_data = response.json()
    cik = None
    for entry in tickers_data.values():
        if entry["ticker"] == ticker:
            cik = str(entry["cik_str"]).zfill(10)
            break

    if cik is None:
        return None

    # Step 2: Find the 10-K filing for the given year
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(submissions_url, headers=HEADERS)
    if response.status_code != 200:
        return None

    filings = response.json().get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accession_numbers = filings.get("accessionNumber", [])
    primary_documents = filings.get("primaryDocument", [])
    filing_dates = filings.get("filingDate", [])

    ten_k_index = next(
        (
            i for i, (form, date) in enumerate(zip(forms, filing_dates))
            if form == "10-K" and date.startswith(str(year))
        ),
        None,
    )
    if ten_k_index is None:
        return None

    accession = accession_numbers[ten_k_index].replace("-", "")
    primary_doc = primary_documents[ten_k_index]

    # Step 3: Fetch the 10-K document
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
    response = requests.get(doc_url, headers=HEADERS)
    if response.status_code != 200:
        return None

    text = BeautifulSoup(response.text, "html.parser").get_text(separator="\n")

    # Step 4: Extract Item 1A (Risk Factors) section
    # Use findall to get all matches (TOC + actual section), then take the longest
    matches = re.findall(
        r"Item\s+1A\.?\s*Risk\s+Factors(.*?)(?:Item\s+1B\.|\n\s*Item\s+2\.)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not matches:
        return None

    return max(matches, key=len).strip()

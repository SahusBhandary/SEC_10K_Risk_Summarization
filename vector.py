import re
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Body/elaboration sentences — these start with "The Company" but are NOT topic headers
_COMPANY_BODY_STARTERS = (
    "The Company continues",
    "The Company also ",
    "The Company and its",
    "The Company and other",
    "The Company currently",
    "The Company has international",
    "The Company has a minority",
    "The Company has outsourced",
    "The Company has also",
    "The Company has entered",
    "The Company offers complex",
    "The Company records",
    "The Company reviews",
    "The Company orders",
    "The Company designs",
    "The Company contracts",
    "The Company believes",
)

# All other body/elaboration starters
_OTHER_BODY_STARTERS = (
    "In addition", "However,", "Such ", "As a result", "For example",
    "These ", "While ", "Although ", "Despite ", "This ", "During ",
    "Given ", "Because ", "Further,", "Additionally,", "Moreover,",
    "Furthermore,", "Substantially all", "Many of the", "Some of the",
    "Any ", "All of", "Each of", "Most of", "There can be", "There are ",
    "It is ", "We ", "Our ",
)

_BODY_STARTERS = _COMPANY_BODY_STARTERS + _OTHER_BODY_STARTERS

# Non-"The Company" topic header openers
_OTHER_TOPIC_STARTERS = (
    "Failure to ",
    "Global markets",
    "To remain competitive",
    "Future operating results",
    "Losses or unauthorized",
    "Investment in new",
    "Because of the following",
    "Increasing focus",
    "Compliance with",
    "Changes in tax",
    "Expectations relating to",
    "The price of the Company",
)

_SECTION_HEADER_RE = re.compile(
    r'^[A-Z][A-Za-z ,&]+(?:Risks?|Compliance|Issues?|Matters?|Obligations?)\.?$'
)

# All apostrophe-like characters (ASCII + Unicode smart quotes)
_APOSTROPHES = ("'", "’", "‘")


def clean_text(text: str) -> str:
    # Remove page markers: "Apple Inc. | 2023 Form 10-K | 7"
    text = re.sub(r'[A-Za-z][\w\s,\.]*\|\s*\d{4}\s*Form\s+\d{2}-K\s*\|\s*\d+', '', text)
    text = re.sub(r'-\s*\d+\s*-|(?<!\w)Page\s+\d+(?!\w)', '', text)

    # Rejoin lines split at inline HTML elements. BS4 emits a possessive like
    # "Company's" as three lines: "Company" / "'" / "s business...".
    # Similarly "R&D" splits into "R&" / "D".
    lines = text.split('\n')
    rejoined: list[str] = []
    for line in lines:
        stripped = line.strip()
        prev = rejoined[-1] if rejoined else ""
        if rejoined and stripped and stripped[0] in _APOSTROPHES:
            # Current line starts with apostrophe — glue to previous word
            rejoined[-1] = prev + stripped
        elif rejoined and prev and prev[-1] in _APOSTROPHES and stripped and stripped[0].islower():
            # Previous ends with apostrophe, current starts lowercase — the "s ..." fragment
            rejoined[-1] = prev + stripped
        elif rejoined and prev.endswith('&'):
            # Previous ended mid-entity (e.g. "R&"), glue next fragment (e.g. "D")
            rejoined[-1] = prev + stripped
        else:
            rejoined.append(stripped)

    text = '\n'.join(rejoined)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _is_section_header(line: str) -> bool:
    words = line.split()
    return (
        2 <= len(words) <= 8
        and not line.endswith('.')
        and bool(_SECTION_HEADER_RE.match(line))
    )


def _is_topic_header(line: str) -> bool:
    words = line.split()
    if not (10 <= len(words) <= 60 and line.endswith('.')):
        return False
    if any(line.startswith(s) for s in _BODY_STARTERS):
        return False
    # Broadly match any "The Company" sentence not filtered by body starters above
    if line.startswith("The Company"):
        return True
    return any(line.startswith(s) for s in _OTHER_TOPIC_STARTERS)


def _extract_header(para: str) -> str:
    clean = ' '.join(para.split())
    first_dot = clean.find('.')
    if first_dot < 30:
        return clean[:120]
    return clean[:first_dot + 1]


def _flush(chunks: list, header: str | None, body_lines: list, section: str):
    if not header or not body_lines:
        return
    body = ' '.join(body_lines)
    if len(body.split()) >= 50:
        chunks.append({"header": header, "body": body, "category": section})


def chunk_filing(filing_text: str) -> list[dict]:
    text = clean_text(filing_text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    chunks: list[dict] = []
    current_section = "Risk Factors"
    current_header: str | None = None
    current_body: list[str] = []

    for line in lines:
        if _is_section_header(line):
            _flush(chunks, current_header, current_body, current_section)
            current_section = line
            current_header = None
            current_body = []
        elif _is_topic_header(line):
            _flush(chunks, current_header, current_body, current_section)
            current_header = line
            current_body = []
        else:
            if current_header is not None:
                current_body.append(line)

    _flush(chunks, current_header, current_body, current_section)

    # Merge chunks under 50 words into the previous chunk
    merged: list[dict] = []
    for chunk in chunks:
        if merged and len(chunk["body"].split()) < 50:
            merged[-1]["body"] += " " + chunk["body"]
        else:
            merged.append(chunk)

    # Remove boilerplate intro chunks (identical across all 10-K filings, useless for comparison)
    _BOILERPLATE = (
        "past financial performance should not be considered",
        "this section should be read in conjunction with",
        "forward-looking statements",
        "can be affected by a number of factors",
        "because of the following factors",
    )
    merged = [
        c for c in merged
        if not any(p in c["header"].lower() for p in _BOILERPLATE)
    ]

    # Fallback: no/single topic header detected — try paragraph splitting first
    if len(merged) <= 1:
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.split()) >= 50]
        if len(paragraphs) > 1:
            fallback_chunks = []
            for para in paragraphs:
                fallback_chunks.append({
                    "header": _extract_header(para)[:300],
                    "body": para,
                    "category": "Risk Factors",
                })
            return fallback_chunks

        # iXBRL / single-newline text: use a sliding window
        words = text.split()
        if len(words) > 300:
            window, step = 500, 450
            fallback_chunks = []
            for i in range(0, len(words), step):
                sub_words = words[i:i + window]
                if len(sub_words) < 50:
                    continue
                sub_body = ' '.join(sub_words)
                fallback_chunks.append({
                    "header": _extract_header(sub_body)[:300],
                    "body": sub_body,
                    "category": "Risk Factors",
                })
            if fallback_chunks:
                return fallback_chunks

        return [{"header": "Risk Factors", "body": text, "category": "Risk Factors"}]

    return merged


def store_filing(ticker: str, year: int, filing_text: str, vectorstore: Chroma):
    chunks = chunk_filing(filing_text)

    texts = [c["body"] for c in chunks]
    metadatas = [
        {
            "ticker": ticker,
            "year": year,
            "category": c["category"],
            "header": c["header"][:500],
            "word_count": len(c["body"].split()),
            "chunk_id": f"{ticker}_{year}_chunk_{i:02d}",
        }
        for i, c in enumerate(chunks)
    ]

    vectorstore.add_texts(texts, metadatas=metadatas)

import re
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Phrases that open body/elaboration sentences, not topic headers
_BODY_STARTERS = (
    "In addition", "However,", "Such ", "As a result", "For example",
    "These ", "While ", "Although ", "The Company continues", "Despite ",
    "This ", "During ", "Given ", "Because ", "Further,", "Additionally,",
    "Moreover,", "Furthermore,", "Substantially all", "Many of the",
    "Some of the", "Any ", "All of", "Each of", "Most of", "There can be",
    "There are ", "It is ", "We ", "Our ",
)

# Phrases that open topic-level risk header sentences
_TOPIC_HEADER_STARTERS = (
    "The Company's",
    "The Company depends",
    "The Company relies",
    "The Company is subject",
    "The Company has invested",
    "The Company distributes",
    "The Company experiences",
    "The Company is exposed",
    "The Company operates",
    "The Company may be",
    "Failure to ",
    "Global markets",
    "To remain competitive",
    "Future operating results",
    "Losses or unauthorized",
    "Investment in new",
    "Because of the following",
)

_SECTION_HEADER_RE = re.compile(
    r'^[A-Z][A-Za-z ,&]+(?:Risks?|Compliance|Issues?|Matters?|Obligations?)\.?$'
)


def clean_text(text: str) -> str:
    # Remove page markers: "Apple Inc. | 2023 Form 10-K | 7"
    text = re.sub(r'[A-Za-z][\w\s,\.]*\|\s*\d{4}\s*Form\s+\d{2}-K\s*\|\s*\d+', '', text)
    text = re.sub(r'-\s*\d+\s*-|(?<!\w)Page\s+\d+(?!\w)', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _is_section_header(line: str) -> bool:
    words = line.split()
    return (
        3 <= len(words) <= 8
        and not line.endswith('.')
        and bool(_SECTION_HEADER_RE.match(line))
    )


def _is_topic_header(line: str) -> bool:
    words = line.split()
    if not (15 <= len(words) <= 60 and line.endswith('.')):
        return False
    if any(line.startswith(s) for s in _BODY_STARTERS):
        return False
    return any(line.startswith(s) for s in _TOPIC_HEADER_STARTERS)


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

    # Fallback: no topic headers detected — store the whole section as one chunk
    if not merged:
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

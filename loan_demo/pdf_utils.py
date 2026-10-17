from pypdf import PdfReader
import re
import unicodedata

def preprocess_text(text: str) -> str:
    """
    Normalize and clean the input text.

    Steps:
    1. Remove URLs (http and www)
    2. Unicode normalization to ASCII where possible (NFKD)
    3. Remove control characters
    4. Remove characters outside a conservative allowed set
    5. Collapse whitespace
    6. Remove duplicate lines and empty lines

    Parameters:
    - text: raw input string

    Returns:
    - cleaned string
    """
    if not text:
        return ""

    # 1. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 2. Normalize unicode (e.g., é → e)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

    # 3. Remove control characters (ASCII & Unicode)
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)

    # 4. Keep only alphanumerics, separators, and common punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"()\-\n]", "", text)

    # 5. Replace multiple spaces/tabs with a single space
    text = re.sub(r"\s+", " ", text)

    # 6. Clean duplicate lines
    lines = text.split('\n')
    cleaned_lines = []
    prev_line = ""
    for line in lines:
        if line.strip() and line.strip() != prev_line.strip():
            cleaned_lines.append(line.strip())
            prev_line = line

    # 7. Return normalized text
    return "\n".join(cleaned_lines).strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return preprocess_text("\n".join(pages))





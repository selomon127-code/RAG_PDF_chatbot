# src/pdf_loader.py
# Uses pdfplumber instead of PyMuPDF — avoids the DLL error on Windows

import pdfplumber
import os
import re


def load_pdf(pdf_path: str) -> list[dict]:
    """Extract text from every page of a PDF using pdfplumber."""

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"File must be a .pdf, got: {pdf_path}")

    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        print(f"[pdf_loader] Opened: {pdf_path}")
        print(f"[pdf_loader] Total pages: {len(pdf.pages)}")

        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text()

            # skip blank pages
            if not raw_text or len(raw_text.strip()) < 20:
                print(f"[pdf_loader] Skipping page {i+1} — blank")
                continue

            cleaned = clean_text(raw_text)

            pages_data.append({
                "page": i + 1,
                "text": cleaned
            })

    print(f"[pdf_loader] Extracted {len(pages_data)} pages.")
    return pages_data


def clean_text(text: str) -> str:
    """Clean raw extracted text."""

    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n")
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"(?<![.!?])\n(?=[a-z])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()
    return text


def get_full_text(pages_data: list[dict]) -> str:
    """Join all pages into one string."""
    return "\n\n".join(page["text"] for page in pages_data)


def save_extracted_text(pages_data: list[dict],
                        output_path: str) -> None:
    """Save extracted text to a debug .txt file."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for page in pages_data:
            f.write(f"\n{'='*60}\n")
            f.write(f"  PAGE {page['page']}\n")
            f.write(f"{'='*60}\n\n")
            f.write(page["text"])
            f.write("\n")

    print(f"[pdf_loader] Saved to: {output_path}")
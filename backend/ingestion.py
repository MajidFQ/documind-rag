"""
ingestion.py — Document text extraction for DocuMind.

Handles reading raw text out of uploaded PDF and DOCX files.
Each function has one job: either detect the file type, extract from PDF,
or extract from DOCX. The public entry point is extract_text().
"""

import pathlib
from typing import List, Dict

# pypdf is used for PDFs; it gives us page-level access
from pypdf import PdfReader

# python-docx is used for Word documents
from docx import Document


def get_file_extension(file_path: str) -> str:
    """
    Return the lowercase file extension (e.g. '.pdf', '.docx') for a given path.

    Centralising this makes the type-detection logic easy to read and test.
    """
    return pathlib.Path(file_path).suffix.lower()


def extract_text_from_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from a PDF file page by page using pypdf.

    Returns a list of dicts, one per page, so callers always know which page
    a piece of text came from — essential for source citations later.

    Each dict has:
        - 'text': the extracted string for that page
        - 'page_number': 1-based page index
        - 'source': the original file path
    """
    try:
        reader = PdfReader(file_path)
    except Exception as error:
        raise RuntimeError(
            f"Could not open PDF '{file_path}'. "
            f"Make sure the file is a valid, non-password-protected PDF. "
            f"Original error: {error}"
        )

    pages = []
    for index, page in enumerate(reader.pages):
        page_text = page.extract_text()

        # Some PDF pages contain only images and yield no text; skip them
        # rather than storing empty strings that would pollute search results.
        if not page_text or not page_text.strip():
            continue

        pages.append({
            "text": page_text.strip(),
            "page_number": index + 1,  # humans expect 1-based page numbers
            "source": file_path,
        })

    if not pages:
        raise ValueError(
            f"No readable text found in '{file_path}'. "
            "The PDF may be scanned/image-only. Consider running it through OCR first."
        )

    return pages


def extract_text_from_docx(file_path: str) -> List[Dict]:
    """
    Extract text from a Word (.docx) file paragraph by paragraph using python-docx.

    DOCX files don't have a built-in concept of 'pages', so we group all text
    into a single entry and use paragraph index as a positional reference.
    The 'page_number' field is set to None to make it clear no page data exists.

    Returns a list with a single dict containing the full document text.
    """
    try:
        document = Document(file_path)
    except Exception as error:
        raise RuntimeError(
            f"Could not open DOCX '{file_path}'. "
            f"Make sure the file is a valid .docx file. "
            f"Original error: {error}"
        )

    # Join all non-empty paragraphs with newlines to preserve visual structure
    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    if not paragraphs:
        raise ValueError(
            f"No readable text found in '{file_path}'. "
            "The document may be empty or contain only images/tables."
        )

    full_text = "\n".join(paragraphs)

    # Return as a list so the return type is consistent with extract_text_from_pdf
    return [{
        "text": full_text,
        "page_number": None,  # DOCX has no accessible page boundaries
        "source": file_path,
    }]


def extract_text(file_path: str) -> List[Dict]:
    """
    Detect the file type and dispatch to the correct extractor.

    This is the main public function other modules should call.
    It returns a list of dicts with 'text', 'page_number', and 'source' keys,
    regardless of whether the input was a PDF or DOCX.

    Raises a ValueError for unsupported file types so the caller gets a clear
    message instead of a confusing downstream failure.
    """
    extension = get_file_extension(file_path)

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif extension == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{extension}' for file '{file_path}'. "
            "DocuMind currently supports .pdf and .docx files only."
        )

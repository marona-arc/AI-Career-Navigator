"""
CV extraction — converts a PDF (bytes) into a candidate profile dict.

Flow:
  1. Extract raw text from PDF using pdfplumber (primary) or pypdf (fallback).
  2. Pass text to ArtifactBundle.build_candidate_profile() which reuses the
     exact same NLP pipeline as the notebook (cells 60/141).
  3. Validate the resulting profile.
"""

from __future__ import annotations
import io
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# PDF text extraction
# ──────────────────────────────────────────────────────────────────────────────


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract raw text from PDF bytes.
    Tries pdfplumber first, falls back to pypdf.
    Raises ValueError if no text can be extracted.
    """
    text = _try_pdfplumber(pdf_bytes)
    if not text or len(text.strip()) < 50:
        logger.warning("pdfplumber returned too little text; trying pypdf fallback")
        text = _try_pypdf(pdf_bytes)

    if not text or len(text.strip()) < 50:
        raise ValueError(
            "Could not extract readable text from this PDF. "
            "Please upload a text-based PDF (not a scanned image)."
        )

    return text


def _try_pdfplumber(pdf_bytes: bytes) -> str:
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)
    except Exception as exc:
        logger.debug("pdfplumber failed: %s", exc)
        return ""


def _try_pypdf(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages  = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as exc:
        logger.debug("pypdf failed: %s", exc)
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {
    "role", "seniority", "years_experience", "industry",
    "education", "skills", "titles", "text",
}


def validate_cv(cv: dict) -> None:
    """
    Validate a candidate profile dict.
    Raises ValueError with a user-friendly message on failure.
    """
    if not isinstance(cv, dict):
        raise ValueError("CV profile must be a dict.")

    missing = _REQUIRED_KEYS - set(cv.keys())
    if missing:
        raise ValueError(f"CV profile is missing required keys: {sorted(missing)}")

    # skills must be a list (can be empty, but warn)
    if not isinstance(cv["skills"], list):
        raise ValueError("CV 'skills' must be a list.")

    # years_experience must be a non-negative integer
    try:
        yrs = int(cv["years_experience"])
        if yrs < 0 or yrs > 60:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError(
            f"'years_experience' must be a non-negative integer, got: {cv['years_experience']!r}"
        )

    # text must be a non-empty string
    if not str(cv.get("text", "")).strip():
        raise ValueError(
            "Extracted CV text is empty after preprocessing. "
            "Please upload a richer CV."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def extract_cv_profile(pdf_bytes: bytes, bundle) -> dict:
    """
    Full pipeline: PDF bytes → validated candidate profile dict.
    bundle: ArtifactBundle instance.
    """
    raw_text   = extract_text_from_pdf(pdf_bytes)
    cv_profile = bundle.build_candidate_profile(raw_text)
    validate_cv(cv_profile)
    logger.info(
        "CV extracted — role: %s | seniority: %s | skills: %d | exp: %d yrs",
        cv_profile["role"],
        cv_profile["seniority"],
        len(cv_profile["skills"]),
        cv_profile["years_experience"],
    )
    return cv_profile

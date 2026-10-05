"""
Job cleaning — converts a raw Arbeitnow API job dict into a clean,
model-ready dict.

Cleans:
  - HTML tags and entities
  - Markdown artifacts
  - URLs inside text
  - Excess whitespace

Preserves:
  - title, company, location, remote, job_types, tags, url, slug, created_at
  - The original job URL is kept intact for the "View Job" button.
"""

from __future__ import annotations
import html
import logging
import re

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# HTML / text cleaning helpers
# ──────────────────────────────────────────────────────────────────────────────


def _strip_html(text: str) -> str:
    """Remove HTML tags using BeautifulSoup; fall back to regex."""
    try:
        from bs4 import BeautifulSoup  # optional but recommended
        return BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except ImportError:
        return re.sub(r"<[^>]+>", " ", text)


def clean_description(text: str) -> str:
    """
    Clean a job description:
    1. Unescape HTML entities (&amp; → &, etc.)
    2. Un-escape backslash-prefixed angle brackets
    3. Remove HTML tags
    4. Remove markdown-style links  [text](url)
    5. Remove bare URLs
    6. Normalize whitespace
    """
    if not text:
        return ""

    text = html.unescape(text)
    text = text.replace("\\<", "<").replace("\\>", ">")
    text = _strip_html(text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)      # [text](url)
    text = re.sub(r"https?://\S+", "", text)                   # bare URLs
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_field(value: str) -> str:
    """Cleaning for short fields (title, company, location), removing HTML and normalizing whitespace."""
    if not value:
        return ""
    text = html.unescape(str(value))
    text = _strip_html(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Job cleaner
# ──────────────────────────────────────────────────────────────────────────────


def clean_job(raw_job: dict) -> dict:
    """
    Convert a raw Arbeitnow API job dict into a clean dict.
    All display fields are cleaned; the original URL is preserved unchanged.
    """
    slug        = raw_job.get("slug", "")
    company     = clean_field(raw_job.get("company_name", "Unknown Company"))
    title       = clean_field(raw_job.get("title", ""))
    description = clean_description(raw_job.get("description", ""))
    remote      = bool(raw_job.get("remote", False))
    url         = raw_job.get("url", "")           # preserved verbatim
    tags        = [clean_field(t) for t in raw_job.get("tags", []) if t]
    job_types   = [clean_field(j) for j in raw_job.get("job_types", []) if j]
    location    = clean_field(raw_job.get("location", ""))
    created_at  = raw_job.get("created_at")

    if not title:
        raise ValueError("Job has no title after cleaning.")
    if not description:
        logger.warning("Job '%s' has no description after cleaning.", title)

    return {
        "slug":        slug,
        "company":     company,
        "title":       title,
        "description": description,
        "remote":      remote,
        "url":         url,
        "tags":        tags,          # kept for skill extraction
        "job_types":   job_types,
        "location":    location,
        "created_at":  created_at,
    }


def clean_jobs(raw_jobs: list[dict]) -> list[dict]:
    """
    Clean a list of raw jobs, skipping any that fail.
    Returns only successfully cleaned jobs.
    """
    cleaned = []
    for i, raw in enumerate(raw_jobs):
        try:
            cleaned.append(clean_job(raw))
        except Exception as exc:
            slug = raw.get("slug", f"index-{i}") if isinstance(raw, dict) else f"index-{i}"
            logger.warning("Skipping job [%s] during cleaning: %s", slug, exc)
    logger.info("Cleaned %d / %d jobs", len(cleaned), len(raw_jobs))
    return cleaned

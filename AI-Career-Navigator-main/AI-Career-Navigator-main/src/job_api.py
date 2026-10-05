"""
Arbeitnow API retrieval — fetches live jobs from:
  https://www.arbeitnow.com/api/job-board-api

Handles:
  - HTTP errors / timeouts
  - Malformed JSON
  - Missing 'data' key
  - Per-job validation
  - Deduplication by slug
"""

from __future__ import annotations
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

_API_URL     = "https://www.arbeitnow.com/api/job-board-api"
_TIMEOUT_SEC = 15

# ──────────────────────────────────────────────────────────────────────────────
# Raw API fetch
# ──────────────────────────────────────────────────────────────────────────────


def fetch_jobs(url: str = _API_URL, timeout: int = _TIMEOUT_SEC) -> list[dict]:
    """
    Fetch raw job list from the Arbeitnow API.
    Returns a list of raw job dicts.
    Raises RuntimeError with a user-friendly message on failure.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "The job board API timed out. Please try again in a moment."
        )
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"The job board API returned an error ({exc.response.status_code}). "
            "Please try again later."
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Could not reach the job board API: {exc}. "
            "Check your internet connection."
        )

    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError(
            "The job board API returned malformed JSON. Please try again later."
        )

    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError(
            "Unexpected response structure from the job board API (missing 'data'). "
            "Please try again later."
        )

    raw_jobs = payload["data"]
    if not raw_jobs:
        raise RuntimeError("The job board returned an empty job list.")

    logger.info("Fetched %d raw jobs from Arbeitnow API", len(raw_jobs))
    return raw_jobs


# ──────────────────────────────────────────────────────────────────────────────
# Per-job validation
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_JOB_KEYS = {"title", "description"}


def validate_raw_job(job: Any) -> None:
    """
    Raise ValueError if a single raw API job is malformed.
    Caller should catch and skip the job.
    """
    if not isinstance(job, dict):
        raise ValueError("Job record is not a dict.")
    for key in _REQUIRED_JOB_KEYS:
        if not job.get(key) or not str(job[key]).strip():
            raise ValueError(f"Job is missing required field: '{key}'.")


def validate_raw_jobs(raw_jobs: list) -> list[dict]:
    """
    Filter out malformed jobs and return the valid subset.
    Logs a warning for each skipped job.
    """
    valid = []
    for i, job in enumerate(raw_jobs):
        try:
            validate_raw_job(job)
            valid.append(job)
        except ValueError as exc:
            slug = job.get("slug", f"index-{i}") if isinstance(job, dict) else f"index-{i}"
            logger.warning("Skipping malformed job [%s]: %s", slug, exc)

    if not valid:
        raise RuntimeError("No valid jobs found after validation.")

    logger.info("%d / %d jobs passed validation", len(valid), len(raw_jobs))
    return valid


def deduplicate_jobs(raw_jobs: list[dict]) -> list[dict]:
    """Remove duplicates by slug (keep first occurrence)."""
    seen: set = set()
    unique   = []
    for job in raw_jobs:
        slug = job.get("slug", "")
        if slug and slug in seen:
            continue
        seen.add(slug)
        unique.append(job)
    logger.debug("Deduplicated: %d → %d jobs", len(raw_jobs), len(unique))
    return unique


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def get_jobs() -> list[dict]:
    """
    Fetch, validate, and deduplicate jobs from the Arbeitnow API.
    Returns a list of validated raw job dicts ready for cleaning.
    """
    raw    = fetch_jobs()
    valid  = validate_raw_jobs(raw)
    unique = deduplicate_jobs(valid)
    return unique

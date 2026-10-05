"""
Skill matching — computes shared and missing skills between a candidate
and a job, using the same normalized vocabulary as the notebook.

Both sets are normalized before comparison so that capitalization
differences (Python vs python) do not create spurious mismatches.
"""

from __future__ import annotations


def _normalize(skills: list[str]) -> set[str]:
    """Normalize a skill list to lowercase for comparison."""
    return {s.strip().lower() for s in skills if s.strip()}


def compute_skill_overlap(
    candidate_skills: list[str],
    job_skills: list[str],
) -> dict:
    """
    Compute shared and missing skills.

    Uses case-insensitive comparison but returns skills in their
    original canonical form (from the training vocabulary).

    Returns:
        {
            "shared_skills": list[str],   # skills candidate has
            "missing_skills": list[str],  # job requires, candidate lacks
            "skill_coverage": float,      # 0-100
        }
    """
    # Build lowercase→canonical maps for both sides
    cand_map = {s.strip().lower(): s for s in candidate_skills if s.strip()}
    job_map  = {s.strip().lower(): s for s in job_skills       if s.strip()}

    cand_lower = set(cand_map.keys())
    job_lower  = set(job_map.keys())

    shared_lower  = cand_lower & job_lower
    missing_lower = job_lower - cand_lower

    # Return canonical forms (capitalisation from vocabulary)
    shared_skills  = sorted(cand_map[k] for k in shared_lower)
    missing_skills = sorted(job_map[k]  for k in missing_lower)

    coverage = (
        round(len(shared_lower) / len(job_lower) * 100, 1)
        if job_lower
        else 0.0
    )

    return {
        "shared_skills":  shared_skills,
        "missing_skills": missing_skills,
        "skill_coverage": coverage,
    }

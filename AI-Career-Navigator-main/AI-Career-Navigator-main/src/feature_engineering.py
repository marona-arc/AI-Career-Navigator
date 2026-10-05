"""
Feature engineering — exact port of make_features() and pair_cosine()
from final_project.ipynb (cells 84-89).

Input:
  cands  : list of candidate profile dicts
  jobs   : list of job profile dicts
  bundle : ArtifactBundle (provides tfidf_text/title/skills, SENIORITY_RANK etc.)

Output:
  pd.DataFrame with columns matching bundle.features
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ── Seniority lookup tables (cell 84 of notebook) ────────────────────────────

SENIORITY_RANK = {
    "Junior": 0,
    "Mid":    1,
    "Senior": 2,
}

SENIORITY_YEARS = {
    "Junior": (0, 2),
    "Mid":    (2, 6),
    "Senior": (6, 12),
}


# ──────────────────────────────────────────────────────────────────────────────
# pair_cosine  (cell 86)
# ──────────────────────────────────────────────────────────────────────────────


def pair_cosine(
    vectorizer,
    texts_a: list[str],
    texts_b: list[str],
) -> np.ndarray:
    """
    Compute element-wise cosine similarity between paired text lists.
    Deduplicates internally for efficiency.  (notebook cell 86)
    """
    texts_a = list(texts_a)
    texts_b = list(texts_b)

    unique_texts = list(dict.fromkeys(texts_a + texts_b))
    index = {text: i for i, text in enumerate(unique_texts)}

    matrix = vectorizer.transform(unique_texts)

    ia = np.fromiter(
        (index[text] for text in texts_a), dtype=int, count=len(texts_a)
    )
    ib = np.fromiter(
        (index[text] for text in texts_b), dtype=int, count=len(texts_b)
    )

    return np.asarray(matrix[ia].multiply(matrix[ib]).sum(axis=1)).ravel()


# ──────────────────────────────────────────────────────────────────────────────
# make_features  (cell 87)
# ──────────────────────────────────────────────────────────────────────────────


def make_features(
    cands: list[dict],
    jobs: list[dict],
    bundle,                 # ArtifactBundle — avoids circular import
) -> pd.DataFrame:
    """
    Construct the 19-column feature DataFrame for a batch of (candidate, job)
    pairs.  This is a direct port of the notebook's make_features() function.
    (cell 87)
    """
    # Pre-compute batch TF-IDF similarities (vectorized for speed)
    cand_texts        = [c["text"]                                  for c in cands]
    job_texts         = [j["text"]                                  for j in jobs]
    cand_titles       = [c["role"].lower()                          for c in cands]
    job_titles        = [j["title"].lower()                         for j in jobs]
    cand_skills_text  = ["|".join(c["skills"])                      for c in cands]
    job_skills_text   = ["|".join(j["must_have"] + j["nice_to_have"]) for j in jobs]

    text_sim         = pair_cosine(bundle.tfidf_text,   cand_texts,       job_texts)
    title_sim        = pair_cosine(bundle.tfidf_title,  cand_titles,      job_titles)
    skill_tfidf_sim  = pair_cosine(bundle.tfidf_skills, cand_skills_text, job_skills_text)

    rows = []
    for i, (candidate, job) in enumerate(zip(cands, jobs)):
        candidate_skills = set(candidate["skills"])
        must_have        = set(job["must_have"])
        nice_to_have     = set(job["nice_to_have"])

        matching_skills = candidate_skills & must_have
        missing_skills  = must_have - candidate_skills
        nice_matching   = candidate_skills & nice_to_have
        union           = candidate_skills | must_have

        skill_jaccard = len(matching_skills) / len(union) if union else 0.0

        required_range  = SENIORITY_YEARS.get(job["seniority"], (0, 100))
        candidate_years = candidate["years_experience"]
        exp_in_range    = int(required_range[0] <= candidate_years <= required_range[1])
        exp_gap         = max(0, required_range[0] - candidate_years)

        candidate_seniority = SENIORITY_RANK.get(candidate["seniority"], 1)
        job_seniority       = SENIORITY_RANK.get(job["seniority"], 1)

        cand_role = str(candidate.get("role", "Other"))
        job_role  = str(job.get("title", "Other"))
        role_match = int(cand_role != "Other" and job_role != "Other" and cand_role.lower() == job_role.lower())

        cand_ind = str(candidate.get("industry", "Unknown"))
        job_ind  = str(job.get("industry", "Unknown"))
        industry_match = int(cand_ind != "Unknown" and job_ind != "Unknown" and cand_ind.lower() == job_ind.lower())

        rows.append(
            {
                # ── Skill features ────────────────────────────────────
                "n_must_have":        len(must_have),
                "n_matching_skills":  len(matching_skills),
                "n_missing_skills":   len(missing_skills),
                "must_have_coverage": (
                    len(matching_skills) / len(must_have) if must_have else 0.0
                ),
                "n_nice_matching":    len(nice_matching),
                "skill_jaccard":      skill_jaccard,
                "n_candidate_skills": len(candidate_skills),
                "skill_tfidf_sim":    float(skill_tfidf_sim[i]),
                # ── Non-skill features ────────────────────────────────
                "years_experience":   candidate_years,
                "exp_in_range":       exp_in_range,
                "exp_gap":            exp_gap,
                "candidate_seniority": candidate_seniority,
                "job_seniority":       job_seniority,
                "seniority_match":    int(candidate["seniority"] == job["seniority"]),
                "seniority_diff":     abs(candidate_seniority - job_seniority),
                "role_match":         role_match,
                "industry_match":     industry_match,
                "title_sim":          float(title_sim[i]),
                "text_sim":           float(text_sim[i]),
            }
        )

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Validate feature DataFrame before model input
# ──────────────────────────────────────────────────────────────────────────────


def validate_features(df: pd.DataFrame, expected_cols: list[str]) -> None:
    """
    Raise ValueError if the feature DataFrame does not match expected columns.
    """
    missing = set(expected_cols) - set(df.columns)
    extra   = set(df.columns)   - set(expected_cols)
    if missing:
        raise ValueError(f"Feature DataFrame missing columns: {sorted(missing)}")
    if extra:
        # Extra columns are not fatal but worth logging
        import logging
        logging.getLogger(__name__).warning(
            "Feature DataFrame has unexpected extra columns: %s", sorted(extra)
        )
    if df.isnull().any().any():
        raise ValueError(
            f"Feature DataFrame contains NaN values in: "
            f"{df.columns[df.isnull().any()].tolist()}"
        )

"""
Matcher — runs the full inference pipeline for one CV against all jobs.

Pipeline per job:
  clean job → build job profile → make features → model predict → skill overlap

Returns a list of result dicts sorted by ML model probability (descending).

Result schema
─────────────
{
    # Identity
    "job_id":           str,
    "title":            str,
    "company":          str,
    "location":         str,
    "remote":           bool,
    "job_type":         str,
    "url":              str,
    "created_at":       int | None,

    # ML output (from trained model — NOT skill coverage)
    "prediction":       int,             # 0 or 1
    "prediction_label": str,             # "MATCHED" | "NOT MATCHED"
    "match_probability":float,           # model.predict_proba positive class, 0–1
    "match_prob_pct":   float,           # match_probability * 100, 1 dp

    # Skill analysis  (explanation layer — NOT the ML prediction)
    "shared_skills":    list[str],
    "missing_skills":   list[str],
    "skill_coverage":   float,           # 0–100

    # Profile summaries (for UI display)
    "candidate_role":        str,
    "candidate_seniority":   str,
    "candidate_experience":  int,
    "candidate_education":   str,
    "candidate_industry":    str,
    "job_canonical_title":   str,
    "job_seniority":         str,
    "job_industry":          str,
    "job_required_years":    int | None,
    "job_skills":            list[str],

    # Debug data (populated only when debug=True)
    "_debug": {
        "candidate_features": dict,
        "job_features":       dict,
        "model_input":        dict,       # feature name → value
        "raw_prediction":     int,
        "class_probabilities": dict,      # class label → probability
        "model_classes":      list,
        "positive_class_idx": int,
    } | None,
}
"""

from __future__ import annotations
import logging
import numpy as np

from src.feature_engineering import make_features, validate_features
from src.skill_matching import compute_skill_overlap

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Job profile validation
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_JOB_PROFILE_KEYS = {
    "title", "seniority", "industry", "must_have", "nice_to_have", "text"
}


def validate_job_profile(jp: dict) -> None:
    missing = _REQUIRED_JOB_PROFILE_KEYS - set(jp.keys())
    if missing:
        raise ValueError(f"Job profile missing keys: {sorted(missing)}")
    if not isinstance(jp["must_have"], list):
        raise ValueError("'must_have' must be a list.")
    if not str(jp.get("text", "")).strip():
        raise ValueError("Job profile 'text' is empty.")


# ──────────────────────────────────────────────────────────────────────────────
# Positive-class detection
# ──────────────────────────────────────────────────────────────────────────────

def _positive_class_index(model) -> int:
    """Return the column index in predict_proba output for the positive class.

    The notebook target is label ∈ {0, 1} where 1 = match.
    We look for class label 1; fall back to the last column if not found.
    """
    classes = list(getattr(model, "classes_", [0, 1]))
    try:
        return classes.index(1)
    except ValueError:
        return len(classes) - 1


# ──────────────────────────────────────────────────────────────────────────────
# Single-pair prediction  (mirrors notebook cell 146)
# ──────────────────────────────────────────────────────────────────────────────


def predict_one(
    candidate: dict,
    job_profile: dict,
    bundle,
    debug: bool = False,
) -> dict:
    """
    Run the trained model for one (candidate, job) pair.

    Returns a dict with:
      - matched / match_probability / prediction
      - shared_skills / missing_skills / skill_coverage
      - _debug  (only when debug=True)
    """
    features_df = make_features([candidate], [job_profile], bundle)
    validate_features(features_df, bundle.features)
    X = features_df[bundle.features]

    raw_prediction = int(bundle.model.predict(X)[0])

    # ── Probability ──────────────────────────────────────────────────────────
    pos_idx = _positive_class_index(bundle.model)
    classes = list(getattr(bundle.model, "classes_", [0, 1]))

    if hasattr(bundle.model, "predict_proba"):
        proba_row = bundle.model.predict_proba(X)[0]
        match_probability = float(proba_row[pos_idx])
        class_probabilities = {int(c): float(p) for c, p in zip(classes, proba_row)}
    else:
        # Fallback: sigmoid of decision function (not a calibrated probability)
        decision_score = float(bundle.model.decision_function(X)[0])
        match_probability = float(1.0 / (1.0 + np.exp(-decision_score)))
        class_probabilities = {1: match_probability, 0: 1.0 - match_probability}

    # ── Skill overlap  (explanation layer, not the ML prediction) ────────────
    skill_info = compute_skill_overlap(
        candidate["skills"],
        job_profile["must_have"],
    )

    result = {
        "matched":           raw_prediction == 1,
        "prediction":        raw_prediction,
        "prediction_label":  "MATCHED" if raw_prediction == 1 else "NOT MATCHED",
        "match_probability": round(match_probability, 6),
        "confidence":        round(match_probability, 4),
        "shared_skills":     skill_info["shared_skills"],
        "missing_skills":    skill_info["missing_skills"],
        "skill_coverage":    skill_info["skill_coverage"],
        "_debug":            None,
    }

    if debug:
        result["_debug"] = {
            "candidate_features": {
                "role":            candidate.get("role"),
                "seniority":       candidate.get("seniority"),
                "years_experience":candidate.get("years_experience"),
                "industry":        candidate.get("industry"),
                "education":       candidate.get("education"),
                "skills":          candidate.get("skills"),
            },
            "job_features": {
                "canonical_title": job_profile.get("title"),
                "raw_title":       job_profile.get("raw_title"),
                "seniority":       job_profile.get("seniority"),
                "industry":        job_profile.get("industry"),
                "must_have":       job_profile.get("must_have"),
                "required_years":  job_profile.get("required_years"),
                "education_req":   job_profile.get("education_required"),
            },
            "model_input":         {col: float(X.iloc[0][col]) for col in bundle.features},
            "raw_prediction":      raw_prediction,
            "class_probabilities": class_probabilities,
            "model_classes":       classes,
            "positive_class_idx":  pos_idx,
        }

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Full matching pipeline
# ──────────────────────────────────────────────────────────────────────────────


def match_cv_to_jobs(
    candidate:    dict,
    cleaned_jobs: list[dict],
    bundle,
    debug: bool = False,
) -> list[dict]:
    """
    Match one candidate profile against all cleaned API jobs.

    For each job:
      1. Build job profile using bundle.build_job_profile()
      2. Validate job profile
      3. Predict ML match probability (from trained model)
      4. Compute shared / missing skills (explanation layer)
      5. Assemble result dict

    Skips malformed jobs instead of crashing.
    Returns results sorted by ML model probability (descending).
    """
    results = []

    for i, job in enumerate(cleaned_jobs):
        job_title = job.get("title", "")
        job_slug  = job.get("slug", f"index-{i}")

        try:
            job_profile = bundle.build_job_profile(
                title       = job_title,
                description = job.get("description", ""),
                tags        = job.get("tags", []),
            )
            validate_job_profile(job_profile)

            pred = predict_one(candidate, job_profile, bundle, debug=debug)

            result = {
                # ── Identity ─────────────────────────────────────────
                "job_id":     job_slug,
                "title":      job_title,
                "company":    job.get("company", ""),
                "location":   job.get("location", ""),
                "remote":     job.get("remote", False),
                "job_type":   ", ".join(job.get("job_types", [])) or "Not specified",
                "url":        job.get("url", ""),
                "created_at": job.get("created_at"),

                # ── ML output (from the trained model) ───────────────
                "prediction":        pred["prediction"],
                "prediction_label":  "MATCHED" if pred["matched"] else "NOT MATCHED",
                "matched":           pred["matched"],
                "match_probability": pred["match_probability"],
                "match_prob_pct":    round(pred["match_probability"] * 100, 1),
                "confidence":        pred["match_probability"],
                "confidence_pct":    round(pred["match_probability"] * 100, 1),

                # ── Skill analysis (explanation layer) ───────────────
                "shared_skills":  pred["shared_skills"],
                "missing_skills": pred["missing_skills"],
                "skill_coverage": pred["skill_coverage"],

                # ── Candidate profile summary (for UI display) ───────
                "candidate_role":       candidate.get("role", ""),
                "candidate_seniority":  candidate.get("seniority", ""),
                "candidate_experience": candidate.get("years_experience", 0),
                "candidate_education":  candidate.get("education", ""),
                "candidate_industry":   candidate.get("industry", "Unknown"),

                # ── Job profile summary (for UI display) ─────────────
                "job_canonical_title": job_profile["title"],
                "job_seniority":       job_profile["seniority"],
                "job_industry":        job_profile["industry"],
                "job_required_years":  job_profile.get("required_years"),
                "job_skills":          job_profile["must_have"],

                # ── Debug data (None unless debug=True) ──────────────
                "_debug": pred["_debug"],
            }
            results.append(result)

        except Exception as exc:
            logger.warning(
                "Skipping job [%s – %s] due to error: %s",
                job_slug, job_title, exc
            )
            continue

    # Sort by ML model probability descending (NOT by skill coverage)
    results.sort(key=lambda r: r["match_probability"], reverse=True)

    matched_count = sum(1 for r in results if r["matched"])
    logger.info(
        "Matching complete: %d jobs processed, %d matched (threshold 0.5)",
        len(results), matched_count
    )
    return results

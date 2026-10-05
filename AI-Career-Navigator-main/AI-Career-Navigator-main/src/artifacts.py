"""
Artifact loader — loads the two .joblib files saved by final_project.ipynb
and wires up all NLP functions that depend on the saved vocabulary.

This module is the single access point for:
  - the trained model
  - the feature list
  - the three TF-IDF vectorizers
  - the skill extractor (built from SKILL_VOCAB + SKILL_ALIASES)
  - the role canonicalizer (built from KNOWN_ROLES)
"""

from __future__ import annotations
import logging
from pathlib import Path

import joblib

from src.nlp_utils import (
    build_skill_extractor,
    build_role_canonicalizer,
    extract_years_experience,
    extract_education,
    extract_seniority,
    extract_industry,
    preprocess_text,
)

logger = logging.getLogger(__name__)

# Default artifact paths (relative to project root)
_MODEL_PATH = Path("models/ai_resume_job_match_model.joblib")
_NLP_PATH   = Path("models/ai_resume_job_match_nlp_artifacts.joblib")


class ArtifactBundle:
    """All trained objects needed for inference, loaded once."""

    def __init__(
        self,
        model_path: Path = _MODEL_PATH,
        nlp_path: Path   = _NLP_PATH,
    ) -> None:
        self._load(model_path, nlp_path)

    # ------------------------------------------------------------------
    # Internal loader
    # ------------------------------------------------------------------

    def _load(self, model_path: Path, nlp_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}\n"
                "Run final_project.ipynb (cells 137-138) to generate it."
            )
        if not nlp_path.exists():
            raise FileNotFoundError(
                f"NLP artifact not found: {nlp_path}\n"
                "Run final_project.ipynb (cell 139) to generate it."
            )

        model_data = joblib.load(model_path)
        nlp_data   = joblib.load(nlp_path)

        # ── Model ────────────────────────────────────────────────────
        self.model            = model_data["model"]
        self.features         = model_data["features"]           # list[str]
        self.feature_set_name = model_data["feature_set_name"]   # str
        self.model_name       = model_data["model_name"]         # str

        # ── TF-IDF vectorizers ───────────────────────────────────────
        self.tfidf_text   = nlp_data["tfidf_text"]
        self.tfidf_title  = nlp_data["tfidf_title"]
        self.tfidf_skills = nlp_data["tfidf_skills"]

        # ── Vocabulary / feature lists ───────────────────────────────
        self.SKILL_VOCAB        = nlp_data["SKILL_VOCAB"]
        self.SKILL_ALIASES      = nlp_data["SKILL_ALIASES"]
        self.SKILL_FEATURES     = nlp_data["SKILL_FEATURES"]
        self.NON_SKILL_FEATURES = nlp_data["NON_SKILL_FEATURES"]
        self.ALL_FEATURES       = nlp_data["ALL_FEATURES"]
        self.KNOWN_ROLES        = nlp_data["KNOWN_ROLES"]

        # ── Wire up vocabulary-dependent NLP functions ────────────────
        self.extract_skills = build_skill_extractor(
            self.SKILL_VOCAB, self.SKILL_ALIASES
        )
        self.canonical_role, self.extract_job_titles = build_role_canonicalizer(
            self.KNOWN_ROLES
        )

        logger.info(
            "Artifacts loaded — model: %s | features: %s | skills vocab: %d",
            self.model_name,
            self.feature_set_name,
            len(self.SKILL_VOCAB),
        )

    # ------------------------------------------------------------------
    # Convenience: build profiles (mirrors notebook cells 60-61 / 141-142)
    # ------------------------------------------------------------------

    def build_candidate_profile(self, cv_text: str) -> dict:
        """Build a candidate profile from raw CV text. (notebook cell 60/141)"""
        years  = extract_years_experience(cv_text)
        titles = self.extract_job_titles(cv_text)
        role   = titles[0] if titles else self.canonical_role(str(cv_text)[:200])

        return {
            "role":             role,
            "seniority":        extract_seniority(cv_text, years),
            "years_experience": years,
            "industry":         extract_industry(cv_text),
            "education":        extract_education(cv_text),
            "skills":           self.extract_skills(cv_text),
            "titles":           titles,
            "text":             preprocess_text(cv_text),
        }

    def build_extended_candidate_profile(self, cv_text: str) -> dict:
        """
        Build a candidate profile using the extended skill vocabulary and
        improved role detector from src.semantic_matcher.

        This recognises AI/ML/DL/Cloud skills missing from the 73-skill
        training vocabulary and detects roles like 'AI / ML Engineer' that
        the training-corpus canonical_role() maps incorrectly.
        """
        from src.semantic_matcher import build_extended_candidate_profile  # noqa: PLC0415
        return build_extended_candidate_profile(cv_text, self)

    def build_job_profile(
        self,
        title: str,
        description: str,
        tags: list[str] | None = None,
        industry: str | None = None,
    ) -> dict:
        """Build a job profile from title + description. (notebook cell 61/142)

        industry: if None (default), auto-detected from title/description/tags.
                  Pass an explicit value to override auto-detection.
        """
        blob = f"{title} {description} {' '.join(tags or [])}"

        required_years = _extract_required_years(blob)

        resolved_industry = industry if industry is not None else extract_industry(blob, tags=tags)

        return {
            "title":              self.canonical_role(title),
            "raw_title":          title,
            "seniority":          extract_seniority(title, required_years),
            "industry":           resolved_industry,
            "must_have":          self.extract_skills(blob),
            "nice_to_have":       [],
            "required_years":     required_years,
            "education_required": _extract_education_requirement(blob),
            "text":               preprocess_text(f"{title} {description}"),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers (notebook cells 61 / 142)
# ──────────────────────────────────────────────────────────────────────────────


def _extract_required_years(text: str) -> int | None:
    import re
    numbers = [
        int(x)
        for x in re.findall(
            r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)",
            str(text).lower(),
        )
    ]
    return min(numbers) if numbers else None


def _extract_education_requirement(text: str) -> str | None:
    level = extract_education(text)
    return None if level == "Unknown" else level

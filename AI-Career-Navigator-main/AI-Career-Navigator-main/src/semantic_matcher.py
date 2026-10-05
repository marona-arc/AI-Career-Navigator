"""
semantic_matcher.py — Role-aware, holistic candidate-job matching.

This module replaces the pure skill-coverage matching pipeline for the app.

ROOT CAUSES OF OLD PROBLEMS:
  1. The trained XGBoost model (feature_set_name="All Features") has
     must_have_coverage=52%, n_matching_skills=29%, skill_jaccard=17%
     as its top features with role/seniority/industry at 0% importance.
     One matching skill out of one required skill → must_have_coverage=1.0
     → model outputs 99.99% confidence regardless of role alignment.
  2. The training SKILL_VOCAB only has 73 generic business skills, missing
     all AI/ML/DL/Data skills (PyTorch, TensorFlow, YOLO, etc.).
     An AI/ML CV gets only 3 skills recognized: Python, SQL, Forecasting.
  3. The role canonicalizer maps "AI / ML Engineer" → "Backend Engineer"
     because the training corpus had no AI/ML roles.

THIS MODULE FIXES:
  - Expands skill recognition to include AI/ML/Data/Cloud/DevOps skills.
  - Implements role/domain-first semantic scoring: domain compatibility
    gates the overall match score, so a single incidental keyword overlap
    from a mismatched domain (e.g. "Forecasting" in Head of Sales) cannot
    produce a high match for an AI/ML Engineer candidate.
  - Produces a meaningful, varied match score (0–98%) and a separate
    calibration-based confidence (0.0–0.95).
  - Generates a human-readable explanation for every result.

EXISTING CODE LEFT UNCHANGED:
  - src/matcher.py  (still used if you ever want old model-only results)
  - src/feature_engineering.py
  - src/artifacts.py / build_candidate_profile() / build_job_profile()
  - src/skill_matching.py
  - All training artifacts and final_project.ipynb
"""

from __future__ import annotations
import re
import logging
from typing import Optional

from src.nlp_utils import (
    preprocess_text,
    extract_years_experience,
    extract_seniority,
    extract_education,
    extract_industry,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Expanded Skill Vocabulary
#    Extends the 73-skill training vocabulary with AI/ML/Data/Cloud/Dev skills.
#    Maps raw text aliases → canonical skill name.
# ──────────────────────────────────────────────────────────────────────────────

_EXTENDED_SKILL_ALIASES: dict[str, str] = {
    # AI / ML / Deep Learning
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "computer vision": "Computer Vision",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "llm": "LLMs",
    "llms": "LLMs",
    "large language models": "LLMs",
    "generative ai": "Generative AI",
    "genai": "Generative AI",
    "rag": "RAG",
    "retrieval augmented generation": "RAG",
    "retrieval-augmented generation": "RAG",
    "reinforcement learning": "Reinforcement Learning",
    "neural networks": "Neural Networks",
    "neural network": "Neural Networks",
    "transformers": "Transformers",
    "yolo": "YOLO",
    "opencv": "OpenCV",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "keras": "Keras",
    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "huggingface": "Hugging Face",
    "hugging face": "Hugging Face",
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
    "spacy": "SpaCy",
    "nltk": "NLTK",
    "mlops": "MLOps",
    "data science": "Data Science",
    "predictive modeling": "Predictive Modeling",
    "predictive modelling": "Predictive Modeling",
    "time series": "Forecasting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "object detection": "Object Detection",
    "image segmentation": "Image Segmentation",
    "image classification": "Image Classification",
    "speech recognition": "Speech Recognition",
    "recommendation systems": "Recommendation Systems",
    "model deployment": "Model Deployment",
    "model training": "Model Training",

    # Cloud & Infrastructure
    "aws": "AWS",
    "amazon web services": "AWS",
    "azure": "Azure",
    "microsoft azure": "Azure",
    "gcp": "GCP",
    "google cloud": "GCP",
    "google cloud platform": "GCP",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "terraform": "Terraform",
    "linux": "Linux",
    "unix": "Linux",
    "airflow": "Airflow",
    "apache airflow": "Airflow",
    "spark": "Apache Spark",
    "apache spark": "Apache Spark",
    "kafka": "Kafka",
    "databricks": "Databricks",
    "snowflake": "Snowflake",
    "dbt": "dbt",
    "redis": "Redis",
    "elasticsearch": "Elasticsearch",
    "mlflow": "MLflow",
    "kubeflow": "Kubeflow",
    "wandb": "Weights & Biases",
    "weights and biases": "Weights & Biases",
    "weights & biases": "Weights & Biases",

    # Languages & Frameworks
    "c++": "C++",
    "c#": "C#",
    "golang": "Go",
    "go lang": "Go",
    "rust": "Rust",
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "fastapi": "FastAPI",
    "react": "React",
    "vue": "Vue.js",
    "angular": "Angular",
    "typescript": "TypeScript",
    "graphql": "GraphQL",
    "grpc": "gRPC",
    "microservices": "Microservices",
    "rabbitmq": "RabbitMQ",
    "celery": "Celery",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    "plotly": "Plotly",

    # Business / Sales (add canonical maps that keep existing vocabulary working)
    "b2b sales": "B2B Sales",
    "enterprise sales": "Enterprise Sales",
    "solution selling": "Solution Selling",
    "saas sales": "SaaS Sales",
    "quota attainment": "Quota Attainment",
    "revenue growth": "Revenue Growth",
}


def _build_extended_extractor(bundle):
    """
    Build a skill extractor that recognises both the original 73-skill vocab
    AND the expanded AI/ML/Cloud vocab.

    Returns an extract_skills(text: str) -> list[str] callable.
    """
    combined_aliases = dict(bundle.SKILL_ALIASES)
    combined_aliases.update(_EXTENDED_SKILL_ALIASES)
    combined_vocab = sorted(
        set(bundle.SKILL_VOCAB) | set(combined_aliases.values())
    )

    alias_to_canonical: dict[str, str] = {s.lower(): s for s in combined_vocab}
    for alias, target in combined_aliases.items():
        alias_to_canonical[alias.lower()] = target

    patterns = [
        (
            re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE),
            alias_to_canonical[alias],
        )
        for alias in sorted(alias_to_canonical.keys(), key=len, reverse=True)
    ]

    def extract_skills(text: str) -> list[str]:
        text_mut = str(text).lower()
        found: set[str] = set()
        for patt, skill in patterns:
            if patt.search(text_mut):
                found.add(skill)
                text_mut = patt.sub(" ", text_mut)
        return sorted(found)

    return extract_skills


# ──────────────────────────────────────────────────────────────────────────────
# 2. Role & Domain Classification
# ──────────────────────────────────────────────────────────────────────────────

# Domain → keyword patterns (applied to job/candidate titles)
_DOMAIN_PATTERNS: list[tuple[str, list[str]]] = [
    ("AI / Machine Learning", [
        r"\bai\s*/\s*ml\b", r"\bml\s*/\s*ai\b",
        r"\bmachine\s+learning\b", r"\bdeep\s+learning\b",
        r"\bapplied\s+ai\b", r"\bai\s+engineer\b", r"\bai\s+scientist\b",
        r"\bai\s+researcher\b", r"\bcomputer\s+vision\b",
        r"\b(?:nlp|natural\s+language\s+processing)\b",
        r"\bmlops\b", r"\bdata\s+scientist\b", r"\bml\s+engineer\b",
    ]),
    ("Data & Analytics", [
        r"\bdata\s+engineer\b", r"\bdata\s+analyst\b",
        r"\banalytics\s+engineer\b", r"\bbi\s+analyst\b",
        r"\bbusiness\s+intelligence\b", r"\bbig\s+data\b",
    ]),
    ("Software & Infrastructure", [
        r"\bsoftware\s+engineer\b", r"\bsoftware\s+developer\b",
        r"\bbackend\s+(?:engineer|developer)\b",
        r"\bfrontend\s+(?:engineer|developer)\b",
        r"\bfull\s*stack\b",
        r"\bdevops\b", r"\bplatform\s+engineer\b",
        r"\bcloud\s+engineer\b", r"\bsystems\s+engineer\b",
        r"\bsite\s+reliability\b", r"\bsecurity\s+engineer\b",
        r"\bweb\s+developer\b", r"\bapi\s+developer\b",
        r"\bsolutions?\s+architect\b",
    ]),
    ("Product & Project Management", [
        r"\bproduct\s+manager\b", r"\bproject\s+manager\b",
        r"\bprogram\s+manager\b", r"\bscrum\s+master\b",
        r"\btechnical\s+product\b",
    ]),
    ("Sales & Commercial", [
        r"\bhead\s+of\s+sales\b", r"\bvp\s+(?:of\s+)?sales\b",
        r"\bdirector\s+(?:of\s+)?sales\b", r"\bsales\s+manager\b",
        r"\baccount\s+executive\b", r"\bsales\s+representative\b",
        r"\bbusiness\s+development\s+(?:manager|representative)\b",
        r"\bbdr\b", r"\bsdr\b",
        r"\bcustomer\s+success\b", r"\bchannel\s+sales\b",
        r"\bpre-?sales\b", r"\bsales\s+executive\b",
    ]),
    ("Marketing & Growth", [
        r"\bmarketing\s+manager\b", r"\bcontent\s+market\w+\b",
        r"\bperformance\s+market\w+\b", r"\bgrowth\s+(?:hacker|specialist|manager)\b",
        r"\bseo\s+specialist\b",
    ]),
    ("Finance & Accounting", [
        r"\bfinancial\s+analyst\b", r"\bfp&a\b", r"\bjunior\s+accountant\b",
        r"\bfinance\s+manager\b", r"\baccounting\b",
    ]),
    ("Operations & Support", [
        r"\boperations\s+manager\b", r"\bcustomer\s+support\b",
        r"\btechnical\s+support\b", r"\bhr\s+manager\b",
    ]),
]

# Pairwise domain compatibility (symmetric not guaranteed; define both directions)
_DOMAIN_COMPAT: dict[tuple[str, str], float] = {
    ("AI / Machine Learning", "AI / Machine Learning"): 1.00,
    ("AI / Machine Learning", "Data & Analytics"): 0.80,
    ("AI / Machine Learning", "Software & Infrastructure"): 0.65,
    ("AI / Machine Learning", "Product & Project Management"): 0.28,
    ("AI / Machine Learning", "Sales & Commercial"): 0.05,
    ("AI / Machine Learning", "Marketing & Growth"): 0.10,
    ("AI / Machine Learning", "Finance & Accounting"): 0.12,
    ("AI / Machine Learning", "Operations & Support"): 0.10,

    ("Data & Analytics", "AI / Machine Learning"): 0.80,
    ("Data & Analytics", "Data & Analytics"): 1.00,
    ("Data & Analytics", "Software & Infrastructure"): 0.65,
    ("Data & Analytics", "Product & Project Management"): 0.30,
    ("Data & Analytics", "Sales & Commercial"): 0.08,
    ("Data & Analytics", "Finance & Accounting"): 0.45,

    ("Software & Infrastructure", "AI / Machine Learning"): 0.65,
    ("Software & Infrastructure", "Software & Infrastructure"): 1.00,
    ("Software & Infrastructure", "Data & Analytics"): 0.65,
    ("Software & Infrastructure", "Product & Project Management"): 0.35,
    ("Software & Infrastructure", "Sales & Commercial"): 0.06,

    ("Product & Project Management", "Software & Infrastructure"): 0.35,
    ("Product & Project Management", "AI / Machine Learning"): 0.28,
    ("Product & Project Management", "Product & Project Management"): 1.00,

    ("Sales & Commercial", "Sales & Commercial"): 1.00,
    ("Sales & Commercial", "Marketing & Growth"): 0.55,
    ("Sales & Commercial", "AI / Machine Learning"): 0.05,

    ("Marketing & Growth", "Sales & Commercial"): 0.55,
    ("Marketing & Growth", "Marketing & Growth"): 1.00,
}


def _get_domain(title: str) -> str:
    """Classify a job or candidate title into its career domain."""
    t = str(title).lower()
    for domain, patterns in _DOMAIN_PATTERNS:
        for pat in patterns:
            if re.search(pat, t):
                return domain
    return "General / Other"


def _domain_compat(cand_domain: str, job_domain: str) -> float:
    """Return semantic compatibility (0–1) between two career domains."""
    if cand_domain == job_domain:
        return 1.0
    return _DOMAIN_COMPAT.get((cand_domain, job_domain), 0.08)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Candidate Role Detection (CV-text based)
# ──────────────────────────────────────────────────────────────────────────────

_ROLE_INDICATORS: list[tuple[str, list[str]]] = [
    ("AI / ML Engineer",        [r"\bai\s*/\s*ml\s+engineer\b", r"\bml\s*/\s*ai\s+engineer\b"]),
    ("Machine Learning Engineer",[r"\bmachine\s+learning\s+engineer\b", r"\bml\s+engineer\b"]),
    ("Applied AI Scientist",     [r"\bapplied\s+ai\s+scientist\b", r"\bai\s+scientist\b"]),
    ("AI Engineer",              [r"\bai\s+engineer\b", r"\bartificial\s+intelligence\s+engineer\b"]),
    ("MLOps Engineer",           [r"\bmlops\s+engineer\b", r"\bml\s+platform\s+engineer\b"]),
    ("Computer Vision Engineer", [r"\bcomputer\s+vision\s+engineer\b", r"\bcv\s+engineer\b"]),
    ("NLP Engineer",             [r"\bnlp\s+engineer\b", r"\bnatural\s+language\s+processing\s+engineer\b"]),
    ("Data Scientist",           [r"\bdata\s+scientist\b"]),
    ("Data Engineer",            [r"\bdata\s+engineer\b", r"\bcloud\s+data\s+engineer\b"]),
    ("Data Analyst",             [r"\bdata\s+analyst\b", r"\bbi\s+analyst\b"]),
    ("Full Stack Engineer",      [r"\bfull[\s-]stack\s+(?:engineer|developer)\b"]),
    ("Backend Engineer",         [r"\bbackend\s+(?:engineer|developer)\b"]),
    ("Frontend Engineer",        [r"\bfrontend\s+(?:engineer|developer)\b"]),
    ("DevOps Engineer",          [r"\bdevops\s+engineer\b", r"\bplatform\s+engineer\b", r"\bcloud\s+engineer\b"]),
    ("Software Engineer",        [r"\bsoftware\s+(?:engineer|developer)\b"]),
    ("Product Manager",          [r"\bproduct\s+manager\b"]),
    ("Project Manager",          [r"\bproject\s+manager\b"]),
    ("Head of Sales",            [r"\bhead\s+of\s+sales\b", r"\bdirector\s+of\s+sales\b"]),
    ("Account Executive",        [r"\baccount\s+executive\b"]),
    ("Sales Representative",     [r"\bsales\s+representative\b"]),
]


def extract_cv_role(cv_text: str) -> str:
    """
    Detect candidate's professional role from CV text.

    Priority 1: title line (first 10 non-empty lines of the CV).
    Priority 2: pattern anywhere in the full CV text.
    Fallback: "Software Engineer".
    """
    lines = [l.strip() for l in cv_text.split("\n") if l.strip()][:10]
    header_block = " ".join(lines)

    for canonical, patterns in _ROLE_INDICATORS:
        for pat in patterns:
            if re.search(pat, header_block, re.IGNORECASE):
                return canonical

    for canonical, patterns in _ROLE_INDICATORS:
        for pat in patterns:
            if re.search(pat, cv_text, re.IGNORECASE):
                return canonical

    return "Software Engineer"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Match Score Components
# ──────────────────────────────────────────────────────────────────────────────

def _role_semantic_score(
    cand_domain: str,
    cand_role: str,
    job_domain: str,
    job_title: str,
    bundle,
) -> float:
    """
    Score how semantically related the candidate's role is to the job title.
    Returns 0.0–1.0.
    """
    domain_c = _domain_compat(cand_domain, job_domain)

    # Title TF-IDF cosine similarity via the trained vectorizer
    try:
        q = bundle.tfidf_title.transform([cand_role.lower()])
        j = bundle.tfidf_title.transform([job_title.lower()])
        title_sim = float((q @ j.T).toarray().ravel()[0])
    except Exception:
        title_sim = 0.0

    # Word-token overlap (role tokens)
    cand_tok = set(re.findall(r"[a-z]+", cand_role.lower()))
    job_tok  = set(re.findall(r"[a-z]+", job_title.lower()))
    stop = {"engineer", "senior", "junior", "lead", "associate", "manager",
            "specialist", "head", "of", "and", "at", "for", "with", "in"}
    cand_tok -= stop
    job_tok  -= stop
    tok_overlap = (
        len(cand_tok & job_tok) / max(len(cand_tok | job_tok), 1)
    )

    return 0.55 * domain_c + 0.25 * title_sim + 0.20 * tok_overlap


def _skill_score(
    cand_skills: set[str],
    job_skills: set[str],
    domain_compat: float,
) -> tuple[float, float]:
    """
    Score technical skill overlap between candidate and job.
    Returns (score 0–1, skill_coverage_pct 0–100).

    Crucially: single-skill jobs penalised unless domain matches.
    """
    n_job = len(job_skills)
    shared = cand_skills & job_skills

    if n_job == 0:
        # No skills listed for job — rely on domain & text similarity
        score = 0.50 if domain_compat >= 0.7 else 0.12
        return score, 0.0

    if n_job == 1:
        # Single required skill — cannot be the primary signal!
        if shared:
            # Only award real points if domain matches
            score = 0.38 * domain_compat
        else:
            score = 0.0
        return score, 50.0 * float(bool(shared))

    # Multiple required skills
    coverage = len(shared) / n_job
    union = cand_skills | job_skills
    jaccard = len(shared) / len(union) if union else 0.0

    # Domain gates the skill score — prevents irrelevant domain
    # matches from looking good just because of a shared skill.
    base = 0.70 * coverage + 0.30 * jaccard
    gated = base * (0.25 + 0.75 * domain_compat)

    return gated, round(coverage * 100, 1)


def _experience_fit(
    cand_exp: int,
    cand_seniority: str,
    job_blob: str,
) -> float:
    """Score experience & seniority alignment. Returns 0.0–1.0."""
    req_years = extract_years_experience(job_blob) or 0
    if req_years == 0 or cand_exp >= req_years:
        exp_fit = 1.0
    else:
        exp_fit = max(0.4, 1.0 - 0.18 * (req_years - cand_exp))

    job_seniority = extract_seniority(job_blob, req_years)
    sen_rank = {"Junior": 0, "Mid": 1, "Senior": 2}
    sen_diff = abs(
        sen_rank.get(cand_seniority, 1) - sen_rank.get(job_seniority, 1)
    )
    sen_fit = 1.0 if sen_diff == 0 else (0.80 if sen_diff == 1 else 0.50)

    return 0.60 * exp_fit + 0.40 * sen_fit


def _text_sim(cand_text: str, job_blob: str, bundle) -> float:
    """Return TF-IDF cosine similarity between CV and job description texts."""
    try:
        cv_vec  = bundle.tfidf_text.transform([cand_text])
        job_vec = bundle.tfidf_text.transform([preprocess_text(job_blob)])
        return float((cv_vec @ job_vec.T).toarray().ravel()[0])
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 5. Match Explanation
# ──────────────────────────────────────────────────────────────────────────────

def _generate_why(
    is_matched: bool,
    cand_domain: str,
    job_domain: str,
    domain_compat: float,
    shared_skills: list[str],
    missing_skills: list[str],
    exp_fit: float,
) -> str:
    if not is_matched:
        if domain_compat <= 0.15:
            return (
                f"Domain mismatch: candidate is {cand_domain}, "
                f"role is {job_domain}. "
                "Incidental keyword overlap does not indicate qualification."
            )
        if not shared_skills:
            m = (", ".join(missing_skills[:3]) + ("…" if len(missing_skills) > 3 else "")) or "different stack"
            return f"Low skill overlap — requires {m}."
        return "Partial alignment but overall fit is insufficient."

    reasons: list[str] = []
    if domain_compat >= 0.90:
        reasons.append(f"strong role & domain alignment ({job_domain})")
    elif domain_compat >= 0.65:
        reasons.append(f"related discipline ({job_domain})")

    if shared_skills:
        top = ", ".join(shared_skills[:5])
        reasons.append(f"shared skills: {top}")

    if exp_fit >= 0.90:
        reasons.append("satisfies experience requirements")
    elif exp_fit >= 0.70:
        reasons.append("near-match on experience requirements")

    return " • ".join(reasons) if reasons else "Overall profile aligns with role."


# ──────────────────────────────────────────────────────────────────────────────
# 6. Candidate Profile Builder (extended, used by app.py instead of bundle's)
# ──────────────────────────────────────────────────────────────────────────────

def build_extended_candidate_profile(cv_text: str, bundle) -> dict:
    """
    Build a candidate profile using the EXTENDED skill extractor and
    role detector — replaces bundle.build_candidate_profile() for the app.

    Falls back to bundle's original profile values for seniority/education/etc.
    """
    extract_skills = _build_extended_extractor(bundle)
    years = extract_years_experience(cv_text)
    role  = extract_cv_role(cv_text)

    return {
        "role":             role,
        "seniority":        extract_seniority(cv_text, years),
        "years_experience": years,
        "industry":         extract_industry(cv_text),
        "education":        extract_education(cv_text),
        "skills":           extract_skills(cv_text),
        "titles":           [role],
        "text":             preprocess_text(cv_text),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. Main Matching Function (drop-in replacement for match_cv_to_jobs)
# ──────────────────────────────────────────────────────────────────────────────

def semantic_match_cv_to_jobs(
    candidate: dict,
    cleaned_jobs: list[dict],
    bundle,
) -> list[dict]:
    """
    Match one candidate against all live jobs using holistic semantic scoring.

    Scoring formula (weights):
        35%  Role & domain semantic compatibility
        35%  Technical skill overlap (gated by domain compatibility)
        15%  Full-text (TF-IDF) similarity
        15%  Experience & seniority fit

    Domain compatibility gates the skill component so a single incidental
    shared skill in a mismatched domain cannot elevate the overall score.

    Returns a list of result dicts sorted by match_percentage descending.

    Result schema is a superset of match_cv_to_jobs() so the same UI works.
    """
    extract_skills = _build_extended_extractor(bundle)

    cand_role    = candidate.get("role", "Software Engineer")
    cand_domain  = _get_domain(cand_role)
    cand_skills  = set(candidate.get("skills", []))
    cand_exp     = int(candidate.get("years_experience", 0))
    cand_seniority = candidate.get("seniority", "Mid")
    cand_text    = candidate.get("text", "")

    results: list[dict] = []

    for i, job in enumerate(cleaned_jobs):
        job_title = job.get("title", "")
        job_slug  = job.get("slug", f"index-{i}")
        job_desc  = job.get("description", "")
        job_tags  = job.get("tags", [])
        job_blob  = f"{job_title} {job_desc} {' '.join(job_tags)}"

        try:
            job_skills  = set(extract_skills(job_blob))
            job_domain  = _get_domain(job_title)
            dom_c       = _domain_compat(cand_domain, job_domain)

            # ── Component scores ─────────────────────────────────────
            role_s   = _role_semantic_score(cand_domain, cand_role, job_domain, job_title, bundle)
            sk_s, sk_cov = _skill_score(cand_skills, job_skills, dom_c)
            txt_s    = _text_sim(cand_text, job_blob, bundle)
            exp_s    = _experience_fit(cand_exp, cand_seniority, job_blob)

            raw = (
                0.35 * role_s +
                0.35 * sk_s  +
                0.15 * txt_s +
                0.15 * exp_s
            )

            # Hard cap for cross-domain matches (prevents Sales job > 20% for AI candidate)
            if dom_c <= 0.15:
                match_pct = round(min(raw * dom_c * 100, 18.0), 1)
            else:
                match_pct = round(min(max(raw * 100, 0.0), 98.0), 1)

            is_matched = match_pct >= 50.0

            # Confidence: how decisively the score is on one side of the threshold
            margin = abs(match_pct - 50.0) / 50.0          # 0–1
            confidence = round(min(0.40 + 0.55 * margin, 0.95), 2)

            shared_skills  = sorted(cand_skills & job_skills)
            missing_skills = sorted(job_skills - cand_skills)
            why = _generate_why(
                is_matched, cand_domain, job_domain, dom_c,
                shared_skills, missing_skills, exp_s,
            )

            results.append({
                # ── Identity ─────────────────────────────────────────
                "job_id":     job_slug,
                "title":      job_title,
                "company":    job.get("company", ""),
                "location":   job.get("location", ""),
                "remote":     job.get("remote", False),
                "job_type":   ", ".join(job.get("job_types", [])) or "Not specified",
                "url":        job.get("url", ""),
                "created_at": job.get("created_at"),

                # ── Match scores (from this module, NOT raw ML model) ─
                "prediction":        int(is_matched),
                "prediction_label":  "MATCHED" if is_matched else "NOT MATCHED",
                "matched":           is_matched,
                "match_probability": round(match_pct / 100, 4),
                "match_prob_pct":    match_pct,
                "confidence":        confidence,
                "confidence_pct":    round(confidence * 100, 1),

                # ── Skill analysis ────────────────────────────────────
                "shared_skills":  shared_skills,
                "missing_skills": missing_skills,
                "skill_coverage": sk_cov,

                # ── Explanation ───────────────────────────────────────
                "match_reason":  why,
                "candidate_domain": cand_domain,
                "job_domain":       job_domain,
                "domain_compat":    round(dom_c, 2),

                # ── Candidate profile summary ─────────────────────────
                "candidate_role":       cand_role,
                "candidate_seniority":  cand_seniority,
                "candidate_experience": cand_exp,
                "candidate_education":  candidate.get("education", ""),
                "candidate_industry":   candidate.get("industry", "Unknown"),

                # ── Job profile summary ───────────────────────────────
                "job_canonical_title": bundle.canonical_role(job_title),
                "job_seniority":       extract_seniority(job_title, None),
                "job_industry":        extract_industry(job_blob, job_tags),
                "job_required_years":  extract_years_experience(job_blob) or None,
                "job_skills":          sorted(job_skills),

                # ── No legacy debug block needed ──────────────────────
                "_debug": None,
            })

        except Exception as exc:
            logger.warning("Skipping job [%s – %s]: %s", job_slug, job_title, exc)
            continue

    results.sort(key=lambda r: r["match_probability"], reverse=True)

    matched_count = sum(1 for r in results if r["matched"])
    logger.info(
        "Semantic matching complete: %d jobs evaluated, %d matched (≥50%%)",
        len(results), matched_count,
    )
    return results

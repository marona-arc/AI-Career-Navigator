"""
NLP utilities — exact port of the preprocessing functions from final_project.ipynb.

These functions are the single source of truth for text cleaning, skill extraction,
education/seniority/experience parsing, and role canonicalization.
They must never diverge from the notebook implementation.
"""

import re
import datetime
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# ──────────────────────────────────────────────────────────────────────────────
# Text cleaning  (cell 42 of final_project.ipynb)
# ──────────────────────────────────────────────────────────────────────────────

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """Lowercase, keep a-z0-9+#/\s, collapse whitespace. (notebook cell 42)"""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+#/\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list:
    return clean_text(text).split()


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline: clean → tokenize → remove stopwords. (cell 42)"""
    return " ".join(remove_stopwords(tokenize(text)))


# ──────────────────────────────────────────────────────────────────────────────
# Education extraction  (cell 54)
# ──────────────────────────────────────────────────────────────────────────────

EDU_RANK = {
    "High School": 0,
    "BA": 1,
    "BSc": 1,
    "MBA": 2,
    "MSc": 2,
    "PhD": 3,
}


def extract_education(text: str) -> str:
    """Extract highest detected education level. (cell 54)"""
    raw = str(text)
    low = raw.lower()
    found = []

    if re.search(r"\bph\.?d\b|doctorate|doctoral", low):
        found.append("PhD")
    if re.search(r"\bmba\b|master of business", low):
        found.append("MBA")
    if re.search(
        r"\bm\.?sc\b|master of science|(?<!scrum )master's|\bmasters\b|master of (?!business)",
        low,
    ):
        found.append("MSc")
    if re.search(
        r"\bb\.?sc\b|bachelor of science|bachelor(?! of arts)|undergraduate degree",
        low,
    ):
        found.append("BSc")
    if re.search(r"\bB\.?A\b", raw) or "bachelor of arts" in low:
        found.append("BA")
    if re.search(r"high school|secondary school|thanaweya", low):
        found.append("High School")

    return max(found, key=EDU_RANK.get) if found else "Unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Years of experience extraction  (cell 52)
# ──────────────────────────────────────────────────────────────────────────────


def extract_years_experience(text: str) -> int:
    """Parse years of experience from free text. (cell 52)"""
    text = str(text).lower()

    explicit = [
        int(m)
        for m in re.findall(
            r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)"
            r"\b(?:\s+of)?(?:\s+\w+){0,3}?\s+experience",
            text,
        )
    ]
    explicit += [
        int(m)
        for m in re.findall(
            r"experience\s*(?:of|:)?\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)",
            text,
        )
    ]

    if explicit:
        return min(max(explicit), 40)

    covered: set = set()
    current_year = datetime.date.today().year

    for start, end in re.findall(
        r"\b((?:19|20)\d{2})\s*(?:-|to|–|—)\s*((?:19|20)\d{2}|present|current|now)\b",
        text,
    ):
        start_year = int(start)
        end_year = (
            current_year
            if end in ("present", "current", "now")
            else int(end)
        )
        if 0 <= end_year - start_year <= 40:
            covered.update(range(start_year, max(end_year, start_year + 1)))

    return min(len(covered), 40)


# ──────────────────────────────────────────────────────────────────────────────
# Seniority extraction  (cell 56)
# ──────────────────────────────────────────────────────────────────────────────


def years_to_seniority(years: int) -> str:
    """Map years of experience to seniority level. (cell 56)"""
    if years < 2:
        return "Junior"
    if years < 6:
        return "Mid"
    return "Senior"


def extract_seniority(text: str, years: int = None) -> str:
    """Extract seniority from text keywords; fall back to years. (cell 56)"""
    low = str(text).lower()

    if re.search(
        r"\b(senior|sr\.?|lead|principal|staff|director|vp|head of)\b",
        low,
    ):
        return "Senior"
    if re.search(
        r"\b(junior|jr\.?|intern|internship|trainee|entry[- ]level|graduate)\b",
        low,
    ):
        return "Junior"
    if re.search(r"\b(mid[- ]level|intermediate)\b", low):
        return "Mid"

    return years_to_seniority(years) if years is not None else "Mid"


# ──────────────────────────────────────────────────────────────────────────────
# Skill extraction  (cells 46-50)
# These functions need the SKILL_VOCAB / SKILL_ALIASES from the saved artifacts.
# They are wired up at import time in artifacts.py.
# ──────────────────────────────────────────────────────────────────────────────


def build_skill_extractor(skill_vocab: list, skill_aliases: dict):
    """
    Build and return an extract_skills() function from the saved vocabulary.
    Mirrors the notebook's cells 48-50 exactly.
    """
    alias_to_skill = {skill.lower(): skill for skill in skill_vocab}
    for alias, target in skill_aliases.items():
        if target in skill_vocab:
            alias_to_skill[alias.lower()] = target

    alias_patterns_sorted = sorted(
        alias_to_skill.keys(), key=len, reverse=True
    )

    alias_regex = [
        (
            re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)"),
            alias_to_skill[alias],
        )
        for alias in alias_patterns_sorted
    ]

    def extract_skills(text: str) -> list:
        """Return sorted list of canonical skills found in text. (cell 50)"""
        text = str(text).lower()
        found = []
        for pattern, skill in alias_regex:
            if pattern.search(text):
                found.append(skill)
                text = pattern.sub(" ", text)
        return sorted(set(found))

    return extract_skills


# ──────────────────────────────────────────────────────────────────────────────
# Role canonicalization  (cell 58)
# Requires KNOWN_ROLES from the saved artifacts.
# ──────────────────────────────────────────────────────────────────────────────


def build_role_canonicalizer(known_roles: list):
    """
    Build and return canonical_role() and extract_job_titles() from saved roles.
    Mirrors the notebook's cell 58 exactly.
    """
    role_vec = TfidfVectorizer(ngram_range=(1, 2))
    role_matrix = role_vec.fit_transform(
        [clean_text(role) for role in known_roles]
    )

    def canonical_role(title: str, threshold: float = 0.35) -> str:
        if not str(title).strip():
            return "Other"
        query = role_vec.transform([clean_text(title)])
        similarities = (role_matrix @ query.T).toarray().ravel()
        best_idx = similarities.argmax()
        if similarities[best_idx] >= threshold:
            return known_roles[best_idx]
        return "Other"

    def extract_job_titles(text: str) -> list:
        low = str(text).lower()
        hits = [
            (low.find(role.lower()), role)
            for role in known_roles
            if role.lower() in low
        ]
        return [role for _, role in sorted(hits)]

    return canonical_role, extract_job_titles


# ──────────────────────────────────────────────────────────────────────────────
# Industry extraction
# Identifies candidate / job industry domain from text and tags.
# ──────────────────────────────────────────────────────────────────────────────

KNOWN_INDUSTRIES = [
    ("FinTech", [r"\bfintech\b", r"\bfinancial\s+technology\b"]),
    ("Finance", [r"\bfinance\b", r"\bfinancial\b", r"\bbanking\b", r"\baccounting\b", r"\baudit\b"]),
    ("Healthcare", [r"\bhealthcare\b", r"\bhealth\b", r"\bmedical\b", r"\bpharma\b", r"\bbiotech\b", r"\bclinical\b"]),
    ("E-Commerce", [r"\be-?commerce\b", r"\bonline\s+retail\b", r"\bmarketplace\b", r"\bshop\b"]),
    ("Gaming", [r"\bgaming\b", r"\bgames?\s+development\b", r"\bvideo\s+games?\b"]),
    ("Technology", [r"\btechnology\b", r"\bsoftware\b", r"\btech\b", r"\bit\s+services\b", r"\bcloud\b", r"\bcybersecurity\b"]),
    ("Retail", [r"\bretail\b", r"\bfashion\b", r"\bconsumer\s+goods\b", r"\bfmcg\b"]),
    ("Education", [r"\beducation\b", r"\bedtech\b", r"\blearning\b", r"\be-?learning\b", r"\buniversity\b"]),
    ("Automotive", [r"\bautomotive\b", r"\bvehicles?\b", r"\bmobility\b", r"\bcars?\b"]),
    ("Consulting", [r"\bconsulting\b", r"\badvisory\b", r"\bmanagement\s+consulting\b"]),
    ("Telecommunications", [r"\btelecommunications?\b", r"\btelecom\b", r"\bnetwork\s+provider\b"]),
    ("Manufacturing", [r"\bmanufacturing\b", r"\bindustrial\b", r"\bfactory\b", r"\bproduction\b"]),
    ("Media", [r"\bmedia\b", r"\bpublishing\b", r"\bbroadcasting\b", r"\bentertainment\b"]),
    ("Human Resources", [r"\bhuman\s+resources\b", r"\bhr\b", r"\brecruiting\b", r"\btalent\s+acquisition\b"]),
]


def extract_industry(text: str, tags: list[str] = None) -> str:
    """Extract recognized industry domain from text and optional tags."""
    combined = (str(text) + " " + " ".join(tags or [])).lower()
    for ind_name, patterns in KNOWN_INDUSTRIES:
        for p in patterns:
            if re.search(p, combined):
                return ind_name
    return "Unknown"


"""
Export script — reproduces the training pipeline from final_project.ipynb
up to cell 139 to generate:
  - models/ai_resume_job_match_model.joblib (and root)
  - models/ai_resume_job_match_nlp_artifacts.joblib (and root)

Uses the exact same logic, splits, parameters, and random seeds (RANDOM_STATE = 42).
"""

import sys
import re
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

print("Starting export pipeline using final_project.ipynb logic...")

RANDOM_STATE = 42
data_path = Path("candidate_job_pairs.csv")
if not data_path.exists():
    data_path = Path("merged_data/candidate_job_pairs.csv")
if not data_path.exists():
    raise FileNotFoundError("Could not find candidate_job_pairs.csv")

print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)
df = df.drop_duplicates()

# ──────────────── Cell 15: must_have_overlap for noise detection ────────────────
def overlap(r, j):
    j = set(str(j).split("|"))
    return len(set(str(r).split("|")) & j) / len(j) if len(j) else 0.0

df["must_have_overlap"] = [
    overlap(r, j) for r, j in zip(df["r_skills"], df["j_must_have_skills"])
]

# ──────────────── Cell 31: Remove noisy rows ────────────────
noise_mask = (df["label"] == 0) & (df["must_have_overlap"] >= 0.6)
df_clean = df[~noise_mask].reset_index(drop=True)
print(f"Cleaned data shape: {df_clean.shape}")

# ──────────────── Cell 33: Drop leaking & constant columns ────────────────
df_clean = df_clean.drop(
    columns=[
        "skill_group",
        "must_have_overlap",
        "role_match",
        "seniority_match",
        "industry_match",
        "r_experience_bullets",
        "j_responsibilities",
        "j_requirements",
    ],
    errors="ignore"
)

# ──────────────── Cell 35-36: Split pipe skills ────────────────
def split_pipe(x):
    if isinstance(x, str) and x != "":
        return x.split("|")
    return []

list_cols = ["r_skills", "j_must_have_skills", "j_nice_to_have_skills"]
for col in list_cols:
    df_clean[col] = df_clean[col].map(split_pipe)

# ──────────────── Cell 42: Text preprocessing ────────────────
STOP_WORDS = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+#/\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    return clean_text(text).split()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

def preprocess_text(text):
    return " ".join(remove_stopwords(tokenize(text)))

# ──────────────── Cell 44: Skill vocabulary ────────────────
all_skill_lists = pd.concat([
    df_clean["r_skills"],
    df_clean["j_must_have_skills"],
    df_clean["j_nice_to_have_skills"],
])

SKILL_VOCAB = sorted({
    skill
    for skill_list in all_skill_lists
    for skill in skill_list
})
print(f"Total skills in vocabulary: {len(SKILL_VOCAB)}")

# ──────────────── Cell 46: Skill aliases ────────────────
SKILL_ALIASES = {
    "js": "JavaScript",
    "node.js": "JavaScript",
    "nodejs": "JavaScript",
    "react": "JavaScript",
    "typescript": "JavaScript",
    "postgresql": "Databases",
    "mysql": "Databases",
    "mongodb": "Databases",
    "database": "Databases",
    "nosql": "Databases",
    "restful": "REST APIs",
    "restful api": "REST APIs",
    "restful apis": "REST APIs",
    "rest api": "REST APIs",
    "rest apis": "REST APIs",
    "object oriented programming": "OOP",
    "object-oriented programming": "OOP",
    "unit tests": "Unit Testing",
    "pytest": "Unit Testing",
    "junit": "Unit Testing",
    "continuous integration": "CI/CD",
    "continuous deployment": "CI/CD",
    "containers": "Docker",
    "containerization": "Docker",
    "github": "Git",
    "gitlab": "Git",
    "version control": "Git",
    "microsoft excel": "Excel",
    "powerbi": "Power BI",
    "data viz": "Data Visualization",
    "dashboards": "Data Visualization",
    "a/b test": "A/B Testing",
    "kanban": "Agile",
    "sprint planning": "Agile",
    "salesforce": "CRM",
    "hubspot": "CRM",
    "google adwords": "Google Ads",
    "ppc": "Google Ads",
    "facebook ads": "Meta Ads",
    "instagram ads": "Meta Ads",
    "product requirements": "PRD",
    "product requirements document": "PRD",
    "financial models": "Financial Modeling",
    "rca": "Root Cause Analysis",
    "service level agreement": "SLA",
    "cold outreach": "Outbound Outreach",
    "cold calling": "Outbound Outreach",
    "help desk": "Ticketing",
    "issue tracking": "Ticketing",
    "kpi": "KPIs",
    "data analytics": "Analytics",
    "google analytics": "Analytics",
    "extract transform load": "ETL",
    "budget": "Budgeting",
    "forecast": "Forecasting",
    "roadmapping": "Roadmap",
    "product roadmap": "Roadmap",
}

# ──────────────── Cell 58: Known roles ────────────────
KNOWN_ROLES = sorted(
    set(df_clean["r_role"]) | set(df_clean["j_job_title"])
)
print(f"Total known roles: {len(KNOWN_ROLES)}")

# ──────────────── Cell 64-65: Candidate & Job profiles ────────────────
resumes = (
    df.drop_duplicates("resume_id")
    [
        [
            "resume_id",
            "r_role",
            "r_seniority",
            "r_years_experience",
            "r_industry",
            "r_education",
            "r_skills",
            "r_summary",
        ]
    ]
    .set_index("resume_id")
)

jobs = (
    df.drop_duplicates("job_id")
    [
        [
            "job_id",
            "j_job_title",
            "j_seniority",
            "j_industry",
            "j_must_have_skills",
            "j_nice_to_have_skills",
            "j_description",
        ]
    ]
    .set_index("job_id")
)

cand_profiles = {
    rid: {
        "role": row.r_role,
        "seniority": row.r_seniority,
        "years_experience": int(row.r_years_experience),
        "industry": row.r_industry,
        "education": row.r_education,
        "skills": split_pipe(row.r_skills),
        "text": preprocess_text(f"{row.r_role} {row.r_summary}"),
    }
    for rid, row in resumes.iterrows()
}

job_profiles = {
    jid: {
        "title": row.j_job_title,
        "seniority": row.j_seniority,
        "industry": row.j_industry,
        "must_have": split_pipe(row.j_must_have_skills),
        "nice_to_have": split_pipe(row.j_nice_to_have_skills),
        "text": preprocess_text(f"{row.j_job_title} {row.j_description}"),
    }
    for jid, row in jobs.iterrows()
}

# ──────────────── Cell 71: Train-Test Split by job_id ────────────────
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(df_clean, groups=df_clean["job_id"]))

train_data = df_clean.iloc[train_idx].reset_index(drop=True)
test_data = df_clean.iloc[test_idx].reset_index(drop=True)
y_train = train_data["label"]

print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# ──────────────── Cell 78-82: Fit TF-IDF Vectorizers ────────────────
train_resume_ids = train_data["resume_id"].unique()
train_job_ids = train_data["job_id"].unique()

train_resume_texts = [cand_profiles[rid]["text"] for rid in train_resume_ids]
train_job_texts = [job_profiles[jid]["text"] for jid in train_job_ids]

print("Fitting tfidf_text...")
tfidf_text = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
tfidf_text.fit(train_resume_texts + train_job_texts)

print("Fitting tfidf_title...")
train_title_texts = [
    cand_profiles[rid]["role"].lower() for rid in train_resume_ids
] + [
    job_profiles[jid]["title"].lower() for jid in train_job_ids
]
tfidf_title = TfidfVectorizer(ngram_range=(1, 2))
tfidf_title.fit(train_title_texts)

print("Fitting tfidf_skills...")
train_skill_texts = [
    "|".join(cand_profiles[rid]["skills"]) for rid in train_resume_ids
] + [
    "|".join(job_profiles[jid]["must_have"] + job_profiles[jid]["nice_to_have"])
    for jid in train_job_ids
]
tfidf_skills = TfidfVectorizer(token_pattern=r"[^|]+")
tfidf_skills.fit(train_skill_texts)

# ──────────────── Cell 84: Seniority Lookups ────────────────
SENIORITY_RANK = {"Junior": 0, "Mid": 1, "Senior": 2}
SENIORITY_YEARS = {"Junior": (0, 2), "Mid": (2, 6), "Senior": (6, 12)}

# ──────────────── Cell 86: pair_cosine ────────────────
def pair_cosine(vectorizer, texts_a, texts_b):
    texts_a = list(texts_a)
    texts_b = list(texts_b)
    unique_texts = list(dict.fromkeys(texts_a + texts_b))
    index = {text: i for i, text in enumerate(unique_texts)}
    matrix = vectorizer.transform(unique_texts)
    ia = np.fromiter((index[text] for text in texts_a), dtype=int, count=len(texts_a))
    ib = np.fromiter((index[text] for text in texts_b), dtype=int, count=len(texts_b))
    return np.asarray(matrix[ia].multiply(matrix[ib]).sum(axis=1)).ravel()

# ──────────────── Cell 87: make_features ────────────────
def make_features(cands, jobs):
    cand_texts = [candidate["text"] for candidate in cands]
    job_texts = [job["text"] for job in jobs]
    cand_titles = [candidate["role"].lower() for candidate in cands]
    job_titles = [job["title"].lower() for job in jobs]
    cand_skills_text = ["|".join(candidate["skills"]) for candidate in cands]
    job_skills_text = ["|".join(job["must_have"] + job["nice_to_have"]) for job in jobs]

    text_sim = pair_cosine(tfidf_text, cand_texts, job_texts)
    title_sim = pair_cosine(tfidf_title, cand_titles, job_titles)
    skill_tfidf_sim = pair_cosine(tfidf_skills, cand_skills_text, job_skills_text)

    rows = []
    for i, (candidate, job) in enumerate(zip(cands, jobs)):
        candidate_skills = set(candidate["skills"])
        must_have = set(job["must_have"])
        nice_to_have = set(job["nice_to_have"])

        matching_skills = candidate_skills & must_have
        missing_skills = must_have - candidate_skills
        nice_matching = candidate_skills & nice_to_have
        union = candidate_skills | must_have

        skill_jaccard = len(matching_skills) / len(union) if union else 0
        required_range = SENIORITY_YEARS.get(job["seniority"], (0, 100))
        candidate_years = candidate["years_experience"]
        exp_in_range = int(required_range[0] <= candidate_years <= required_range[1])
        exp_gap = max(0, required_range[0] - candidate_years)

        candidate_seniority = SENIORITY_RANK.get(candidate["seniority"], 1)
        job_seniority = SENIORITY_RANK.get(job["seniority"], 1)

        rows.append({
            "n_must_have": len(must_have),
            "n_matching_skills": len(matching_skills),
            "n_missing_skills": len(missing_skills),
            "must_have_coverage": len(matching_skills) / len(must_have) if must_have else 0,
            "n_nice_matching": len(nice_matching),
            "skill_jaccard": skill_jaccard,
            "n_candidate_skills": len(candidate_skills),
            "skill_tfidf_sim": skill_tfidf_sim[i],
            "years_experience": candidate_years,
            "exp_in_range": exp_in_range,
            "exp_gap": exp_gap,
            "candidate_seniority": candidate_seniority,
            "job_seniority": job_seniority,
            "seniority_match": int(candidate["seniority"] == job["seniority"]),
            "seniority_diff": abs(candidate_seniority - job_seniority),
            "role_match": int(candidate["role"] == job["title"]),
            "industry_match": int(candidate["industry"] == job["industry"]),
            "title_sim": title_sim[i],
            "text_sim": text_sim[i],
        })
    return pd.DataFrame(rows)

# ──────────────── Cell 89: Feature sets ────────────────
SKILL_FEATURES = [
    "n_must_have",
    "n_matching_skills",
    "n_missing_skills",
    "must_have_coverage",
    "n_nice_matching",
    "skill_jaccard",
    "n_candidate_skills",
    "skill_tfidf_sim",
]

NON_SKILL_FEATURES = [
    "years_experience",
    "exp_in_range",
    "exp_gap",
    "candidate_seniority",
    "job_seniority",
    "seniority_match",
    "seniority_diff",
    "role_match",
    "industry_match",
    "title_sim",
    "text_sim",
]

ALL_FEATURES = SKILL_FEATURES + NON_SKILL_FEATURES
FEATURE_SETS = {
    "All Features": ALL_FEATURES,
    "Without Skill Overlap": NON_SKILL_FEATURES,
}

print("Computing train_features...")
train_features = make_features(
    [cand_profiles[r] for r in train_data["resume_id"]],
    [job_profiles[j] for j in train_data["job_id"]],
)
print(f"train_features shape: {train_features.shape}")

# ──────────────── Cell 103: Train XGBoost (selected winning model) ────────────────
print("Training final XGBoost model (matching cell 103 / cell 137)...")
best_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
best_model.fit(train_features[ALL_FEATURES], y_train)

best_features = ALL_FEATURES
best_features_name = "All Features"
best_model_name = "XGBoost"

# ──────────────── Save artifacts to models/ and root ────────────────
out_paths = [
    Path("models"),
    Path("."),
]

for out_dir in out_paths:
    out_dir.mkdir(parents=True, exist_ok=True)
    m_path = out_dir / "ai_resume_job_match_model.joblib"
    nlp_path = out_dir / "ai_resume_job_match_nlp_artifacts.joblib"

    print(f"Saving model artifact to {m_path}...")
    joblib.dump(
        {
            "model": best_model,
            "features": best_features,
            "feature_set_name": best_features_name,
            "model_name": best_model_name,
        },
        m_path,
    )

    print(f"Saving NLP artifacts to {nlp_path}...")
    joblib.dump(
        {
            "tfidf_text": tfidf_text,
            "tfidf_title": tfidf_title,
            "tfidf_skills": tfidf_skills,
            "SKILL_VOCAB": SKILL_VOCAB,
            "SKILL_ALIASES": SKILL_ALIASES,
            "SKILL_FEATURES": SKILL_FEATURES,
            "NON_SKILL_FEATURES": NON_SKILL_FEATURES,
            "ALL_FEATURES": ALL_FEATURES,
            "KNOWN_ROLES": KNOWN_ROLES,
        },
        nlp_path,
    )

print("Export completed successfully!")

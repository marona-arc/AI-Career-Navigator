import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import pandas as pd
from src.artifacts import ArtifactBundle
from src.cv_extractor import extract_cv_profile
from src.job_api import get_jobs
from src.job_cleaner import clean_jobs
from src.feature_engineering import make_features

bundle = ArtifactBundle()
cv_pdf = Path("data/sample_cv.pdf").read_bytes()
cand = extract_cv_profile(cv_pdf, bundle)

raw_jobs = get_jobs()
cleaned_jobs = clean_jobs(raw_jobs)

job_profiles = [
    bundle.build_job_profile(
        title=j["title"],
        description=j["description"],
        tags=j.get("tags", [])
    )
    for j in cleaned_jobs[:20]
]

features_df = make_features([cand]*len(job_profiles), job_profiles, bundle)
X = features_df[bundle.features]
probs = bundle.model.predict_proba(X)[:, 1]
preds = bundle.model.predict(X)

print(f"X shape: {X.shape}")
print(f"X dtypes:\n{X.dtypes}")

for i in range(min(10, len(job_profiles))):
    print(f"\n--- Job {i}: {job_profiles[i]['raw_title']} ---")
    print(f"  Canonical role: {job_profiles[i]['title']}")
    print(f"  Seniority: {job_profiles[i]['seniority']}")
    print(f"  Must-have: {job_profiles[i]['must_have']}")
    print(f"  Pred: {preds[i]} | Prob[1]: {probs[i]:.6f}")
    print("  Key features:")
    for col in bundle.features:
        print(f"    {col:20s}: {X.iloc[i][col]}")

import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')
import joblib
import pandas as pd
import numpy as np

m = joblib.load('models/ai_resume_job_match_model.joblib')
model = m['model']
features = m['features']

# Let's test a variety of (candidate, job) features
# Vary must_have_coverage from 0.0 to 1.0 in steps of 0.1
rows = []
for cov in [0.0, 0.2, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0]:
    n_must = 5
    n_match = round(cov * n_must)
    n_miss = n_must - n_match
    jaccard = n_match / (10 + n_miss)
    row = {f: 0.0 for f in features}
    row['n_must_have'] = n_must
    row['n_matching_skills'] = n_match
    row['n_missing_skills'] = n_miss
    row['must_have_coverage'] = cov
    row['skill_jaccard'] = jaccard
    row['n_candidate_skills'] = 10
    row['years_experience'] = 3
    row['candidate_seniority'] = 1
    row['job_seniority'] = 1
    row['seniority_match'] = 1
    row['role_match'] = 1
    row['title_sim'] = 0.8
    row['text_sim'] = 0.5
    rows.append(row)

df = pd.DataFrame(rows)[features]
probs = model.predict_proba(df)[:, 1]
preds = model.predict(df)

for i, r in enumerate(rows):
    cov = r['must_have_coverage']
    print(f"Coverage: {cov:4.2f} (Match: {r['n_matching_skills']}/{r['n_must_have']}) | Pred: {preds[i]} | Prob[1]: {probs[i]:.6f}")

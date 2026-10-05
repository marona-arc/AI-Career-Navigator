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

rows = []
covs = np.linspace(0.50, 0.62, 25)
for cov in covs:
    n_must = 10
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

for i, cov in enumerate(covs):
    print(f"Coverage: {cov:6.4f} | Prob[1]: {probs[i]:.6f}")

import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from src.artifacts import ArtifactBundle
from src.cv_extractor import extract_cv_profile
from src.job_api import get_jobs
from src.job_cleaner import clean_jobs
from src.matcher import match_cv_to_jobs

bundle = ArtifactBundle()
cv_pdf = Path("data/sample_cv.pdf").read_bytes()
cand = extract_cv_profile(cv_pdf, bundle)

print("Candidate:")
for k, v in cand.items():
    if k != 'text':
        print(f"  {k}: {v}")

print("\nFetching Arbeitnow jobs...")
raw_jobs = get_jobs()
cleaned_jobs = clean_jobs(raw_jobs)
print(f"Cleaned {len(cleaned_jobs)} jobs.")

results = match_cv_to_jobs(cand, cleaned_jobs, bundle)
print(f"Total results: {len(results)}")
print(f"Matched count: {sum(1 for r in results if r['matched'])}")

print("\nTop 15 results:")
for r in results[:15]:
    print(f"Title: {r['title'][:35]:35s} | Matched: {str(r['matched']):5s} | Conf: {r['confidence_pct']:5.1f}% | Cov: {r['skill_coverage']:5.1f}% | Shared: {r['shared_skills']} | Missing: {r['missing_skills']}")

print("\nBottom 10 results:")
for r in results[-10:]:
    print(f"Title: {r['title'][:35]:35s} | Matched: {str(r['matched']):5s} | Conf: {r['confidence_pct']:5.1f}% | Cov: {r['skill_coverage']:5.1f}% | Shared: {r['shared_skills']} | Missing: {r['missing_skills']}")

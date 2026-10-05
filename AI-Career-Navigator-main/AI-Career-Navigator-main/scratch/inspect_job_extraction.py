import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')
from src.job_api import get_jobs
from src.job_cleaner import clean_jobs
from src.artifacts import ArtifactBundle

bundle = ArtifactBundle()
raw_jobs = get_jobs()
cleaned = clean_jobs(raw_jobs)

print("Sample 10 cleaned Arbeitnow jobs:")
for j in cleaned[:10]:
    title = j['title']
    desc = j['description']
    tags = j.get('tags', [])
    jp = bundle.build_job_profile(title, desc, tags=tags)
    print(f"Title: {title}")
    print(f"  Tags: {tags}")
    print(f"  Extracted must_have: {jp['must_have']}")
    print(f"  Required years: {jp['required_years']}")
    print(f"  Seniority: {jp['seniority']}")
    print(f"  Education required: {jp['education_required']}")
    print(f"  Desc length: {len(desc)} chars")
    print()

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
j = cleaned[1] # Senior Machine Learning Engineer
print("Title:", j['title'])
print("Tags:", j['tags'])
print("Description snippet:")
print(j['description'][:1000])

skills = bundle.extract_skills(j['description'])
print("\nExtracted skills:", skills)

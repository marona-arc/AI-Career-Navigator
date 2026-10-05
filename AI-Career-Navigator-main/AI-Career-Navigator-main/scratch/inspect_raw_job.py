import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')
from src.job_api import get_jobs

jobs = get_jobs()
print(f"Total raw jobs fetched: {len(jobs)}")
sample = jobs[0]
for k, v in sample.items():
    if k == 'description':
        print(f"  {k}: {str(v)[:150]}...")
    else:
        print(f"  {k}: {repr(v)}")

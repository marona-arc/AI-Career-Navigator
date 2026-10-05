import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')
from src.artifacts import ArtifactBundle
from src.feature_engineering import make_features

bundle = ArtifactBundle()

# Candidate Profile
cv_text = """
Alex Morgan
Data Analyst with 3 years of experience in data analysis, business intelligence, and predictive modeling.
Education: Bachelor of Science in Computer Science.
Industry: Technology
Skills: Python, SQL, Pandas, Data Visualization, Statistics, Power BI, Excel.
Experience:
Analyzed large datasets using SQL and Python. Created dashboards in Power BI and Excel.
Conducted statistical hypothesis testing and built predictive analysis reports.
"""

candidate = bundle.build_candidate_profile(cv_text)
print("Candidate Profile:")
for k, v in candidate.items():
    if k != 'text':
        print(f"  {k}: {v}")

jobs_data = [
    {
        "title": "Data Scientist",
        "description": "We are seeking a Data Scientist with 3+ years experience. Required skills: Python, SQL, Statistics, Machine Learning, Data Visualization. Bachelor's degree in Computer Science or related field required. Technology industry.",
        "tags": ["Technology", "Data Science"]
    },
    {
        "title": "Data Analyst",
        "description": "Looking for a Data Analyst with 2-4 years of experience. Must have strong SQL, Excel, Power BI, Python, and Reporting skills. Experience with data visualization and KPI reporting.",
        "tags": ["Technology", "Analytics"]
    },
    {
        "title": "Backend Engineer",
        "description": "Senior Backend Engineer with 5+ years of experience in Java, Spring Boot, Microservices, Kubernetes, Docker, and CI/CD. Degree in Computer Science.",
        "tags": ["Technology", "Software Engineering"]
    },
    {
        "title": "Network Engineer",
        "description": "Network Engineer with 3+ years experience in Cisco routing, switching, firewalls, network protocols, troubleshooting, and infrastructure support.",
        "tags": ["Telecommunications", "IT"]
    },
    {
        "title": "Graphic Designer",
        "description": "Creative Graphic Designer needed with expertise in Adobe Photoshop, Illustrator, Figma, brand design, typography, and visual assets creation. Design portfolio required.",
        "tags": ["Design", "Creative"]
    }
]

print("\n--- Diagnostic Test Results ---")
print(f"{'Job Title':22s} | {'Pred':4s} | {'Probability':12s} | {'Coverage':9s} | {'Shared / Total Skills'}")
print("-" * 75)

for jd in jobs_data:
    jp = bundle.build_job_profile(jd["title"], jd["description"], tags=jd["tags"])
    feats = make_features([candidate], [jp], bundle)
    X = feats[bundle.features]
    prob = float(bundle.model.predict_proba(X)[0, 1])
    pred = int(bundle.model.predict(X)[0])
    
    cand_s = set(candidate["skills"])
    job_s = set(jp["must_have"])
    shared = cand_s & job_s
    cov = len(shared) / len(job_s) if job_s else 0.0
    
    print(f"{jd['title']:22s} | {pred:4d} | {prob:12.4f} | {cov*100:8.1f}% | {len(shared)}/{len(job_s)} {sorted(shared)}")

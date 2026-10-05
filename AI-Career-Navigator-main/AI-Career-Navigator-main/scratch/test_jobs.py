import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')
from src.artifacts import ArtifactBundle
from src.feature_engineering import make_features

bundle = ArtifactBundle()
print("Model:", bundle.model_name)

cand_text = """Alex Smith
Data Scientist with 3 years of experience in Python, SQL, Machine Learning, Scikit-learn, Pandas.
Education: Bachelor of Science in Computer Science.
Experience:
Built machine learning predictive models using scikit-learn and pandas.
Designed SQL queries and data analysis pipelines in Python.
"""

candidate = bundle.build_candidate_profile(cand_text)
print("\nExtracted Candidate Profile:")
for k, v in candidate.items():
    if k != "text":
        print(f"  {k}: {v}")

test_jobs = [
    ("Data Scientist", "Looking for a Data Scientist with 3 years experience. Requirements: Python, SQL, Machine Learning, Scikit-learn, Pandas.", ["Python", "SQL"]),
    ("Data Analyst", "Data Analyst position. 2 years experience with SQL, Python, Excel, Power BI, data analysis.", ["SQL", "Power BI"]),
    ("Backend Engineer", "Senior Backend Engineer with 5+ years experience in Java, Spring Boot, microservices, Kubernetes, Docker, AWS.", ["Java", "Docker"]),
    ("Network Engineer", "Network Engineer needed with CCNA, Cisco routing, switching, firewalls, network protocols, 3+ years experience.", ["Cisco"]),
    ("Graphic Designer", "Graphic Designer required with Figma, Photoshop, Illustrator, visual branding, typography.", ["Figma"]),
]

print("\n--- Diagnostic Results ---")
for title, desc, tags in test_jobs:
    jp = bundle.build_job_profile(title, desc, tags=tags)
    feats = make_features([candidate], [jp], bundle)
    X = feats[bundle.features]
    prob = bundle.model.predict_proba(X)[0]
    pred = bundle.model.predict(X)[0]
    must_have = jp["must_have"]
    print(f"Job: {title:18s} | Pred: {pred} | Prob[1]: {prob[1]:.4f} | Skills: {must_have}")

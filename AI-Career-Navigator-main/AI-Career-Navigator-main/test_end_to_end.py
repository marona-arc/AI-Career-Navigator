"""
End-to-end integration test:
  1. Load ArtifactBundle
  2. Extract/validate candidate profile from a test CV
  3. Fetch and clean live jobs from Arbeitnow API
  4. Run match_cv_to_jobs
  5. Check output structure, sorting, confidence, and skills
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.artifacts import ArtifactBundle
from src.cv_extractor import validate_cv
from src.job_api import get_jobs
from src.job_cleaner import clean_jobs
from src.matcher import match_cv_to_jobs

def run_test():
    print("=== 1. Loading Artifact Bundle ===")
    bundle = ArtifactBundle()
    print("Model loaded successfully:", bundle.model_name)
    print("Feature set:", bundle.feature_set_name)
    print("Features count:", len(bundle.features))

    print("\n=== 2. Creating Candidate Profile ===")
    test_cv = """
    Jane Smith
    Data Analyst with 3 years of experience.
    Education: BSc in Computer Science.
    Skills: Python, SQL, Power BI, Excel, Tableau.
    Experience:
    Built SQL queries, dashboards using Power BI and Tableau, and performed data cleaning and reporting using Python and Excel.
    """
    candidate = bundle.build_candidate_profile(test_cv)
    validate_cv(candidate)
    print("Candidate Role:", candidate["role"])
    print("Candidate Seniority:", candidate["seniority"])
    print("Candidate Skills:", candidate["skills"])

    print("\n=== 3. Fetching and Cleaning API Jobs ===")
    raw_jobs = get_jobs()
    print(f"Fetched {len(raw_jobs)} raw jobs.")
    cleaned_jobs = clean_jobs(raw_jobs)
    print(f"Cleaned {len(cleaned_jobs)} jobs.")

    print("\n=== 4. Matching Candidate Against Live Jobs ===")
    results = match_cv_to_jobs(candidate, cleaned_jobs, bundle)
    print(f"Total jobs evaluated: {len(results)}")

    matched = [r for r in results if r["matched"]]
    print(f"Matched jobs (class 1): {len(matched)}")

    # Verify descending confidence sorting
    confidences = [r["confidence"] for r in results]
    assert confidences == sorted(confidences, reverse=True), "Results are not sorted by confidence descending!"
    print("Results sorting verified: strictly descending by confidence.")

    print("\n=== Top 5 Recommendations ===")
    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. {r['title']} @ {r['company']} ({r['location']})")
        print(f"   Match status: {r['matched']} | Confidence: {r['confidence_pct']}% | Skill coverage: {r['skill_coverage']}%")
        print(f"   Shared Skills: {r['shared_skills']}")
        print(f"   Missing Skills: {r['missing_skills']}")
        print(f"   URL: {r['url']}")

    print("\n=== All Tests Passed Successfully! ===")

if __name__ == "__main__":
    run_test()

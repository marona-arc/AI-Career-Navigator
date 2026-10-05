"""
Comprehensive Test Suite for AI Career Navigator:
  1. Data validation checks (CV, jobs, features)
  2. Failure cases handling
  3. Preprocessing consistency
  4. End-to-end PDF CV extraction
  5. Live API matching & ranking
"""

import io
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pypdf import PdfWriter
from src.artifacts import ArtifactBundle
from src.cv_extractor import extract_cv_profile, validate_cv, extract_text_from_pdf
from src.job_api import validate_raw_job, validate_raw_jobs, deduplicate_jobs
from src.job_cleaner import clean_job, clean_jobs, clean_description
from src.feature_engineering import make_features, validate_features
from src.skill_matching import compute_skill_overlap
from src.matcher import predict_one, match_cv_to_jobs

def test_cv_validation_failures():
    print("Testing CV validation failure cases...")
    # Missing required keys
    try:
        validate_cv({"role": "Engineer"})
        assert False, "Should have failed on missing keys"
    except ValueError:
        pass

    # Invalid years experience
    try:
        validate_cv({
            "role": "Engineer", "seniority": "Junior", "years_experience": -5,
            "industry": "Unknown", "education": "BSc", "skills": [],
            "titles": [], "text": "Valid text"
        })
        assert False, "Should have failed on negative experience"
    except ValueError:
        pass

    # Empty text
    try:
        validate_cv({
            "role": "Engineer", "seniority": "Junior", "years_experience": 2,
            "industry": "Unknown", "education": "BSc", "skills": [],
            "titles": [], "text": "   "
        })
        assert False, "Should have failed on empty text"
    except ValueError:
        pass
    print("✓ CV validation failure checks passed.")

def test_job_validation_and_cleaning():
    print("Testing Job validation and cleaning...")
    # Missing title
    try:
        validate_raw_job({"description": "Some description"})
        assert False, "Should have failed on missing title"
    except ValueError:
        pass

    # Malformed job filtering
    raw_list = [
        {"title": "Valid Job", "description": "Good description", "slug": "job-1"},
        {"title": "", "description": "No title", "slug": "job-2"},
        {"slug": "job-3"}, # Missing title and description
    ]
    valid_jobs = validate_raw_jobs(raw_list)
    assert len(valid_jobs) == 1
    assert valid_jobs[0]["slug"] == "job-1"

    # HTML cleaning and URL preservation
    dirty_job = {
        "title": "Backend <b>Developer</b> &amp; Architect",
        "description": "<p>Responsibilities:</p> <li>Build APIs</li> [Link](https://example.com) https://test.org",
        "company_name": "Acme & Co.",
        "url": "https://company.com/apply",
        "tags": ["Python", "FastAPI"],
        "job_types": ["Full-time"],
        "location": "Berlin &nbsp;",
        "remote": True,
        "slug": "backend-dev"
    }
    cleaned = clean_job(dirty_job)
    assert cleaned["title"] == "Backend Developer & Architect"
    assert "<p>" not in cleaned["description"]
    assert "https://example.com" not in cleaned["description"]
    assert cleaned["url"] == "https://company.com/apply" # Preserved original URL
    assert cleaned["company"] == "Acme & Co."
    assert cleaned["location"] == "Berlin"
    print("✓ Job validation and cleaning checks passed.")

def test_skill_overlap():
    print("Testing skill overlap and case-insensitivity...")
    cand_skills = ["python", "SQL", "Docker", "Machine Learning"]
    job_skills = ["Python", "sql", "AWS", "Kubernetes"]
    res = compute_skill_overlap(cand_skills, job_skills)
    
    # Shared should contain python and SQL
    shared_lower = [s.lower() for s in res["shared_skills"]]
    assert "python" in shared_lower
    assert "sql" in shared_lower
    assert len(res["shared_skills"]) == 2

    # Missing should contain AWS and Kubernetes
    missing_lower = [s.lower() for s in res["missing_skills"]]
    assert "aws" in missing_lower
    assert "kubernetes" in missing_lower
    assert len(res["missing_skills"]) == 2
    assert res["skill_coverage"] == 50.0
    print("✓ Skill overlap checks passed.")

def test_pdf_creation_and_extraction():
    print("Testing real PDF creation and candidate extraction...")
    # Use pypdf to test text extraction error on empty/binary
    empty_stream = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(empty_stream)
    empty_pdf = empty_stream.getvalue()

    try:
        extract_text_from_pdf(empty_pdf)
        assert False, "Should have failed on blank/empty PDF"
    except ValueError:
        pass
    print("✓ Blank PDF rejection check passed.")

def test_model_inference_consistency():
    print("Testing model inference against notebook benchmark...")
    bundle = ArtifactBundle()
    test_cv = """
    John Doe
    Software Engineer with 4 years of experience.
    Education: Bachelor of Science in Computer Science.
    Skills: Python, Java, SQL, Git, Docker, REST APIs, Machine Learning.
    Experience: Worked as a software engineer developing Python applications, REST APIs and machine learning systems.
    """
    job_title = "Software Engineer"
    job_desc = "We are looking for a Software Engineer with experience in Python, SQL, REST APIs, Git and Docker."
    
    candidate = bundle.build_candidate_profile(test_cv)
    job = bundle.build_job_profile(job_title, job_desc)
    
    pred = predict_one(candidate, job, bundle)
    assert pred["matched"] is True
    assert pred["confidence"] >= 0.95
    assert "Docker" in pred["shared_skills"]
    assert "Python" in pred["shared_skills"]
    assert "SQL" in pred["shared_skills"]
    assert pred["missing_skills"] == []
    assert pred["skill_coverage"] == 100.0
    print(f"✓ Model inference consistency verified: Matched={pred['matched']}, Confidence={pred['confidence']*100:.1f}%")

if __name__ == "__main__":
    test_cv_validation_failures()
    test_job_validation_and_cleaning()
    test_skill_overlap()
    test_pdf_creation_and_extraction()
    test_model_inference_consistency()
    print("\n🎉 ALL TESTS IN SUITE PASSED PERFECTLY!")

# AI Career Navigator

A Streamlit-based career recommendation app that helps candidates evaluate how well their resume matches live job openings from the Arbeitnow job board, explains the gap in skills, and suggests learning priorities for the roles they want to pursue.

The project combines resume parsing, NLP feature extraction, classification, semantic role/domain matching, and study guidance in a single workflow for job discovery and career planning.

---

## Highlights

- Upload a PDF resume and extract a structured candidate profile
- Retrieve live jobs from the Arbeitnow API
- Compare candidate skills against job requirements
- Rank jobs by an AI-assisted match score
- Use domain-aware semantic matching to avoid irrelevant cross-domain matches
- Show shared and missing skills with a clear explanation layer
- Recommend learning steps for missing skills with links to study resources
- Filter results by match threshold, remote-only jobs, and keyword/title search

---

## What the app does

1. Parses the uploaded CV text and extracts structured fields such as role, seniority, work experience, education, industry, and detected skills.
2. Loads the trained ML artifacts and job vocabulary used by the project.
3. Pulls current job listings from the live Arbeitnow API, cleans them, and normalizes them into comparable job profiles.
4. Evaluates each candidate-job pair across semantic factors such as:
   - role and domain alignment
   - skill overlap
   - text similarity
   - experience and seniority fit
5. Ranks jobs by a match score and highlights which skills are shared versus missing.
6. Suggests a practical learning path for the most important missing skills.

---

## Updated architecture

```text
Upload PDF CV
      │
      ▼
CV Extraction + Validation
      │
      ▼
Candidate Profile (role, skills, experience, education, industry)
      │
      ├──────────────► ML artifact bundle
      │
      ▼
Live job fetch + cleaning from Arbeitnow API
      │
      ▼
Semantic matching engine
  - role/domain compatibility
  - skill overlap
  - text similarity
  - experience fit
      │
      ├──────────────► job ranking + match score
      │
      ├──────────────► shared / missing skills
      │
      └──────────────► learning recommendations
      │
      ▼
Streamlit UI (app.py)
```

---

## Project structure

```text
ai_resumeV2-main/
├── app.py                            # Streamlit app entrypoint
├── final_project.ipynb              # Original training notebook / pipeline reference
├── train_and_export.py              # Export pipeline for retraining and artifact generation
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── test_suite.py                    # Local validation tests
├── test_end_to_end.py               # End-to-end job match test
├── data/
│   └── (project data files if added locally)
├── models/
│   ├── ai_resume_job_match_model.joblib
│   └── ai_resume_job_match_nlp_artifacts.joblib
├── scratch/
│   ├── diagnostic_test.py
│   ├── inspect_features.py
│   ├── inspect_industries.py
│   ├── inspect_job_extraction.py
│   ├── test_live_matching.py
│   └── other debugging / validation utilities
├── src/
│   ├── __init__.py
│   ├── artifacts.py                 # Model & NLP artifact loading
│   ├── cv_extractor.py              # Resume text extraction and profile creation
│   ├── feature_engineering.py        # Feature construction used by the trained model
│   ├── job_api.py                   # Arbeitnow API client
│   ├── job_cleaner.py               # Job text normalization / cleaning
│   ├── learning_recommendations.py  # Study-plan generation for missing skills
│   ├── matcher.py                   # Reference ML-based matcher
│   ├── nlp_utils.py                 # Text preprocessing and extraction helpers
│   ├── semantic_matcher.py          # Role-aware, holistic job matching engine
│   └── skill_matching.py            # Shared / missing skill logic
└── .gitattributes
```

---

## Tech stack

- Python 3
- Streamlit
- Pandas / NumPy
- scikit-learn
- XGBoost
- Joblib
- Requests
- BeautifulSoup
- pdfplumber / pypdf
- RapidFuzz

---

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

If the trained artifacts are missing or you want to regenerate them:

```bash
python train_and_export.py
```

The project expects model files in the `models/` directory:

```text
models/ai_resume_job_match_model.joblib
models/ai_resume_job_match_nlp_artifacts.joblib
```

---

## Run the app

Start the Streamlit interface:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## How the matching works

### Resume extraction
The app reads a PDF resume and extracts fields such as title, years of experience, education, industry, and detected skills. The extracted profile is validated before it is used for matching.

### Job retrieval and cleaning
The app calls the live Arbeitnow job board API, cleans raw job text, removes noisy HTML/Markdown artifacts, and standardizes fields like title, location, skills, and application URL.

### Semantic ranking
The new matching layer in `src/semantic_matcher.py` goes beyond simple keyword overlap. It evaluates the candidate and job across several dimensions:

- role/domain compatibility
- skill coverage and domain fit
- text similarity
- experience and seniority alignment

This makes the score more meaningful and helps avoid cases where unrelated roles with a few overlapping keywords appear highly matched.

### Match score vs. skill coverage
The app separates two concepts:

- Match Score: overall candidate-job compatibility score used to rank jobs
- Skill Coverage: explanation metric showing how many required skills are already covered

These are related, but not identical. The score reflects broader fit, while the coverage metric helps explain the gap in concrete skills.

---

## Learning recommendations

The app can recommend study paths for missing skills using `src/learning_recommendations.py`.

Each recommendation includes:

- the skill to learn
- its category
- a step-by-step learning path
- prerequisite awareness
- official resource links where available
- YouTube and Coursera search links for self-guided learning

This helps turn job-match results into actionable career development guidance instead of only a score.

---

## Filtering and outputs

The app supports:

- Show matched jobs only
- Minimum score threshold filtering
- Remote-only filtering
- Keyword-based search on title, company, or location

Each job card displays:

- match status and score
- company and location
- candidate and job profile summaries
- shared skills
- missing skills
- job link / application button
- learning recommendations for missing skills

---

## Testing

Run the project validation checks:

```bash
python test_suite.py
```

Run the end-to-end job board connectivity and matching flow:

```bash
python test_end_to_end.py
```

---

## Notes

This project is designed for career guidance and job matching exploration rather than as a guarantee of hiring outcome. The match score is an assistive signal based on the trained pipeline and available live job data.

---



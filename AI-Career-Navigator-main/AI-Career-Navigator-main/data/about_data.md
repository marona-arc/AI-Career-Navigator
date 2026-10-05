## 1. Dataset

The main training dataset will be the **Candidate Matching Synthetic Dataset** from Hugging Face:

[Hugging Face — Candidate Matching Synthetic Dataset](https://huggingface.co/datasets/michaelozon/candidate-matching-synthetic?utm_source=chatgpt.com)

The dataset will be used to create candidate-job examples and train the matching model.

Because the dataset is synthetic, the project should be presented as a **prototype/research project**, not as a real-world hiring prediction system.

### 1.1 Source Tables

The dataset is distributed as three separate tables (stored as Parquet files in the `resumes/`, `jobs/` and `matches/` folders of the repository). The `embeddings/` folder is not used in the core project.

| Table | Rows | Columns |
|---|---|---|
| **Resumes** | 10,000 | `resume_id`, `role`, `seniority`, `years_experience`, `industry`, `education`, `skills`, `summary`, `experience_bullets` |
| **Jobs** | 2,500 | `job_id`, `job_title`, `seniority`, `industry`, `must_have_skills`, `nice_to_have_skills`, `description`, `responsibilities`, `requirements` |
| **Matches** | 2,500 | `job_id`, `relevant_resume_ids` (about 30 matching resumes per job) |

The tables cannot be used for training directly, because resumes and jobs are separate entities and the matches table stores its links as a list inside a single cell.

### 1.2 Merged Training Dataset (Candidate-Job Pairs)

The three tables are merged into **one CSV file of candidate-job pairs**, where each row is one resume paired with one job:

```text
Matches table
      ↓
Explode lists → one row per (job, resume) pair      (~75,000 pairs, label = 1)
      +
Sampled non-matching pairs                          (~75,000 pairs, label = 0)
      ↓
Merge resume details (on resume_id)  +  job details (on job_id)
      ↓
candidate_job_pairs.csv   (~150,000 rows)
```

- Resume columns are prefixed with `r_` and job columns with `j_` (both tables contain `seniority` and `industry`).
- List columns (skills, bullets, responsibilities, requirements) are stored as `|`-separated text in the CSV and split back into lists when loaded.
- The original resume and job text is not modified.

### 1.3 Target Variable

The target is **`label`**:

- `1` = the resume matches the job
- `0` = the resume does not match the job

In the source dataset, a match is defined by a skill rule: **at least 60% of the job's `must_have_skills` appear in the resume's skills**. The dataset only contains matches, so the **non-matching pairs (label = 0) are created by the project** by pairing each job with resumes that are not in its relevant list. The final dataset is balanced (50% / 50%).

Non-matching pairs that accidentally satisfy the 60% rule are checked and removed. To make the task harder than random pairing, **hard negatives** (resumes with about 30–59% must-have skill overlap) can also be included.

### 1.4 Data Notes and Limitations

- **Label is skill-based:** because the label comes from skill overlap, skill-overlap features will make the task easy for the models. Results should be reported with and without the direct overlap features, and this limitation should be stated clearly.
- **Repeated entities:** the same resume and job appear in many pairs (10,000 unique resumes and 2,500 unique jobs in about 150,000 rows). The train/test split must be grouped by `job_id` so no job appears in both sets.
- **Balanced classes are artificial:** in real job search, most CV-job pairs do not match, so real-world performance can differ.
- **Limited fields:** jobs have no required years of experience and no education requirement, only `seniority`.
- **Generic text:** summaries, descriptions, responsibilities, requirements and experience bullets are mostly template text, so the most informative columns are the skills, seniority, role/title and industry.
- **Small skill vocabulary:** the dataset contains 73 unique skills, so skills extracted from real CVs and live job postings need to be mapped to this vocabulary.

---
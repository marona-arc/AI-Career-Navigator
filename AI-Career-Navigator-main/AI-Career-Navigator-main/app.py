"""
Streamlit application for AI Career Navigator.

Built around final_project.ipynb:
- Model and NLP artifacts loaded once via @st.cache_resource
- Live Arbeitnow API jobs cached via @st.cache_data (TTL=10 mins)
- Full candidate profile extraction & validation
- Exact notebook feature engineering and model prediction
- ML model probability ranking with separate skill-coverage explanation
- Debug mode for inspecting model inputs / outputs

IMPORTANT — terminology used in this app:
    • "Model Match Probability" = predict_proba(X)[:, positive_class]
      This is the trained XGBoost model's probability that the
      candidate-job pair belongs to the positive (match) class.
      It is NOT a hire probability and NOT a skill percentage.

    • "Prediction" = model.predict(X)
      Binary classification result (MATCHED / NOT MATCHED).
      Threshold: 0.5 on the positive-class probability.

    • "Skill Coverage" = shared must-have skills / total must-have skills
      This is a separate explanation metric, NOT the model prediction.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.artifacts import ArtifactBundle
from src.cv_extractor import extract_cv_profile, validate_cv
from src.job_api import get_jobs
from src.job_cleaner import clean_jobs
from src.matcher import match_cv_to_jobs                     # kept for reference
from src.semantic_matcher import semantic_match_cv_to_jobs   # new holistic engine
from src.learning_recommendations import recommend_learning  # study plan for missing skills

# ──────────────────────────────────────────────────────────────────────────────
# Developer debug flag  (set to True only for local debugging)
# ──────────────────────────────────────────────────────────────────────────────
DEBUG_MODE = False

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration & Styling
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Career Navigator",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-match {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.28rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        letter-spacing: 0.03em;
    }
    .badge-no-match {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.28rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        letter-spacing: 0.03em;
    }
    .prob-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1E3A8A;
        line-height: 1;
    }
    .prob-label {
        font-size: 0.72rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.1rem;
    }
    .skill-tag-shared {
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-size: 0.82rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        display: inline-block;
        font-weight: 500;
    }
    .skill-tag-missing {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-size: 0.82rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        display: inline-block;
        font-weight: 500;
    }
    .info-disclaimer {
        font-size: 0.8rem;
        color: #6B7280;
        margin-top: 0.5rem;
    }
    .section-divider {
        border-top: 1px solid #E5E7EB;
        margin: 0.75rem 0;
    }
    .profile-row {
        font-size: 0.85rem;
        color: #374151;
        margin: 0.15rem 0;
    }
    .profile-label {
        font-weight: 600;
        color: #111827;
    }
    .debug-box {
        background-color: #1F2937;
        color: #D1FAE5;
        font-family: monospace;
        font-size: 0.75rem;
        padding: 0.8rem;
        border-radius: 6px;
        overflow-x: auto;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Cached Resources: Model and API Job retrieval
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading trained ML pipeline...")
def load_ml_bundle():
    """Load model and TF-IDF artifacts once."""
    return ArtifactBundle()


@st.cache_data(ttl=600, show_spinner="Fetching available jobs from Arbeitnow Job Board...")
def load_and_clean_live_jobs():
    """Fetch and clean live jobs from Arbeitnow API, cached for 10 minutes."""
    raw_jobs = get_jobs()
    cleaned = clean_jobs(raw_jobs)
    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Helper renderers
# ──────────────────────────────────────────────────────────────────────────────

def _render_skill_tags(skills: list[str], css_class: str, prefix: str = "") -> str:
    return " ".join(
        f'<span class="{css_class}">{prefix}{s}</span>' for s in skills
    )


def _fmt_years(years: int | None) -> str:
    if years is None:
        return "Not specified"
    return f"{years}+"


def _render_plan_items(plan: list[dict], counts: dict | None = None) -> None:
    """Render learning-plan items (output of recommend_learning)."""
    for item in plan:
        header = f"**{item['step']}. {item['skill']}** &nbsp;·&nbsp; _{item['category']}_"
        if counts and item["skill"] in counts:
            header += f" &nbsp;·&nbsp; missing in {counts[item['skill']]} job(s)"
        st.markdown(header, unsafe_allow_html=True)
        st.markdown("📖 **Study path:** " + " → ".join(item["topics"]))

        if item["prerequisites_also_missing"]:
            st.caption("Learn first: " + ", ".join(item["prerequisites_also_missing"]))
        elif item["prerequisites_you_have"]:
            st.caption("Builds on what you already know: " + ", ".join(item["prerequisites_you_have"]))

        links = []
        if item["official_resource"]:
            links.append(f"[Official guide ↗]({item['official_resource']})")
        links.append(f"[YouTube ↗]({item['search_links']['YouTube']})")
        links.append(f"[Coursera ↗]({item['search_links']['Coursera']})")
        st.markdown(" &nbsp;•&nbsp; ".join(links))
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def _render_learning_plan(missing: list[str], candidate_skills: list[str], max_skills: int = 5) -> None:
    """Per-job expander: what to study for this job's missing skills."""
    if not missing:
        return
    plan = recommend_learning(missing, candidate_skills=candidate_skills, max_skills=max_skills)
    label = f"📚 Learning Recommendations ({len(plan)} of {len(missing)} missing skills)"
    with st.expander(label, expanded=False):
        _render_plan_items(plan)
        if len(missing) > len(plan):
            st.caption(f"Showing the first {len(plan)} steps. {len(missing) - len(plan)} more missing skill(s) not shown.")


def _render_top_skills_to_learn(results: list[dict], candidate_skills: list[str], top_n: int = 6) -> None:
    """Across all displayed jobs, which missing skills would unlock the most of them?"""
    counts: dict[str, int] = {}
    for r in results:
        for skill in r.get("missing_skills", []):
            counts[skill] = counts.get(skill, 0) + 1
    if not counts:
        return

    ranked = sorted(counts, key=lambda s: (-counts[s], s))[:top_n]
    plan = recommend_learning(ranked, candidate_skills=candidate_skills)
    with st.expander(
        f"📚 Skills worth learning across your {len(results)} displayed job(s)",
        expanded=True,
    ):
        st.caption("Ranked by how many of the jobs below list the skill as missing for you.")
        _render_plan_items(plan, counts=counts)


def _render_job_card(job: dict, show_debug: bool, candidate_skills: list[str] | None = None) -> None:
    """Render one job result card."""
    is_match  = job["matched"]
    badge_cls = "badge-match" if is_match else "badge-no-match"
    badge_txt = "✓ MATCHED" if is_match else "✗ NOT MATCHED"
    remote_txt = "🌐 Remote" if job["remote"] else "🏢 On-site"
    prob_pct  = job["match_prob_pct"]

    with st.container(border=True):
        # ── Header row ────────────────────────────────────────────────
        col_title, col_badge = st.columns([7, 3])
        with col_title:
            st.markdown(f"### {job['title']}")
            st.markdown(
                f"🏢 **{job['company']}** &nbsp;•&nbsp; "
                f"📍 {job['location'] or 'Location not specified'} &nbsp;•&nbsp; "
                f"{remote_txt} &nbsp;•&nbsp; ⏱️ {job['job_type']}"
            )
        with col_badge:
            st.markdown(
                f'<div style="text-align:right; padding-top:0.5rem;">'
                f'<span class="{badge_cls}">{badge_txt}</span>'
                f'<div class="prob-value" style="margin-top:0.5rem;">{prob_pct:.1f}%</div>'
                f'<div class="prob-label">Match Score</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Candidate / Job profile summaries ─────────────────────────
        col_cand, col_job = st.columns(2)
        with col_cand:
            st.markdown("**📄 Candidate Profile**")
            cand_items = [
                ("Role",       job.get("candidate_role", "—")),
                ("Seniority",  job.get("candidate_seniority", "—")),
                ("Experience", f"{job.get('candidate_experience', 0)} yrs"),
                ("Education",  job.get("candidate_education", "—")),
                ("Industry",   job.get("candidate_industry", "Unknown")),
            ]
            for label, val in cand_items:
                st.markdown(
                    f'<div class="profile-row"><span class="profile-label">{label}:</span> {val}</div>',
                    unsafe_allow_html=True,
                )

        with col_job:
            st.markdown("**💼 Job Profile**")
            job_items = [
                ("Role",               job.get("job_canonical_title", job["title"])),
                ("Domain",             job.get("job_domain", "—")),
                ("Seniority",          job.get("job_seniority", "—")),
                ("Experience Req.",    _fmt_years(job.get("job_required_years")) + " yrs" if job.get("job_required_years") else "Not specified"),
                ("Industry",           job.get("job_industry", "Unknown")),
                ("Required Skills",    str(len(job.get("job_skills", []))) + " detected"),
            ]
            for label, val in job_items:
                st.markdown(
                    f'<div class="profile-row"><span class="profile-label">{label}:</span> {val}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Match Reason ──────────────────────────────────────────────
        match_reason = job.get("match_reason", "")
        if match_reason:
            is_positive = job["matched"]
            icon = "✅" if is_positive else "ℹ️"
            st.markdown(
                f'<div class="profile-row"><span class="profile-label">{icon} Why:</span> {match_reason}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Skills section ────────────────────────────────────────────
        col_shared, col_missing, col_btn = st.columns([4, 4, 2])
        with col_shared:
            shared = job.get("shared_skills", [])
            st.markdown(f"**✅ Shared Skills ({len(shared)})**")
            if shared:
                st.markdown(_render_skill_tags(shared, "skill-tag-shared", "✓ "), unsafe_allow_html=True)
            else:
                st.caption("None from the 73-skill vocabulary")

        with col_missing:
            missing = job.get("missing_skills", [])
            st.markdown(f"**⚠️ Missing Skills ({len(missing)})**")
            if missing:
                st.markdown(_render_skill_tags(missing, "skill-tag-missing", "• "), unsafe_allow_html=True)
            else:
                st.caption("All required skills covered")

        with col_btn:
            st.write("")
            if job.get("url"):
                st.link_button("View / Apply ↗", url=job["url"], use_container_width=True)
            else:
                st.button("View Job", disabled=True, use_container_width=True, key=f"btn_{job['job_id']}")

        # ── Skill coverage note ───────────────────────────────────────
        cov = job.get("skill_coverage", 0.0)
        st.caption(
            f"Skill Coverage (explanation only): **{cov:.1f}%** of required skills matched "
            f"— this is separate from the ML model probability above."
        )

        # ── Learning recommendations for missing skills ───────────────
        _render_learning_plan(job.get("missing_skills", []), candidate_skills or [])

        # ── Debug panel ───────────────────────────────────────────────
        if show_debug and job.get("_debug"):
            dbg = job["_debug"]
            with st.expander("🔬 Debug: Model Input & Output", expanded=False):
                st.markdown("**Candidate features extracted:**")
                st.json(dbg["candidate_features"])

                st.markdown("**Job features extracted:**")
                st.json(dbg["job_features"])

                st.markdown("**Feature vector passed to model:**")
                # Sort by absolute value descending for readability
                model_input = dbg["model_input"]
                sorted_feats = dict(
                    sorted(model_input.items(), key=lambda x: abs(x[1]), reverse=True)
                )
                st.json(sorted_feats)

                st.markdown(
                    f"**Model classes:** `{dbg['model_classes']}`  \n"
                    f"**Positive class index:** `{dbg['positive_class_idx']}`  \n"
                    f"**Raw prediction:** `{dbg['raw_prediction']}`  \n"
                    f"**Class probabilities:** `{dbg['class_probabilities']}`"
                )


# ──────────────────────────────────────────────────────────────────────────────
# Application Layout
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown('<div class="main-header">🎯 AI Career Navigator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload your CV to find live matching job opportunities '
        'ranked by your trained ML model\'s match probability.</div>',
        unsafe_allow_html=True,
    )

    # ── Initialize ML bundle ──────────────────────────────────────────
    try:
        bundle = load_ml_bundle()
    except Exception as exc:
        st.error(f"❌ Failed to load trained ML model: {exc}")
        st.info(
            "Please ensure `models/ai_resume_job_match_model.joblib` and "
            "`models/ai_resume_job_match_nlp_artifacts.joblib` exist."
        )
        return

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration & Filters")

        # Model info
        st.markdown(f"**Model:** `{bundle.model_name}`")
        st.markdown(f"**Feature Set:** `Features without overlap`  \n 11 features)")
        st.markdown(f"**Skill Vocabulary:** {len(bundle.SKILL_VOCAB)} canonical skills")

        classes = list(getattr(bundle.model, "classes_", [0, 1]))
        try:
            pos_idx = classes.index(1)
        except ValueError:
            pos_idx = len(classes) - 1
        st.markdown(f"**Positive class:** `{classes[pos_idx]}` (index {pos_idx})")

        st.divider()

        filter_matched_only = st.checkbox("Show Matched Jobs Only", value=True)
        min_prob = st.slider(
            "Minimum Match Score (%)",
            min_value=0, max_value=100, value=50, step=5,
            help="Filters by the holistic candidate-job match score (0–100%).",
        )
        remote_only = st.checkbox("Remote Jobs Only", value=False)
        search_query = st.text_input(
            "Filter by Job Title / Keyword",
            placeholder="e.g. Analyst, Engineer",
        )

        st.divider()

        # Developer debug mode toggle (kept for any future use, hidden from semantic matcher)
        show_debug = st.checkbox(
            "🔬 Developer Debug Mode",
            value=DEBUG_MODE,
            help=(
                "Enables additional debugging information for troubleshooting."
            ),
        )

        st.divider()
        st.markdown(
            """
            <div class="info-disclaimer">
            <b>Match Score</b> is computed by holistic semantic evaluation
            across four dimensions: role/domain alignment, skill overlap,
            full-text similarity, and experience fit.<br><br>
            A single shared skill in a mismatched domain
            <b>cannot</b> produce a high score.
            <br><br>
            <em>Skill Coverage</em> is shown separately as an explanation metric.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Step 1: Upload CV ─────────────────────────────────────────────
    st.subheader("1. Upload Your Resume")
    uploaded_file = st.file_uploader(
        "Choose a text-based PDF resume",
        type=["pdf"],
        help="Upload a standard PDF resume. Scanned images will not extract correctly.",
    )

    if uploaded_file is None:
        st.info("👆 Please upload your CV in PDF format to begin matching against live jobs.")
        return

    pdf_bytes = uploaded_file.read()
    try:
        from src.cv_extractor import extract_text_from_pdf
        raw_text = extract_text_from_pdf(pdf_bytes)
        candidate_profile = bundle.build_extended_candidate_profile(raw_text)
        validate_cv(candidate_profile)
    except ValueError as val_err:
        st.error(f"⚠️ {val_err}")
        return
    except Exception as exc:
        st.error(
            f"⚠️ Could not extract enough information from this CV. "
            f"Please upload a valid text-based PDF. (Error: {exc})"
        )
        return

    # ── Step 2: Candidate Profile Display ────────────────────────────
    st.markdown("---")
    st.subheader("2. Extracted Candidate Profile")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Role", candidate_profile["role"])
    with c2:
        st.metric("Seniority", candidate_profile["seniority"])
    with c3:
        st.metric("Experience", f"{candidate_profile['years_experience']} yrs")
    with c4:
        st.metric("Education", candidate_profile["education"])
    with c5:
        st.metric("Industry", candidate_profile.get("industry", "Unknown"))

    st.markdown("**Identified Skills** *(matched against expanded AI/ML/Data/Cloud + business skill vocabulary)*:")
    if candidate_profile["skills"]:
        skills_html = _render_skill_tags(candidate_profile["skills"], "skill-tag-shared")
        st.markdown(skills_html, unsafe_allow_html=True)
    else:
        st.warning(
            "No canonical skills from the 73-skill vocabulary were detected in this CV. "
            "The model will rely on text similarity and other features."
        )

    if show_debug:
        with st.expander("🔬 Debug: Full Candidate Profile", expanded=False):
            display_profile = {k: v for k, v in candidate_profile.items() if k != "text"}
            st.json(display_profile)
            st.caption(f"CV text length: {len(candidate_profile.get('text', ''))} preprocessed tokens")

    # ── Step 3: Fetch & Clean Jobs ────────────────────────────────────
    st.markdown("---")
    st.subheader("3. Job Match Results")

    try:
        cleaned_jobs = load_and_clean_live_jobs()
    except RuntimeError as r_err:
        st.error(f"⚠️ {r_err}")
        return
    except Exception as exc:
        st.error(f"⚠️ Unable to retrieve jobs from Arbeitnow API. Please try again later. ({exc})")
        return

    if not cleaned_jobs:
        st.warning("No jobs are currently available.")
        return

    # ── Run inference ─────────────────────────────────────────────────
    with st.spinner("Evaluating candidate against all available jobs..."):
        all_results = semantic_match_cv_to_jobs(
            candidate_profile, cleaned_jobs, bundle
        )

    if not all_results:
        st.warning("No jobs could be processed successfully.")
        return

    # ── Apply filters ─────────────────────────────────────────────────
    filtered = all_results

    if filter_matched_only:
        filtered = [r for r in filtered if r["matched"]]

    if min_prob > 0:
        filtered = [r for r in filtered if r["match_prob_pct"] >= min_prob]

    if remote_only:
        filtered = [r for r in filtered if r["remote"]]

    if search_query.strip():
        q = search_query.strip().lower()
        filtered = [
            r for r in filtered
            if q in r["title"].lower()
            or q in r["company"].lower()
            or q in r["location"].lower()
        ]

    # ── Summary metrics ───────────────────────────────────────────────
    total_evaluated = len(all_results)
    total_matched   = sum(1 for r in all_results if r["matched"])
    showing_count   = len(filtered)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Live Jobs Analyzed", total_evaluated)
    col_m2.metric("Matched (≥50%)", total_matched)
    col_m3.metric("Displayed Results", showing_count)
    if all_results:
        avg_prob = sum(r["match_prob_pct"] for r in all_results) / len(all_results)
        col_m4.metric("Avg. Match Score", f"{avg_prob:.1f}%")

    if total_matched == 0:
        st.info(
            "No jobs reached the 50% match threshold. "
            "This may mean the available jobs are not closely related to the candidate's profile. "
            "Try unchecking 'Show Matched Jobs Only' to see all results ranked by score."
        )

    # ── About the Match Score ─────────────────────────────────────────
    with st.expander("ℹ️ About the Match Score", expanded=False):
        st.markdown(
            """
            **How the match score is computed:**
            Each job is evaluated against the complete candidate profile across four dimensions:

            | Dimension | Weight | What it measures |
            |-----------|--------|-----------------|
            | Role & domain alignment | 35% | How semantically related the candidate's role is to the job domain and title |
            | Skill overlap | 35% | Technical skill match, gated by domain compatibility |
            | Text similarity | 15% | Full-text TF-IDF cosine similarity between CV and job description |
            | Experience & seniority | 15% | Candidate's years of experience vs. job requirements |

            **Why a single shared skill cannot produce a high score:**
            The skill overlap component is multiplied by the domain compatibility factor.
            If a candidate is an AI/ML Engineer and the job is Head of Sales, the domain
            compatibility is ~0.05 — so even perfect skill overlap would contribute only 5% of 35%.
            The overall score is hard-capped at 18% for cross-domain mismatches.

            **Confidence** reflects how decisively the score sits above or below the 50% threshold.
            It is not an ML model probability — it is a calibration of how clear-cut the decision is.

            **Skill Coverage** (shown at the bottom of each card) is a separate explanation metric
            showing what fraction of the job's listed skills the candidate has.
            """
        )

    if not filtered:
        st.info(
            "No jobs match your selected filters. "
            "Try lowering the minimum match score or unchecking 'Show Matched Jobs Only'."
        )
        return

    # ── Job Cards ─────────────────────────────────────────────────────
    st.markdown(f"**Showing {showing_count} results, ranked by Match Score (highest first):**")

    cand_skills = candidate_profile.get("skills", [])
    _render_top_skills_to_learn(filtered, cand_skills)

    for job in filtered:
        _render_job_card(job, show_debug=show_debug, candidate_skills=cand_skills)


if __name__ == "__main__":
    main()

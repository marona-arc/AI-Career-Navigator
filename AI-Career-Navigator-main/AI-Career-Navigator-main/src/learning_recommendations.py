"""
learning_recommendations.py
---------------------------
For every skill a candidate is MISSING for a job, recommend what to study.

Pure Python (no ML dependencies), so it can be imported by the notebook,
train_and_export.py, or the Streamlit app.

Covers all 73 skills in SKILL_VOCAB. Official links are only given where a
stable official page exists; every skill also gets generated search links
(YouTube / Coursera), which can never go stale.
"""

from urllib.parse import quote_plus

# skill -> (category, [topics to study, in order], official_url or None)
SKILL_GUIDE = {
    # ── Engineering ────────────────────────────────────────────
    "Python": ("Engineering", ["Syntax and data types", "Functions and modules", "File I/O and error handling", "Small projects"], "https://docs.python.org/3/tutorial/"),
    "Java": ("Engineering", ["Syntax and types", "Classes and interfaces", "Collections", "Build tools (Maven/Gradle)"], "https://dev.java/learn/"),
    "JavaScript": ("Engineering", ["Core language and ES6+", "DOM manipulation", "Async/await and fetch", "A framework such as React or Node.js"], "https://developer.mozilla.org/en-US/docs/Learn/JavaScript"),
    "SQL": ("Engineering", ["SELECT, WHERE, ORDER BY", "JOINs", "GROUP BY and aggregates", "Window functions and subqueries"], None),
    "Databases": ("Engineering", ["Relational vs NoSQL", "Schema design and normalization", "Indexes and query performance", "Transactions"], "https://www.postgresql.org/docs/current/tutorial.html"),
    "Git": ("Engineering", ["Commit, branch, merge", "Pull requests and code review", "Resolving merge conflicts", "Rebasing and history"], "https://git-scm.com/book"),
    "Docker": ("Engineering", ["Images vs containers", "Writing a Dockerfile", "Volumes and networking", "Docker Compose"], "https://docs.docker.com/get-started/"),
    "CI/CD": ("Engineering", ["Build/test/deploy pipeline concepts", "GitHub Actions or GitLab CI", "Automated testing in pipelines", "Deployment strategies"], "https://docs.github.com/en/actions"),
    "REST APIs": ("Engineering", ["HTTP methods and status codes", "Resource design and versioning", "Authentication (API keys, OAuth, JWT)", "Build one with Flask or FastAPI"], "https://restfulapi.net/"),
    "OOP": ("Engineering", ["Classes and objects", "Encapsulation, inheritance, polymorphism", "SOLID principles", "Common design patterns"], None),
    "Unit Testing": ("Engineering", ["Test structure (arrange/act/assert)", "pytest or JUnit basics", "Mocking", "Test coverage"], "https://docs.pytest.org/en/stable/getting-started.html"),

    # ── Data & analytics ───────────────────────────────────────
    "Pandas": ("Data & Analytics", ["Series and DataFrames", "Cleaning and missing values", "groupby and merge", "Time series basics"], "https://pandas.pydata.org/docs/getting_started/intro_tutorials/"),
    "Statistics": ("Data & Analytics", ["Descriptive statistics", "Probability distributions", "Hypothesis testing and p-values", "Regression basics"], "https://www.khanacademy.org/math/statistics-probability"),
    "Excel": ("Data & Analytics", ["Formulas (VLOOKUP/XLOOKUP, IF, SUMIFS)", "Pivot tables", "Charts", "Data cleaning"], None),
    "Power BI": ("Data & Analytics", ["Connecting and shaping data", "Data modeling", "DAX basics", "Building dashboards"], "https://learn.microsoft.com/en-us/power-bi/fundamentals/"),
    "Tableau": ("Data & Analytics", ["Connecting data sources", "Charts and calculated fields", "Dashboards", "Storytelling with data"], "https://www.tableau.com/learn/training"),
    "Data Visualization": ("Data & Analytics", ["Choosing the right chart", "Color and layout principles", "Dashboard design", "matplotlib/seaborn or a BI tool"], None),
    "ETL": ("Data & Analytics", ["Extract/transform/load concepts", "Data quality checks", "Batch vs streaming", "Orchestration (e.g. Airflow)"], None),
    "Analytics": ("Data & Analytics", ["Core metrics and funnels", "Google Analytics fundamentals", "Cohort and retention analysis", "Turning data into recommendations"], "https://skillshop.withgoogle.com/"),
    "A/B Testing": ("Data & Analytics", ["Hypotheses and metrics", "Sample size and power", "Significance and common pitfalls", "Interpreting results"], None),
    "KPIs": ("Data & Analytics", ["Choosing meaningful KPIs", "Leading vs lagging indicators", "Setting targets", "KPI dashboards"], None),
    "Reporting": ("Data & Analytics", ["Structuring a report", "Automating recurring reports", "Writing for executives", "Data storytelling"], None),
    "Variance Analysis": ("Data & Analytics", ["Budget vs actual", "Price/volume/mix variance", "Explaining drivers", "Reporting variances"], None),

    # ── Product & project management ───────────────────────────
    "Agile": ("Product & Project", ["Agile values and principles", "Kanban basics", "User stories and backlog", "Retrospectives"], "https://agilemanifesto.org/"),
    "Scrum": ("Product & Project", ["Roles, events and artifacts", "Sprint planning and review", "Backlog refinement", "Velocity and burndown"], "https://scrumguides.org/"),
    "Jira": ("Product & Project", ["Projects, issues and boards", "Workflows", "Sprints and backlogs", "Filters (JQL) and dashboards"], "https://www.atlassian.com/software/jira/guides"),
    "Asana": ("Product & Project", ["Projects, tasks and sections", "Boards and timelines", "Rules and automation", "Team workflows"], "https://asana.com/guide"),
    "PRD": ("Product & Project", ["Problem statement and goals", "User stories and requirements", "Success metrics", "Reviewing with engineering"], None),
    "Roadmap": ("Product & Project", ["Outcome-based roadmaps", "Now/Next/Later format", "Aligning stakeholders", "Communicating changes"], None),
    "Product Strategy": ("Product & Project", ["Vision and positioning", "Market and competitor analysis", "Prioritization frameworks", "Measuring product-market fit"], None),
    "Prioritization": ("Product & Project", ["RICE and ICE scoring", "MoSCoW", "Impact vs effort", "Saying no to stakeholders"], None),
    "User Research": ("Product & Project", ["Interviews", "Surveys", "Usability testing", "Synthesizing insights"], None),
    "Project Planning": ("Product & Project", ["Scope and work breakdown", "Estimation", "Dependencies and critical path", "Status tracking"], None),
    "Timeline Management": ("Product & Project", ["Gantt charts", "Milestones", "Buffering and re-planning", "Tracking progress"], None),
    "Risk Management": ("Product & Project", ["Risk identification", "Probability/impact matrix", "Mitigation plans", "Risk registers"], None),
    "Process Improvement": ("Product & Project", ["Process mapping", "Lean basics", "Finding bottlenecks", "Measuring improvement"], None),
    "Documentation": ("Product & Project", ["Writing clear technical docs", "READMEs and runbooks", "Docs-as-code", "Keeping docs current"], None),
    "Cross-functional Coordination": ("Product & Project", ["Running effective meetings", "Aligning goals across teams", "RACI matrices", "Async updates"], None),

    # ── Marketing ──────────────────────────────────────────────
    "SEO": ("Marketing", ["Keyword research", "On-page SEO", "Technical SEO", "Link building"], "https://developers.google.com/search/docs/fundamentals/seo-starter-guide"),
    "Google Ads": ("Marketing", ["Campaign structure", "Keywords and match types", "Bidding strategies", "Ad copy and quality score"], "https://skillshop.withgoogle.com/"),
    "Meta Ads": ("Marketing", ["Ads Manager basics", "Audience targeting", "Creative testing", "Pixel and conversion tracking"], "https://www.facebook.com/business/learn"),
    "Content Marketing": ("Marketing", ["Content strategy and calendar", "Audience personas", "Distribution", "Measuring content ROI"], "https://academy.hubspot.com/"),
    "Email Marketing": ("Marketing", ["List building and segmentation", "Subject lines and copy", "Automation flows", "Deliverability and metrics"], "https://academy.hubspot.com/"),
    "Copywriting": ("Marketing", ["Headlines and hooks", "Benefits vs features", "Calls to action", "Editing for clarity"], None),
    "Landing Pages": ("Marketing", ["Page structure", "Messaging and CTA", "Page speed", "Testing variants"], None),
    "Conversion Optimization": ("Marketing", ["Funnel analysis", "User behavior tools", "Hypothesis-driven testing", "Form and checkout optimization"], None),
    "Marketing Analytics": ("Marketing", ["Attribution models", "CAC, LTV and ROAS", "Campaign dashboards", "Channel mix analysis"], "https://skillshop.withgoogle.com/"),

    # ── Sales & customer success ───────────────────────────────
    "CRM": ("Sales & Customer", ["CRM data hygiene", "Pipelines and stages", "Reports and dashboards", "Workflow automation"], "https://academy.hubspot.com/"),
    "Lead Generation": ("Sales & Customer", ["Ideal customer profile", "Inbound vs outbound", "Lead scoring", "Nurturing sequences"], "https://academy.hubspot.com/"),
    "Prospecting": ("Sales & Customer", ["Researching accounts", "Finding decision-makers", "Personalized outreach", "Qualification"], None),
    "Outbound Outreach": ("Sales & Customer", ["Cold email structure", "Cold calling basics", "Follow-up cadences", "Handling objections"], None),
    "Discovery Calls": ("Sales & Customer", ["Question frameworks (e.g. SPIN)", "Active listening", "Qualifying pain and budget", "Next-step setting"], None),
    "Pipeline Management": ("Sales & Customer", ["Stage definitions", "Forecasting deals", "Pipeline reviews", "Deal velocity"], None),
    "Closing": ("Sales & Customer", ["Closing techniques", "Handling objections", "Proposals and pricing", "Negotiating terms"], None),
    "Negotiation": ("Sales & Customer", ["Preparation and BATNA", "Anchoring", "Win-win trade-offs", "Concession strategy"], None),
    "Account Management": ("Sales & Customer", ["Onboarding customers", "Renewals and upsells", "Business reviews", "Churn prevention"], None),
    "Customer Satisfaction": ("Sales & Customer", ["CSAT and NPS", "Collecting feedback", "Closing the loop", "Improving support quality"], None),
    "Escalations": ("Sales & Customer", ["Escalation criteria", "Handoff communication", "De-escalation", "Post-incident follow-up"], None),
    "Ticketing": ("Sales & Customer", ["Ticket lifecycle", "Prioritization and tagging", "Macros and templates", "Support metrics"], None),
    "Zendesk": ("Sales & Customer", ["Tickets and views", "Triggers and automations", "Macros", "Reporting"], None),
    "Intercom": ("Sales & Customer", ["Inbox and conversations", "Messenger and bots", "Help center articles", "Automation rules"], None),
    "SLA": ("Sales & Customer", ["Defining SLAs and SLOs", "Response and resolution targets", "Monitoring compliance", "Handling breaches"], None),
    "Troubleshooting": ("Sales & Customer", ["Reproducing issues", "Isolating variables", "Reading logs", "Documenting fixes"], None),
    "Root Cause Analysis": ("Sales & Customer", ["5 Whys", "Fishbone diagrams", "Fault-tree thinking", "Writing postmortems"], None),

    # ── Finance ────────────────────────────────────────────────
    "Accounting": ("Finance", ["Debits and credits", "Financial statements", "Accruals", "Reconciliations"], None),
    "Budgeting": ("Finance", ["Top-down vs bottom-up budgets", "Cost drivers", "Budget vs actual tracking", "Scenario planning"], None),
    "Forecasting": ("Finance", ["Trend and seasonality", "Driver-based forecasts", "Rolling forecasts", "Forecast accuracy"], None),
    "Financial Modeling": ("Finance", ["Three-statement models", "Assumptions and drivers", "Sensitivity analysis", "Model auditing in Excel"], None),
    "Cash Flow": ("Finance", ["Operating/investing/financing cash flows", "Cash flow forecasting", "Working capital", "Free cash flow"], None),
    "Valuation": ("Finance", ["DCF", "Comparable companies", "Precedent transactions", "WACC"], None),

    # ── Soft skills ────────────────────────────────────────────
    "Communication": ("Soft Skills", ["Clear written communication", "Presenting to different audiences", "Active listening", "Giving and receiving feedback"], None),
    "Stakeholder Communication": ("Soft Skills", ["Mapping stakeholders", "Tailoring updates", "Status reporting", "Managing expectations"], None),
    "Stakeholder Management": ("Soft Skills", ["Influence and interest mapping", "Building trust", "Handling conflict", "Securing buy-in"], None),
}

# Learn these before the dependent skill (only used to order the plan).
PREREQUISITES = {
    "Pandas": ["Python"],
    "Unit Testing": ["Python"],
    "CI/CD": ["Git", "Docker"],
    "Docker": ["Git"],
    "REST APIs": ["Python"],
    "ETL": ["SQL", "Python"],
    "Databases": ["SQL"],
    "Data Visualization": ["Excel"],
    "Power BI": ["Excel"],
    "Tableau": ["Excel"],
    "Financial Modeling": ["Excel", "Accounting"],
    "Valuation": ["Financial Modeling"],
    "Variance Analysis": ["Budgeting"],
    "Forecasting": ["Excel"],
    "Scrum": ["Agile"],
    "Roadmap": ["Product Strategy"],
    "PRD": ["User Research"],
    "Marketing Analytics": ["Analytics"],
    "Conversion Optimization": ["A/B Testing", "Analytics"],
    "Closing": ["Discovery Calls"],
    "Pipeline Management": ["CRM"],
    "SLA": ["Ticketing"],
}

_GENERIC_TOPICS = ["Core concepts and terminology", "Hands-on practice project", "Common tools and best practices"]


def _search_links(skill):
    q = quote_plus(f"{skill} tutorial for beginners")
    return {
        "YouTube": f"https://www.youtube.com/results?search_query={q}",
        "Coursera": f"https://www.coursera.org/search?query={quote_plus(skill)}",
    }


def _order_missing(missing):
    """Put prerequisites that are also missing before the skills that need them."""
    missing = list(dict.fromkeys(missing))
    ordered, seen = [], set()

    def visit(skill):
        if skill in seen:
            return
        seen.add(skill)
        for pre in PREREQUISITES.get(skill, []):
            if pre in missing:
                visit(pre)
        ordered.append(skill)

    for s in missing:
        visit(s)
    return ordered


def recommend_learning(missing_skills, candidate_skills=None, max_skills=None):
    """
    Build a study plan for missing skills.

    missing_skills   : list of skill names (e.g. result["missing_skills"])
    candidate_skills : optional, the candidate's current skills. Used to say
                       which prerequisites they already have.
    max_skills       : optional cap on how many missing skills to cover.

    Returns a list of dicts, in the order the skills should be learned.
    """
    candidate_skills = set(candidate_skills or [])
    ordered = _order_missing(missing_skills)
    if max_skills is not None:
        ordered = ordered[:max_skills]

    plan = []
    for step, skill in enumerate(ordered, start=1):
        category, topics, url = SKILL_GUIDE.get(skill, ("General", _GENERIC_TOPICS, None))
        prereqs = PREREQUISITES.get(skill, [])
        plan.append({
            "step": step,
            "skill": skill,
            "category": category,
            "topics": topics,
            "official_resource": url,
            "search_links": _search_links(skill),
            "prerequisites_you_have": [p for p in prereqs if p in candidate_skills],
            "prerequisites_also_missing": [p for p in prereqs if p in ordered and p not in candidate_skills],
        })
    return plan


def format_learning_plan(plan):
    """Plain-text version, handy for the notebook and for logging."""
    if not plan:
        return "No missing skills - nothing to study for this job."
    lines = []
    for item in plan:
        lines.append(f"{item['step']}. {item['skill']}  [{item['category']}]")
        lines.append("   Study: " + " -> ".join(item["topics"]))
        if item["prerequisites_you_have"]:
            lines.append("   You already know: " + ", ".join(item["prerequisites_you_have"]))
        if item["prerequisites_also_missing"]:
            lines.append("   Learn first: " + ", ".join(item["prerequisites_also_missing"]))
        if item["official_resource"]:
            lines.append("   Official: " + item["official_resource"])
        lines.append("   Videos:   " + item["search_links"]["YouTube"])
        lines.append("   Courses:  " + item["search_links"]["Coursera"])
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    demo = recommend_learning(
        ["CI/CD", "Docker", "Pandas", "Git", "Statistics"],
        candidate_skills=["Python", "SQL"],
    )
    print(format_learning_plan(demo))

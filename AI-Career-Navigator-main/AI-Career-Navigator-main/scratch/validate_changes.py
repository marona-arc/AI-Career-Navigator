import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')
from src.artifacts import ArtifactBundle
from src.feature_engineering import make_features
from src.matcher import predict_one, match_cv_to_jobs

bundle = ArtifactBundle()
print('Bundle loaded OK')
print('Model:', bundle.model_name)
print('Feature set:', bundle.feature_set_name)
classes = list(getattr(bundle.model, 'classes_', [0, 1]))
pos_idx = classes.index(1) if 1 in classes else len(classes) - 1
print('Model classes:', classes)
print('Positive class index:', pos_idx)

cv_text = """
Alex Morgan
Data Analyst with 3 years of experience.
Education: Bachelor of Science in Computer Science. Technology industry.
Skills: Python, SQL, Pandas, Data Visualization, Statistics, Power BI, Excel.
Experience: Analyzed large datasets using SQL and Python. Created dashboards.
"""
candidate = bundle.build_candidate_profile(cv_text)
print()
print('Candidate industry (should NOT be Unknown):', candidate['industry'])
print('Candidate skills:', candidate['skills'])

test_jobs = [
    ('Data Scientist',  'Data Scientist 3 years experience Python SQL Machine Learning Statistics.', []),
    ('Backend Engineer','Senior Backend Engineer Java Spring Boot Kubernetes Docker AWS 5 years.', []),
    ('Graphic Designer','Graphic Designer Figma Photoshop Illustrator brand design portfolio.', []),
]

print()
print('--- Diagnostic ---')
for title, desc, tags in test_jobs:
    jp = bundle.build_job_profile(title, desc, tags=tags)
    pred = predict_one(candidate, jp, bundle, debug=True)
    prediction = pred['prediction']
    prob = pred['match_probability']
    industry = jp['industry']
    print(f'Job: {title:22s} | Pred: {prediction} | Prob: {prob:.4f} | Industry: {industry}')
    dbg = pred['_debug']
    if dbg:
        print(f'  classes={dbg["model_classes"]}  pos_idx={dbg["positive_class_idx"]}')
        print(f'  class_probs={dbg["class_probabilities"]}')
        # print key model input features
        mi = dbg['model_input']
        print(f'  must_have_coverage={mi["must_have_coverage"]:.3f}  skill_jaccard={mi["skill_jaccard"]:.3f}  role_match={mi["role_match"]}  text_sim={mi["text_sim"]:.3f}')

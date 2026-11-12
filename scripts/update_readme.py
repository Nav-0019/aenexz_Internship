"""
Dynamic README Tracker Script
Scans the internship repository, updates progress metrics, and synchronizes README.md.
"""

import os
import re

ASSIGNMENTS = [
    {"name": "Python Core & Algorithmic Exercises", "file": "python_exercises.ipynb", "category": "Python Fundamentals"},
    {"name": "Exploratory Data Analysis (EDA)", "file": "visualizations/actual_vs_predicted.png", "category": "Exploratory Data Analysis (EDA)"},
    {"name": "Machine Learning: Diabetes Progression Prediction", "file": "diabetes_progression_prediction.ipynb", "category": "Supervised Learning (Regression)"},
    {"name": "Classification Algorithms", "file": "classification_project.ipynb", "category": "Classification Algorithms"},
    {"name": "Unsupervised Learning & Clustering", "file": "clustering_project.ipynb", "category": "Unsupervised Learning & Clustering"},
    {"name": "Internship Capstone Project", "file": "capstone_project.ipynb", "category": "Capstone Internship Project"}
]

def make_bar(percent, length=20):
    filled = int(length * (percent / 100))
    return f"[{'█' * filled}{'░' * (length - filled)}]"

def update_readme():
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        print("README.md not found.")
        return

    total = len(ASSIGNMENTS)
    completed = 0

    for item in ASSIGNMENTS:
        if os.path.exists(item["file"]):
            completed += 1

    overall_pct = int((completed / total) * 100)
    bar_str = make_bar(overall_pct, 20)

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Update overall completion line
    new_tracker = f"Overall Internship Completion: {bar_str} {overall_pct}%"
    updated_content = re.sub(
        r"Overall Internship Completion:\s*\[[█░]+\]\s*\d+%",
        new_tracker,
        content
    )

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print(f"Dynamic update complete: {completed}/{total} modules detected ({overall_pct}%).")

if __name__ == "__main__":
    update_readme()


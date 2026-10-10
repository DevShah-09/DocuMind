import sys
sys.path.insert(0, ".")
from app.services.local_classifier import local_classifier

# Simulate typical resume text sections
resume_texts = [
    # Short header only (worst case)
    "Dhruv Soni | dhruv.soni@gmail.com | +91 9876543210 | LinkedIn: linkedin.com/in/dhruvsoni",
    
    # Header + Education (typical first 300 chars)
    "Dhruv Soni\ndhruv@email.com | +91 9876543210\nOBJECTIVE\nSeeking a challenging position to leverage skills\nEDUCATION\nB.Tech Computer Science, CGPA: 8.5/10, 2023",
    
    # Header + Skills (what most resumes look like in first 500 chars)
    "Dhruv Soni | Software Engineer\nEmail: dhruv@gmail.com | GitHub: github.com/dhruvsoni\nSKILLS\nPython, Java, React, Node.js, Django, SQL, Machine Learning, Git, REST API\nEDUCATION\nB.Tech CSE, University of Mumbai, CGPA 8.7",
    
    # Full typical resume excerpt (1000+ chars)
    """Dhruv Soni
dhruv.soni@gmail.com | +91 9876543210 | LinkedIn | GitHub

OBJECTIVE
Seeking a challenging Software Engineer position to leverage my skills in Python and Machine Learning.

EDUCATION
B.Tech in Computer Science Engineering
University of Mumbai, 2020-2024 | CGPA: 8.7/10

WORK EXPERIENCE
Software Developer Intern — TCS, Mumbai (Jun 2023 – Aug 2023)
- Developed REST API endpoints using Django and PostgreSQL
- Implemented ML model for document classification achieving 94% accuracy
- Collaborated in Agile team of 8 engineers

SKILLS
Python, Java, JavaScript, React, Node.js, Django, Flask, SQL, TensorFlow, PyTorch, Git, AWS

PROJECTS
DocuMind AI — Document Intelligence Platform
Built using React, FastAPI, Qdrant vector DB, BGE embeddings
GitHub: github.com/dhruvsoni/documind

Certifications: AWS Cloud Practitioner, Google Data Analytics
References available upon request""",
]

SEP = "-" * 88
print("\nRESUME ANCHOR CALIBRATION TEST")
print(SEP)
print(f"{'Text length':<15} {'Predicted':<20} {'Top':>7}  {'Second':<18}  {'Margin':>7}  Decision")
print(SEP)

for text in resume_texts:
    r = local_classifier.classify(text)
    decision = "LOCAL ACCEPT" if r.accepted else "fallback->Qwen"
    label = f"{len(text)} chars"
    print(f"{label:<15} {r.category:<20} {r.top_score:>7.4f}  {r.second_score:<8.4f}  {r.margin:>7.4f}  {decision}")

print(SEP)

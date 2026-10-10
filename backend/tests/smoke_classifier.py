import sys
sys.path.insert(0, ".")
from app.services.local_classifier import local_classifier

tests = [
    ("electricity_bill", "Consumer No: 1234567890\nMSEDCL Monthly Bill\nPrevious Reading: 1250 Current Reading: 1488 Units Consumed: 238 Amount Payable Rs 1240 Due Date 25/09/2026"),
    ("pan_card",         "INCOME TAX DEPARTMENT Permanent Account Number Card Name RAHUL SHARMA Date of Birth 15/08/1995 PAN ABCDE1234F"),
    ("bank_statement",   "SBI BANK ACCOUNT STATEMENT Account Number 12345678901 Opening Balance 45200 Credits 20000 Salary Debits 5400 UPI Closing Balance 59800"),
    ("resume",           "Rahul Sharma Software Engineer 3 years experience EDUCATION BTech Computer Science 2020 EXPERIENCE Software Developer TCS SKILLS Python Java SQL React"),
    ("ambiguous_mixed",  "This document contains various information about policies and regulations. It references both financial data and legal clauses. The purpose is unclear."),
]

SEP = "-" * 88
print("\nBGE CLASSIFIER SMOKE TEST")
print(SEP)
print(f"{'Case Label':<20} {'Predicted':<20} {'Top':>7}  {'Second':>7}  {'Margin':>7}  Decision")
print(SEP)

for label, text in tests:
    r = local_classifier.classify(text)
    decision = "LOCAL ACCEPT" if r.accepted else "FALLBACK Qwen"
    print(f"{label:<20} {r.category:<20} {r.top_score:>7.4f}  {r.second_score:>7.4f}  {r.margin:>7.4f}  {decision}")

print(SEP)

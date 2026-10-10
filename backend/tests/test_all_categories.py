"""
Full calibration test for all 18 known document categories.
Tests each category with realistic text snippets and shows
which ones pass the local classifier vs fall through to Qwen.
"""
import sys
sys.path.insert(0, ".")
from app.services.local_classifier import local_classifier, SCORE_THRESHOLD, MARGIN_THRESHOLD

# Realistic document text samples (mimicking actual PDF content)
CATEGORY_TESTS = {
    "electricity_bill": (
        "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD\n"
        "Consumer No: 123456789012  Name: RAHUL SHARMA\n"
        "Meter No: WU4523891  Bill No: 2024-25/09/12345\n"
        "Billing Period: 01-AUG-2026 to 31-AUG-2026\n"
        "Previous Reading: 12450  Current Reading: 12688\n"
        "Units Consumed: 238  Amount: Rs. 1240.00\n"
        "Fixed Charges: 120.00  Energy Charges: 950.00\n"
        "Electricity Duty: 95.00  Fuel Adjustment: 75.00\n"
        "Due Date: 25-SEP-2026  Total Amount Payable: Rs. 1240.00"
    ),
    "water_bill": (
        "BRIHAN MUMBAI MUNICIPAL CORPORATION\n"
        "Water Bill  Consumer No: W-45678  Name: SURESH PATEL\n"
        "Property No: WARD-A/123  Billing Period: Q2 2026\n"
        "Previous Reading: 2340 KL  Current Reading: 2398 KL\n"
        "Units Consumed: 58 Kilolitres\n"
        "Water Charges: Rs. 580.00  Sewerage: Rs. 145.00\n"
        "Service Charges: Rs. 50.00  Total: Rs. 775.00\n"
        "Due Date: 30-SEP-2026"
    ),
    "gas_bill": (
        "MAHANAGAR GAS LIMITED\n"
        "Consumer No: 123456  Name: PRIYA MEHTA\n"
        "Billing Period: AUG 2026  Meter No: DL-78923\n"
        "Previous Reading: 456 SCM  Current Reading: 489 SCM\n"
        "Consumption: 33 SCM  Rate: Rs. 51.50/SCM\n"
        "Gas Charges: Rs. 1699.50  Fixed Charges: Rs. 100.00\n"
        "Total Amount Payable: Rs. 1799.50  Due: 20-SEP-2026"
    ),
    "mobile_bill": (
        "Airtel  Account Number: 9876543210\n"
        "Customer: AMIT KUMAR  Plan: Postpaid 499\n"
        "Billing Period: 01-AUG-2026 to 31-AUG-2026\n"
        "Monthly Rental: Rs. 499.00  Talk Time: Unlimited\n"
        "Data Used: 28.5 GB / 40 GB  SMS: 234\n"
        "Roaming Charges: Rs. 0.00  International: Rs. 0.00\n"
        "CGST: Rs. 44.91  SGST: Rs. 44.91\n"
        "Total Amount Due: Rs. 588.82  Due Date: 15-SEP-2026"
    ),
    "bank_statement": (
        "STATE BANK OF INDIA  ACCOUNT STATEMENT\n"
        "Account Holder: NEHA SHARMA  A/C No: 32145678901\n"
        "IFSC: SBIN0001234  Branch: Andheri West, Mumbai\n"
        "Statement Period: 01-AUG-2026 to 31-AUG-2026\n"
        "Opening Balance: Rs. 45,200.00\n"
        "Date        Narration              Chq/Ref    Debit     Credit    Balance\n"
        "02-AUG-26   UPI/Salary/TCS                             50000.00  95200.00\n"
        "05-AUG-26   NEFT/Rent Transfer     REF123456  15000.00           80200.00\n"
        "10-AUG-26   ATM WDL/Andheri                  5000.00            75200.00\n"
        "Closing Balance: Rs. 75,200.00"
    ),
    "invoice": (
        "TAX INVOICE\n"
        "Invoice No: INV-2026-0892  Date: 05-SEP-2026\n"
        "GSTIN: 27AADCS1234R1ZA  PAN: AADCS1234R\n"
        "Seller: ABC Technologies Pvt Ltd, Mumbai\n"
        "Buyer: XYZ Corp, Pune  GSTIN: 27XYZC5678K1ZB\n"
        "Item Description          HSN    Qty  Rate    Amount\n"
        "Software License          9983   1    50000   50000.00\n"
        "Annual Support            9983   1    10000   10000.00\n"
        "Taxable Value: Rs. 60,000  CGST @9%: 5400  SGST @9%: 5400\n"
        "Total Amount: Rs. 70,800.00"
    ),
    "receipt": (
        "PAYMENT RECEIPT\n"
        "Receipt No: RCP-20261234  Date: 10-SEP-2026\n"
        "Received From: MR. VIKRAM NAIR\n"
        "Amount Received: Rs. 25,000.00 (Rupees Twenty Five Thousand Only)\n"
        "Payment Mode: NEFT  Reference: NEFT20261234\n"
        "Being Payment Of: Advance for property booking\n"
        "Property: Flat No 302, Green Valley Apartments\n"
        "Balance Due: Rs. 75,000.00\n"
        "Authorized Signature  Stamp"
    ),
    "marksheet": (
        "UNIVERSITY OF MUMBAI\n"
        "MARK SHEET  Examination: B.Tech Semester VI\n"
        "Seat No: MU2024CSE1234  Roll No: 2021CSE045\n"
        "Student Name: POOJA DESAI  Division: First\n"
        "Subject Code  Subject Name              Max  Obtained  Grade\n"
        "CS601         Machine Learning           100   87        A\n"
        "CS602         Computer Networks          100   79        B+\n"
        "CS603         Database Management        100   91        O\n"
        "Total Marks: 530/600  Percentage: 88.33%  CGPA: 8.7\n"
        "Result: PASS  Class: DISTINCTION"
    ),
    "certificate": (
        "This is to certify that\n"
        "ARJUN KAPOOR\n"
        "has successfully completed the course\n"
        "MACHINE LEARNING FUNDAMENTALS\n"
        "Duration: 40 Hours  Grade: A+\n"
        "This Certificate is awarded in recognition of successful completion\n"
        "of the above-mentioned program on 15-SEP-2026\n"
        "Issued by: Coursera in association with Stanford University\n"
        "Certificate ID: CERT-ML-20261234\n"
        "Director, Certification Programs"
    ),
    "aadhaar": (
        "Government of India  UIDAI\n"
        "Aadhaar  1234 5678 9012\n"
        "Name: RAMESH KUMAR\n"
        "Date of Birth: 15/08/1985\n"
        "Male\n"
        "Address: 12, Gandhi Nagar, Kurla West, Mumbai - 400070, Maharashtra\n"
        "VID: 9876 5432 1234 5678\n"
        "Enrollment No: 1234/56789/12345\n"
        "This card is issued by UIDAI under Aadhaar Act 2016"
    ),
    "pan_card": (
        "INCOME TAX DEPARTMENT\n"
        "GOVT. OF INDIA\n"
        "Permanent Account Number Card\n"
        "ABCDE1234F\n"
        "Name: SANJAY GUPTA\n"
        "Father's Name: RAJESH GUPTA\n"
        "Date of Birth: 12/05/1990\n"
        "Signature\n"
        "Income Tax Department PAN Card"
    ),
    "passport": (
        "REPUBLIC OF INDIA  PASSPORT\n"
        "Type: P  Country Code: IND\n"
        "Passport No: A1234567\n"
        "Surname: SHARMA  Given Names: RAHUL KUMAR\n"
        "Nationality: Indian\n"
        "Date of Birth: 20/03/1995  Sex: M\n"
        "Place of Birth: Mumbai, Maharashtra\n"
        "Date of Issue: 15/01/2020  Date of Expiry: 14/01/2030\n"
        "P<INDSHARMA<<RAHUL<KUMAR<<<<<<<<<<<<<<<<<<<<<<<\n"
        "A1234567<3IND9503204M3001148<<<<<<<<<<<<<<<4"
    ),
    "insurance": (
        "LIFE INSURANCE CORPORATION OF INDIA\n"
        "Policy Schedule\n"
        "Policy No: 123456789  Policy Name: Jeevan Anand\n"
        "Insured: KAVITHA RAO  Date of Birth: 10/04/1988\n"
        "Nominee: RAJESH RAO (Husband)\n"
        "Sum Assured: Rs. 10,00,000.00\n"
        "Premium: Rs. 4,200.00 Quarterly\n"
        "Policy Term: 25 Years  Premium Paying Term: 25 Years\n"
        "Date of Commencement: 01-JAN-2020  Maturity Date: 01-JAN-2045\n"
        "Coverage: Death Benefit + Maturity Benefit"
    ),
    "medical_report": (
        "DR. LALPATHLAB  DIAGNOSTIC REPORT\n"
        "Patient Name: ANITA SINGH  Age: 34  Sex: Female\n"
        "Ref. Doctor: Dr. Sharma  Report No: LAB20261234\n"
        "Sample: Blood  Collection: 10-SEP-2026\n"
        "Test Name              Result    Normal Range    Unit\n"
        "Haemoglobin (Hb)       11.2      12.0-16.0       g/dL   LOW\n"
        "WBC Count              7800      4000-11000      /cumm  Normal\n"
        "Platelet Count         2.5 Lac   1.5-4.5 Lac     /cumm  Normal\n"
        "Blood Glucose (F)      95        70-100          mg/dL  Normal\n"
        "Creatinine             0.9       0.6-1.2         mg/dL  Normal\n"
        "Dr. Observations: Haemoglobin slightly low; advised dietary iron."
    ),
    "prescription": (
        "Dr. RAJESH MEHTA  MBBS, MD (Internal Medicine)\n"
        "Reg No: MCI/12345  Apollo Hospital, Mumbai\n"
        "Date: 10-SEP-2026  Patient: SURESH NAIR  Age: 45\n"
        "Rx\n"
        "1. Tab Metformin 500mg  Twice daily after meals  x 30 days\n"
        "2. Tab Atorvastatin 10mg  Once at night  x 30 days\n"
        "3. Tab Aspirin 75mg  Once daily after breakfast  x 30 days\n"
        "Advice: Low sugar diet, regular exercise, recheck HbA1c after 3 months\n"
        "Next Visit: 10-OCT-2026\n"
        "Signature & Stamp"
    ),
    "legal_document": (
        "THIS AGREEMENT is made on the 10th day of September 2026\n"
        "BETWEEN: ABC Properties Pvt Ltd (hereinafter referred to as 'Landlord')\n"
        "AND: Mr. Vikram Nair (hereinafter referred to as 'Tenant')\n"
        "WHEREAS the Landlord is the lawful owner of premises situated at\n"
        "Flat 302, Green Valley Apartments, Andheri West, Mumbai.\n"
        "NOW THIS AGREEMENT WITNESSETH as follows:\n"
        "1. The Landlord agrees to let and the Tenant agrees to take on lease\n"
        "the said premises for a period of 11 months commencing 01-OCT-2026.\n"
        "2. The monthly rent shall be Rs. 25,000 payable on or before 5th of each month.\n"
        "Signed and Witnessed by the parties hereto on the day and year first above written."
    ),
    "tax_document": (
        "FORM 16  CERTIFICATE UNDER SECTION 203\n"
        "TDS Certificate  Financial Year: 2025-26\n"
        "Name of Employer: TCS Ltd  TAN: MUMТ12345G\n"
        "Name of Employee: SHUBHAM SHAH  PAN: ABCDE1234F\n"
        "Designation: Software Engineer\n"
        "Gross Salary: Rs. 12,00,000  HRA Exemption: Rs. 1,20,000\n"
        "Standard Deduction: Rs. 50,000\n"
        "Gross Total Income: Rs. 10,30,000\n"
        "Deductions u/s 80C: Rs. 1,50,000  80D: Rs. 25,000\n"
        "Total Tax Payable: Rs. 98,500  TDS Deducted: Rs. 98,500"
    ),
    "resume": (
        "SHUBHAM SHAH\nshubham.shah@gmail.com | +91 9876543210 | LinkedIn | GitHub\n"
        "OBJECTIVE\nSeeking a challenging Software Engineer role to leverage expertise in Python and ML.\n"
        "EDUCATION\nB.Tech Computer Science  |  DJ Sanghvi College  |  2020-2024  |  CGPA: 8.9\n"
        "WORK EXPERIENCE\nSoftware Engineer Intern — TCS, Mumbai  (Jun 2023 – Sep 2023)\n"
        "- Built REST API using Django and PostgreSQL, reducing latency by 30%\n"
        "- Implemented ML pipeline using scikit-learn and TensorFlow\n"
        "SKILLS\nPython, Java, JavaScript, React, Node.js, Django, Flask, SQL, Git, AWS, Docker\n"
        "PROJECTS\nDocuMind AI: Document Intelligence Platform using FastAPI, Qdrant, BGE embeddings\n"
        "Certifications: AWS Cloud Practitioner, Google Data Analytics\n"
        "References available upon request"
    ),
}

SEP = "=" * 90
print("\nFULL 18-CATEGORY CALIBRATION TEST")
print(f"SCORE_THRESHOLD={SCORE_THRESHOLD}  MARGIN_THRESHOLD={MARGIN_THRESHOLD}")
print(SEP)
print(f"{'Category':<20} {'Predicted':<20} {'Score':>6}  {'2nd Score':>9}  {'Margin':>7}  Decision")
print("-" * 90)

passed = 0
failed = 0
fallback = 0

for category, text in CATEGORY_TESTS.items():
    r = local_classifier.classify(text, max_chars=2500)
    correct = (r.category == category)
    decision = "LOCAL OK" if r.accepted else "->Qwen"
    marker = "[OK]" if correct else "[WRONG]"
    if r.accepted and correct:
        passed += 1
    elif r.accepted and not correct:
        failed += 1
    else:
        fallback += 1
    print(
        f"{category:<20} {r.category:<20} {r.top_score:>6.4f}  {r.second_score:>9.4f}  "
        f"{r.margin:>7.4f}  {decision} {marker}"
    )

print(SEP)
print(f"LOCAL ACCEPT correct  : {passed}/18")
print(f"LOCAL ACCEPT wrong    : {failed}/18  <-- these need better anchors")
print(f"Fallback to Qwen      : {fallback}/18")
print(f"API calls saved       : {passed}/{passed+fallback+failed} ({100*passed/18:.0f}%)")

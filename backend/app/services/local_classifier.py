"""
local_classifier.py — BGE Multi-Prototype Local Document Classifier
====================================================================
Uses the already-loaded BAAI/bge-base-en-v1.5 embedding model to classify
documents without an API call.

Design:
  - Multiple prototype sentences per category (3 per category minimum)
  - Per-category score = max(cosine_sim(doc_vec, proto)) across that category's protos
  - Confidence gate: top_score >= SCORE_THRESHOLD AND margin >= MARGIN_THRESHOLD
  - If confident  → return category immediately, skip Qwen-72B
  - If uncertain  → return None, caller falls back to Qwen-72B

Thresholds are intentionally conservative defaults.
Run tests/calibrate_classifier.py against your real document set to tune them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Confidence Gate (tune via tests/calibrate_classifier.py) ─────────────────
# Calibration notes (BAAI/bge-base-en-v1.5):
#   pan_card        -> ~0.88  (short, highly distinctive vocab)
#   electricity_bill-> ~0.78  (utility-specific terms)
#   bank_statement  -> ~0.72  (transaction vocabulary)
#   resume          -> ~0.60-0.69  (rich but diverse vocabulary)
# The MARGIN check is the primary quality gate; SCORE is just a noise floor.
SCORE_THRESHOLD  = 0.62   # Floor above random noise (~0.50); margin is the real gate
MARGIN_THRESHOLD = 0.10   # Minimum gap vs second-best (raised from 0.08 for safety)

# ── Category Prototype Sentences ──────────────────────────────────────────────
# IMPORTANT: Write anchors in document-content style — as if they are lines
# FROM the document, not descriptions OF the document.
# BGE embedding space puts actual document text close to other actual document
# text, not to meta-descriptions. Descriptive anchors cause low cosine scores.
# Each anchor is embedded once at startup and cached.
# Per-category score = MAX cosine similarity across its anchors.
CATEGORY_ANCHORS: Dict[str, List[str]] = {
    "electricity_bill": [
        # Exclusive: utility company names + kWh/unit-specific vocabulary
        "MSEDCL BESCOM TNEB TPDDL Electricity Bill kWh Units Consumed Sanctioned Load",
        "Power Bill Previous Reading Current Reading Units kWh Electricity Duty Fuel Adjustment",
        "Energy Charges Fixed Charges Electricity Tariff Load kWh Consumption Surcharge",
        # Grid/distribution specific
        "Distribution Company Consumer Number Electricity Meter Billing Period Power Supply",
        "Connected Load Sanctioned Load Power Factor Units Consumed Arrears Electricity Bill",
    ],
    "water_bill": [
        # Exclusive: water-specific units and municipal authority
        "Water Bill Kilolitres KL Water Consumption Sewerage Charges BMC Municipal Corporation",
        "Municipal Water Supply Meter Reading Kilolitres Sewerage Service Charges Water Tax",
        "Water Connection Consumer Water Usage KL Water Charges Drainage Cess Water Department",
    ],
    "gas_bill": [
        # Exclusive: gas-specific units and company names
        "Mahanagar Gas MGL IGL Gas Bill SCM Cubic Meter PNG Piped Natural Gas Consumption",
        "PNG Gas Bill Meter Reading SCM Standard Cubic Meter Gas Charges Subsidy Cylinder",
        "LPG PNG Gas Connection Consumer Cylinder Consumption Gas Tariff Subsidy Amount",
    ],
    "mobile_bill": [
        # Exclusive: telecom/data vocabulary
        "Airtel Jio Vi Vodafone BSNL Mobile Bill GB Data Postpaid Plan Talk Time SMS",
        "Telecom Bill Data Usage GB Outgoing Calls SMS Balance Rental Roaming Recharge",
        "Mobile Postpaid Account Bill Data Consumed Talk Time Unlimited SMS International Roaming",
    ],
    "bank_statement": [
        # Exclusive: IFSC/NEFT/transaction-table vocabulary
        "IFSC Code NEFT IMPS RTGS UPI Account Statement Narration Withdrawal Deposit Closing Balance",
        "Bank Account Statement Chq Ref No Debit Credit NEFT Transfer ATM Withdrawal Opening Closing Balance",
        "SBI HDFC ICICI Axis Kotak Bank Statement Transactions NEFT UPI IMPS RTGS ATM Salary Credit",
        "Date Narration Cheque Ref Debit Credit Running Balance NEFT IMPS UPI ATM EMI Salary",
        "IFSC Branch Account Holder Statement Period Opening Balance Closing Balance Transaction History",
    ],
    "invoice": [
        "Tax Invoice GSTIN HSN SAC Code Taxable Value CGST SGST IGST Total Amount Due",
        "Invoice Number Date Bill To Ship To Qty Unit Price GSTIN Total GST Invoice",
        "Invoice GSTIN HSN Item Description Rate GST CGST SGST Total Amount Due",
    ],
    "receipt": [
        # Exclusive: "received from" / "being payment of" phrasing
        "Receipt Received From Amount Rupees Being Payment Of Balance Due Acknowledgement",
        "Payment Receipt No Received From Amount Rs Being Payment Of Advance Booking",
        "Receipt Cash Cheque NEFT Received From Amount Being Payment Authorized Signature Stamp",
    ],
    "marksheet": [
        # Exclusive: exam result table vocabulary — marks numbers and academic grading
        "Marksheet Roll No Seat No Subject Code Marks Obtained Out of Maximum Marks Grade Pass Fail",
        "Examination Result Theory Practical Internal External Total Percentage CGPA SGPA Grade",
        "Mark Sheet Marks Obtained Maximum Marks Subject Wise Result Distinction First Division",
        # Specific grading vocabulary
        "Subject Code Subject Name Max Marks Marks Obtained Grade Points CGPA Percentage Class",
    ],
    "certificate": [
        # Exclusive: formal award/recognition language — NOT training/course (which overlaps resume)
        "This is to certify that Name has been awarded Certificate of Excellence Achievement",
        "Certificate of Participation Completion awarded to Name by Institute Organization Date",
        "This Certificate is hereby granted awarded recognized Certificate Number Valid",
        # ID/verification lines unique to certificates
        "Certificate ID Verify Certificate at Certificate Number Issued by Authorized Signatory Seal",
    ],
    "aadhaar": [
        # Exclusive: UIDAI/biometric vocabulary
        "Aadhaar UIDAI 12 digit Unique Identification Number Biometric Enrollment QR Code VID",
        "UIDAI Unique Identification Authority India Aadhaar Enrollment Biometric Fingerprint Iris",
        "Aadhaar Card Government India UIDAI UID Biometric VID Virtual ID QR Enrollment Slip",
    ],
    "pan_card": [
        "Income Tax Department Permanent Account Number PAN Name Date of Birth",
        "INCOME TAX DEPARTMENT GOVT OF INDIA Permanent Account Number",
        "PAN Card Name Father Name Date of Birth Signature PAN Number",
    ],
    "passport": [
        "Passport Republic of India Surname Given Names Nationality Date of Birth",
        "Passport Number Place of Birth Date of Issue Date of Expiry",
        "REPUBLIC OF INDIA Passport Holder Name Nationality IND MRZ Machine Readable",
    ],
    "insurance": [
        "Insurance Policy Number Insured Name Sum Assured Premium Due Date",
        "Policy Schedule Life Health Motor Insurance Coverage Period Premium Amount",
        "Certificate of Insurance Policy Holder Nominee Coverage Maturity Benefits",
    ],
    "medical_report": [
        "Patient Name Age Sex Lab Report Test Name Result Reference Range Unit",
        "Blood Test Haemoglobin WBC RBC Platelet Glucose Creatinine Report",
        "Pathology Report Radiology Diagnosis Doctor Observations Clinical Findings",
    ],
    "prescription": [
        # Exclusive: Rx + drug-dosage vocabulary
        "Rx Prescription Tab Cap Syrup Dosage Twice Daily Once Daily After Food Before Food",
        "Prescribed by Dr Medicine Name Dosage mg Frequency Duration Refill Rx Patient Age",
        "Prescription Doctor Name Reg No Hospital Tab Capsule Medicine Dosage Twice Daily Next Visit",
    ],
    "legal_document": [
        # Exclusive: legal boilerplate phrases
        "This Agreement made between hereinafter referred to as Landlord Tenant Party Parties",
        "Whereas the party hereinafter referred to NOW THIS AGREEMENT WITNESSETH as follows",
        "Affidavit I solemnly sincerely affirm declare depose true correct best knowledge belief",
        # Deed/notice vocabulary
        "Legal Deed Notice Agreement Covenant Clause Jurisdiction Arbitration Signed Witnessed Stamp",
    ],
    "tax_document": [
        "Form 16 TDS Certificate Employer Employee PAN Salary Income Tax Deducted",
        "Income Tax Return ITR Assessment Year Gross Total Income Deductions Tax",
        "GST Return GSTR Filing Taxable Turnover Output Tax Input Tax Credit",
    ],
    "student_id_card": [
        "Student ID Card College Identity Card University Student Roll No Enrollment Number Department Branch",
        "Identity Card Student Name Roll No Course Branch Academic Year Valid Upto Principal Signature",
        "College Student Identity Card Library Card Student ID Department Registration Number Campus",
        "School Identity Card School Student ID Admission No Class Section Roll No School Name Parent Contact",
    ],
    "research_paper": [
        "Abstract Introduction Methodology System Architecture Experiments Results Conclusion References Discussion",
        "Research Paper Conference Journal IEEE ACM arXiv DOI Authors Affiliation Abstract Section Figure Table",
        "Proposed System Evaluation Performance Benchmark Dataset Experimental Results Abstract Keywords References",
    ],
    "report": [
        "Project Report Executive Summary Technical Report Table of Contents Overview Findings Recommendations Analysis",
        "Final Report Status Report Performance Analysis System Specifications Functional Requirements Architecture",
    ],
    "resume": [
        # Header — contact details unique to resumes
        "Resume CV Curriculum Vitae Email LinkedIn GitHub Portfolio Objective Career Summary",
        # Education — degrees and academic performance
        "B.Tech B.E B.Sc M.Tech M.Sc Graduation CGPA GPA University College Dean's List",
        # Professional experience — unique to resumes
        "Work Experience Internship Software Engineer Developer Analyst Role Responsibilities Achievements",
        # Technical skills — never on ID cards
        "Skills Python Java JavaScript React Node.js Django Flask SQL TensorFlow PyTorch Git",
        # Projects — unique to resumes
        "Projects Built Developed Implemented Designed REST API Machine Learning GitHub Repository",
        # Achievements / certifications
        "Certifications AWS Google Cloud Coursera HackerRank LeetCode Competitive Programming Awards",
        # Objective / profile summary — resume-specific prose
        "Seeking challenging position leverage skills experience contribute organisation growth",
        # References line common in resumes
        "References available upon request Hobbies Interests Languages Known Declaration",
    ],
}

ALL_CATEGORIES: List[str] = list(CATEGORY_ANCHORS.keys())


import os
import json

LEARNED_PROTOTYPES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "learned_prototypes.json")
)

def load_learned_prototypes() -> Dict[str, List[str]]:
    if os.path.exists(LEARNED_PROTOTYPES_PATH):
        try:
            with open(LEARNED_PROTOTYPES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("[CLASSIFIER] Failed to load learned prototypes: %s", e)
    return {}

def save_learned_prototype_to_disk(category: str, text_snippet: str):
    try:
        os.makedirs(os.path.dirname(LEARNED_PROTOTYPES_PATH), exist_ok=True)
        data = load_learned_prototypes()
        if category not in data:
            data[category] = []
        if text_snippet not in data[category]:
            data[category].append(text_snippet)
            with open(LEARNED_PROTOTYPES_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("[CLASSIFIER] Saved dynamic prototype for '%s' to disk.", category)
    except Exception as e:
        logger.error("[CLASSIFIER] Failed to save learned prototype: %s", e)


@dataclass
class ClassifierResult:
    """Full diagnostic output from the local classifier."""
    category: str
    top_score: float
    second_score: float
    margin: float
    accepted: bool          # True → skip Qwen; False → fall through to Qwen-72B
    text_chars_used: int    # How many leading chars of doc text were embedded


class LocalBGEClassifier:
    """
    Zero-shot multi-prototype document classifier reusing BAAI/bge-base-en-v1.5.
    Supports dynamic self-learning: unknown documents classified by Qwen-72B
    automatically train new prototype vectors in memory and on disk.
    """

    _instance: "LocalBGEClassifier | None" = None

    def __new__(cls) -> "LocalBGEClassifier":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._proto_cache: Optional[Dict[str, List[np.ndarray]]] = None
        return cls._instance

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_model(self):
        """Return the already-initialised BGE singleton — no second model load."""
        from app.embeddings.embedding_service import embedding_service
        embedding_service._initialize()
        return embedding_service._model

    def _build_prototype_cache(self) -> Dict[str, List[np.ndarray]]:
        """Embed static + learned prototype sentences once and cache as numpy arrays."""
        model = self._get_model()
        dynamic_dict = load_learned_prototypes()

        all_categories = list(set(ALL_CATEGORIES + list(dynamic_dict.keys())))
        all_sentences: List[str] = []
        category_lens: List[int] = []
        ordered_categories: List[str] = []

        for cat in all_categories:
            protos = CATEGORY_ANCHORS.get(cat, []).copy()
            if cat in dynamic_dict:
                protos.extend(dynamic_dict[cat])
            if protos:
                all_sentences.extend(protos)
                category_lens.append(len(protos))
                ordered_categories.append(cat)

        logger.info(
            "[CLASSIFIER] Embedding %d prototype sentences for %d categories (static + learned)...",
            len(all_sentences), len(ordered_categories),
        )

        vectors = model.encode(
            all_sentences,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        cache: Dict[str, List[np.ndarray]] = {}
        idx = 0
        for cat, n in zip(ordered_categories, category_lens):
            cache[cat] = [vectors[idx + i] for i in range(n)]
            idx += n

        logger.info("[CLASSIFIER] Prototype cache ready (%d categories).", len(cache))
        return cache

    def learn_prototype(self, category: str, text_snippet: str):
        """
        Dynamically train/adapt local classifier on a newly categorized document.
        Updates in-memory vector cache and persists to disk.
        """
        if not category or category == "miscellaneous" or not text_snippet:
            return
        
        snippet = text_snippet[:400].strip()
        if len(snippet) < 15:
            return

        cache = self._ensure_cache()
        model = self._get_model()

        # Check if snippet or similar vector already exists to avoid duplication
        vec = model.encode([snippet], batch_size=1, normalize_embeddings=True, show_progress_bar=False)[0]

        if category not in cache:
            cache[category] = []
            
        # Avoid duplicate vectors (cosine sim > 0.96)
        is_duplicate = any(self._cosine(vec, existing_v) > 0.96 for existing_v in cache[category])
        if not is_duplicate:
            cache[category].append(vec)
            save_learned_prototype_to_disk(category, snippet)
            logger.info(
                "[CLASSIFIER] Self-Learning Success! Dynamically trained prototype for category '%s'. Total vectors: %d",
                category, len(cache[category])
            )

    def reset_cache(self):
        """Force rebuild of prototype embedding cache."""
        self._proto_cache = self._build_prototype_cache()

    def _ensure_cache(self) -> Dict[str, List[np.ndarray]]:
        if self._proto_cache is None:
            self._proto_cache = self._build_prototype_cache()
        return self._proto_cache

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity for already-normalised unit vectors (dot product)."""
        return float(np.dot(a, b))

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(
        self,
        text: str,
        max_chars: int = 1500,
        score_threshold: float = SCORE_THRESHOLD,
        margin_threshold: float = MARGIN_THRESHOLD,
    ) -> ClassifierResult:
        """
        Classify a document text string locally.

        Args:
            text:             Raw extracted document text.
            max_chars:        Leading characters to embed (default 1500).
            score_threshold:  Minimum top-category cosine sim to accept locally.
            margin_threshold: Minimum gap vs second-best category to accept.

        Returns:
            ClassifierResult with full diagnostics.
            result.accepted == True  → use result.category, skip Qwen-72B.
            result.accepted == False → fall through to Qwen-72B.
        """
        cache = self._ensure_cache()
        model = self._get_model()

        snippet = text[:max_chars].strip()
        if not snippet:
            logger.warning("[CLASSIFIER] Empty text — cannot classify locally.")
            return ClassifierResult(
                category="", top_score=0.0, second_score=0.0,
                margin=0.0, accepted=False, text_chars_used=0,
            )

        # Embed the document snippet (doc-side — no BGE instruction prefix)
        doc_vec: np.ndarray = model.encode(
            [snippet],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # Per-category score = max cosine sim across that category's prototypes
        category_scores: Dict[str, float] = {
            cat: max(self._cosine(doc_vec, pv) for pv in proto_vecs)
            for cat, proto_vecs in cache.items()
        }

        # Rank by descending score
        ranked: List[Tuple[float, str]] = sorted(
            ((score, cat) for cat, score in category_scores.items()),
            reverse=True,
        )

        top_score,    top_cat    = ranked[0]
        second_score, second_cat = ranked[1]
        margin = top_score - second_score

        accepted = (top_score >= score_threshold) and (margin >= margin_threshold)

        logger.info(
            "[CLASSIFIER] top=%s (%.4f) second=%s (%.4f) margin=%.4f → %s",
            top_cat, top_score, second_cat, second_score, margin,
            "ACCEPTED" if accepted else "FALLBACK→Qwen-72B",
        )

        return ClassifierResult(
            category=top_cat,
            top_score=top_score,
            second_score=second_score,
            margin=margin,
            accepted=accepted,
            text_chars_used=len(snippet),
        )


# ── Module-level singleton ────────────────────────────────────────────────────
local_classifier = LocalBGEClassifier()

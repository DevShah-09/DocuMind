"""
query_parser.py — Natural Language Query Parser
===============================================
Converts free-text queries into structured filter objects using Qwen 2.5
(text-only call via OpenRouter) with regex fallback.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from openai import OpenAI

from app.core.config import settings
from app.schemas.retrieval import AmountFilter, ParsedQuery

logger = logging.getLogger(__name__)

DOC_TYPE_MAP: Dict[str, str] = {
    # Electricity
    "electricity bill": "electricity_bill", "electricity": "electricity_bill",
    "electric bill": "electricity_bill", "electric": "electricity_bill",
    "light bill": "electricity_bill", "power bill": "electricity_bill",
    # Water
    "water bill": "water_bill", "water": "water_bill",
    # Gas
    "gas bill": "gas_bill", "gas": "gas_bill", "lpg": "gas_bill", "png": "gas_bill",
    # Mobile
    "mobile bill": "mobile_bill", "phone bill": "mobile_bill",
    "mobile": "mobile_bill", "telephone bill": "mobile_bill",
    # Bank
    "bank statement": "bank_statement", "bank": "bank_statement",
    "statement": "bank_statement", "passbook": "bank_statement",
    "account statement": "bank_statement",
    # Invoice / Receipt
    "invoice": "invoice", "bill": "invoice",
    "receipt": "receipt", "payment receipt": "receipt",
    # Tax
    "tax": "tax_document", "tax document": "tax_document",
    "form 16": "tax_document", "itr": "tax_document", "tds": "tax_document",
    "income tax": "tax_document", "gst": "tax_document",
    # Aadhaar — all common Indian spellings
    "aadhaar": "aadhaar", "aadhar": "aadhaar", "adhar": "aadhaar",
    "aadhar card": "aadhaar", "aadhaar card": "aadhaar", "adhar card": "aadhaar",
    "uid": "aadhaar", "uidai": "aadhaar",
    # PAN
    "pan card": "pan_card", "pan": "pan_card",
    "permanent account number": "pan_card",
    # Passport
    "passport": "passport",
    # Insurance
    "insurance": "insurance", "policy": "insurance", "insurance policy": "insurance",
    # Medical
    "medical report": "medical_report", "medical": "medical_report",
    "lab report": "medical_report", "blood test": "medical_report",
    "pathology": "medical_report", "health report": "medical_report",
    # Prescription
    "prescription": "prescription", "rx": "prescription", "medicine": "prescription",
    "doctor note": "prescription", "doctor prescription": "prescription",
    # Marksheet
    "marksheet": "marksheet", "mark sheet": "marksheet",
    "marks": "marksheet", "result": "marksheet", "exam result": "marksheet",
    "grade card": "marksheet", "report card": "marksheet",
    # Certificate
    "certificate": "certificate", "certification": "certificate",
    # Resume / CV
    "resume": "resume", "cv": "resume", "curriculum vitae": "resume",
    "biodata": "resume",
    # Student / College / School ID Card
    "college id": "student_id_card", "college id card": "student_id_card",
    "student id": "student_id_card", "student id card": "student_id_card",
    "school id": "student_id_card", "school id card": "student_id_card",
    "school card": "student_id_card", "college card": "student_id_card",
    "id card": "student_id_card", "identity card": "student_id_card",
    "student card": "student_id_card",
    # Research Paper / Academic Report
    "research paper": "research_paper", "paper": "research_paper",
    "research": "research_paper", "academic paper": "research_paper",
    "report": "report", "project report": "report",
}

MONTH_MAP: Dict[str, str] = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "oct": "October", "nov": "November", "dec": "December",
    "january": "January", "february": "February", "march": "March",
    "april": "April", "june": "June", "july": "July", "august": "August",
    "september": "September", "october": "October", "november": "November",
    "december": "December",
}

_YEAR_RE  = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september"
    r"|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(r"([₹$€£]?\s*\d[\d,]*(?:\.\d{1,2})?)", re.IGNORECASE)
_ABOVE_RE  = re.compile(r"\b(above|over|more than|greater than)\b", re.IGNORECASE)
_BELOW_RE  = re.compile(r"\b(below|under|less than)\b", re.IGNORECASE)

SYSTEM_PROMPT = """Parse user query into JSON. Return ONLY JSON:
{"document_type": "<slug or null>", "year": <int or null>, "month": "<month or null>", "entities": ["<names>"], "amount_filters": null, "semantic_query": "<search text>"}
Slugs: electricity_bill, water_bill, gas_bill, mobile_bill, bank_statement, invoice, receipt, marksheet, certificate, aadhaar, pan_card, passport, insurance, medical_report, prescription, legal_document, tax_document, resume, student_id_card.
Map aadhar/adhar->aadhaar, pan->pan_card, cv->resume, college/student id->student_id_card."""


class QueryParser:
    """Parses NL queries into structured ParsedQuery using Qwen + regex fallback."""

    def __init__(self) -> None:
        self._client = OpenAI(
            base_url=settings.OPENROUTER_URL,
            api_key=settings.OPENROUTER_API_KEY,
        )

    async def parse(self, query: str) -> ParsedQuery:
        import asyncio
        loop = asyncio.get_event_loop()
        try:
            raw = await loop.run_in_executor(None, self._call_llm, query)
            parsed = self._validate(raw, query)
            logger.info("[PARSER] LLM parsed query: %s", parsed.model_dump())
            return parsed
        except Exception as exc:
            logger.warning("[PARSER] LLM failed (%s) — using regex fallback", exc)
            return self._regex_fallback(query)

    def _call_llm(self, query: str) -> str:
        response = self._client.chat.completions.create(
            model="qwen/qwen-2.5-7b-instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            max_tokens=80,
            temperature=0.0,
        )
        return response.choices[0].message.content

    def _validate(self, raw: str, query: str) -> ParsedQuery:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        data: Dict[str, Any] = json.loads(raw)

        doc_type = self._norm_doc_type(data.get("document_type"))
        if not doc_type:
            # Fallback 1: check if LLM extracted the document type into entities
            for ent in (data.get("entities") or []):
                matched = self._norm_doc_type(str(ent))
                if matched:
                    doc_type = matched
                    break
        if not doc_type:
            # Fallback 2: regex/substring search on raw user query
            query_lower = query.lower()
            for phrase, slug in sorted(DOC_TYPE_MAP.items(), key=lambda x: -len(x[0])):
                if phrase in query_lower:
                    doc_type = slug
                    break

        month = data.get("month")
        if isinstance(month, str):
            month = MONTH_MAP.get(month.lower().strip(), month.title())
        year = data.get("year")
        if year is not None:
            try:
                year = int(year)
                if not (1950 <= year <= 2099):
                    year = None
            except (TypeError, ValueError):
                year = None

        amount_filter = None
        af = data.get("amount_filters")
        if af and isinstance(af, dict):
            try:
                amount_filter = AmountFilter(
                    operator=af.get("operator", "gt"),
                    value=float(af.get("value", 0)),
                    currency=af.get("currency"),
                )
            except Exception:
                pass

        sem_q = str(data.get("semantic_query") or query).strip()
        if len(sem_q.split()) <= 1 and len(query.split()) > 1:
            sem_q = query.strip()

        return ParsedQuery(
            document_type=doc_type,
            year=year,
            month=month,
            entities=[str(e) for e in (data.get("entities") or [])],
            amount_filters=amount_filter,
            semantic_query=sem_q,
            raw_query=query,
        )

    def _regex_fallback(self, query: str) -> ParsedQuery:
        query_lower = query.lower()

        year = None
        m = _YEAR_RE.search(query)
        if m:
            year = int(m.group(1))

        month = None
        m = _MONTH_RE.search(query_lower)
        if m:
            month = MONTH_MAP.get(m.group(1).lower(), m.group(1).title())

        doc_type = None
        for phrase, slug in sorted(DOC_TYPE_MAP.items(), key=lambda x: -len(x[0])):
            if phrase in query_lower:
                doc_type = slug
                break

        amount_filter = None
        m = _AMOUNT_RE.search(query_lower)
        if m:
            raw_val = re.sub(r"[₹$€£,\s]", "", m.group(1))
            try:
                value = float(raw_val)
                op = "gt" if _ABOVE_RE.search(query) else ("lt" if _BELOW_RE.search(query) else "gte")
                currency = "₹" if ("₹" in query or "rs" in query_lower) else None
                amount_filter = AmountFilter(operator=op, value=value, currency=currency)
            except ValueError:
                pass

        # Extract entities from possessives (e.g. "Shubham's resume" -> "shubham")
        entities = [m.lower() for m in re.findall(r"\b([A-Za-z]+)'s\b", query)]

        semantic = query
        if year:
            semantic = re.sub(str(year), "", semantic)
        semantic = re.sub(r"\s+", " ", semantic).strip()

        return ParsedQuery(
            document_type=doc_type, year=year, month=month, entities=entities,
            amount_filters=amount_filter,
            semantic_query=semantic or query,
            raw_query=query,
        )

    @staticmethod
    def _norm_doc_type(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        slug = raw.lower().strip().replace("-", "_").replace(" ", "_")
        if slug in DOC_TYPE_MAP.values():
            return slug
        for phrase, s in sorted(DOC_TYPE_MAP.items(), key=lambda x: -len(x[0])):
            if phrase in raw.lower():
                return s
        return None


query_parser = QueryParser()

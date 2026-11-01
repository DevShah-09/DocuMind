import base64
from openai import AsyncOpenAI
from app.core.config import settings

class VisionProcessor:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )

    async def analyze_text_document(self, extracted_text: str, document_name: str):
        prompt = f"""Analyze the document text from '{document_name}'.

Classification Rules:
1. Available Document Types: [electricity_bill, water_bill, gas_bill, mobile_bill, bank_statement, invoice, receipt, marksheet, certificate, aadhaar, pan_card, passport, insurance, medical_report, prescription, legal_document, tax_document, resume, student_id_card, research_paper, report]
2. Report vs. Research Paper Distinction:
   - "report": Project reports, course submissions (e.g., Innovative Product Development / IPD, Capstone, Final Year project), technical reports, thesis, or documents with "Report" in the filename or title. Summarize accurately as a "project report" or "technical report", NOT a research paper.
   - "research_paper": Formal peer-reviewed academic papers published in journals or conferences (e.g. IEEE, ACM, arXiv) with Abstract, Methodology, Experiments, Literature Review, and DOI/Conference headers.
3. If this single file contains MULTIPLE different document types merged together (e.g. an electricity bill + a resume + a PAN card):
   - Set "category": "miscellaneous"
   - List ALL detected document types in "tags": ["electricity_bill", "resume", "pan_card", ...]
4. If this file contains a single coherent document:
   - Set "category" to its specific slug (e.g. "report", "research_paper", "bank_statement", "resume").
   - Set "tags" to [category].

Extract key metadata fields (dates, amounts, person names, IDs) into a concise dictionary.
Write a short searchable summary (2-3 sentences max).
Extract all visible text.

Return ONLY a single top-level JSON object with keys: is_blur, blur_reason, category, tags, confidence, metadata, summary, text."""

        try:
            response = await self.client.chat.completions.create(
                model="qwen/qwen-2.5-72b-instruct",
                messages=[
                    {"role": "system", "content": "You are a document extraction assistant. Respond ONLY with a single top-level valid JSON object."},
                    {"role": "user", "content": f"{prompt}\n\nDOCUMENT TEXT:\n{extracted_text[:4000]}"}
                ],
                response_format={"type": "json_object"},
                max_tokens=2500,
                temperature=0.0,
            )
            result_str = response.choices[0].message.content
            import json
            data = json.loads(result_str)
            data["text"] = extracted_text  # always store full text
            return json.dumps(data)
        except Exception as e:
            print(f"Text Analysis Qwen-72B API Error: {e}, falling back to Gemini 2.5 Flash...")
            try:
                response = await self.client.chat.completions.create(
                    model="google/gemini-2.5-flash",
                    messages=[
                        {"role": "system", "content": "You are a document extraction assistant. Respond ONLY with a single top-level valid JSON object."},
                        {"role": "user", "content": f"{prompt}\n\nDOCUMENT TEXT:\n{extracted_text[:4000]}"}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=2500,
                    temperature=0.0,
                )
                result_str = response.choices[0].message.content
                import json
                data = json.loads(result_str)
                data["text"] = extracted_text
                return json.dumps(data)
            except Exception as e2:
                print(f"Text Analysis Fallback Error: {e2}")
                import json
                fallback_json = {
                    "category": "miscellaneous",
                    "tags": ["miscellaneous"],
                    "confidence": 70,
                    "metadata": {},
                    "summary": extracted_text[:300],
                    "text": extracted_text
                }
                return json.dumps(fallback_json)

    async def analyze_document(self, images_base64: list[str]):
        prompt = """Analyze this document image collection (which may contain multiple pages).

Step 1 — Quality: Is it too blurry/dark to read? If yes: is_blur=true, explain in blur_reason.
Step 2 — Classification & Multi-Doc Bundles:
- Available types: [electricity_bill, water_bill, gas_bill, mobile_bill, bank_statement, invoice, receipt, marksheet, certificate, aadhaar, pan_card, passport, insurance, medical_report, prescription, legal_document, tax_document, resume, student_id_card, research_paper, report]
- Report vs. Research Paper Distinction:
    * "report": Project reports, course submissions (e.g., IPD report, Capstone, Final Year project), technical reports, thesis, or documents with "Report" in the filename or title. Summarize accurately as a "project report" or "technical report", NOT a research paper.
    * "research_paper": Formal peer-reviewed academic papers (IEEE, ACM, arXiv) with Abstract, Methodology, DOI, Literature Review.
- If this file contains MULTIPLE different document types merged together:
    * Set "category": "miscellaneous"
    * List ALL detected document types in "tags": [...]
- If it is ONE coherent document:
    * Set "category" to its specific slug (e.g. "report", "research_paper", "student_id_card").
    * Set "tags": [category].

Step 3 — Extract key metadata fields (person names, dates, IDs, organizations).
Step 4 — Write a short searchable summary.
Step 5 — Extract all visible text across the pages into the "text" field.

CRITICAL: Return ONLY a single top-level JSON object with keys: is_blur, blur_reason, category, tags, confidence, metadata, summary, text."""

        content_array = [{"type": "text", "text": prompt}]
        for b64_img in images_base64:
            content_array.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})

        try:
            response = await self.client.chat.completions.create(
                model="qwen/qwen2.5-vl-72b-instruct",
                messages=[{"role": "user", "content": content_array}],
                response_format={"type": "json_object"},
                max_tokens=2500,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Vision Qwen-VL API Error: {e}, trying fallback Gemini 2.5 Flash...")
            try:
                response = await self.client.chat.completions.create(
                    model="google/gemini-2.5-flash",
                    messages=[{"role": "user", "content": content_array}],
                    response_format={"type": "json_object"},
                    max_tokens=2500,
                )
                return response.choices[0].message.content
            except Exception as e2:
                print(f"Vision API Fallback Error: {e2}")
                import json
                fallback_json = {
                    "category": "miscellaneous",
                    "tags": ["miscellaneous"],
                    "confidence": 0,
                    "metadata": {},
                    "summary": "Failed to process image during analysis.",
                    "text": ""
                }
                return json.dumps(fallback_json)
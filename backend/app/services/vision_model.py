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

Classify into one of: [electricity_bill, water_bill, gas_bill, mobile_bill, bank_statement, invoice, receipt, marksheet, certificate, aadhaar, pan_card, passport, insurance, medical_report, prescription, legal_document, tax_document, resume, student_id_card]
OR generate a descriptive snake_case slug (e.g. 'math_notes', 'rent_agreement'). Never use 'miscellaneous' or 'other'.

Extract key metadata fields (dates, amounts, names, IDs) into a concise dictionary.
Write a short searchable summary (2-3 sentences max).
Confidence: 80-100.

Return ONLY a single top-level JSON object with keys: is_blur, blur_reason, category, confidence, metadata, summary, text."""

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
            data["text"] = extracted_text  # always store full text
            return json.dumps(data)
        except Exception as e:
            print(f"Text Analysis Gemini API Error: {e}, falling back to Qwen-72B...")
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
                data["text"] = extracted_text
                return json.dumps(data)
            except Exception as e2:
                print(f"Text Analysis Fallback Error: {e2}")
                import json
                fallback_json = {
                    "category": "miscellaneous",
                    "confidence": 70,
                    "metadata": {},
                    "summary": extracted_text[:300],
                    "text": extracted_text
                }
                return json.dumps(fallback_json)

    async def analyze_document(self, images_base64: list[str]):
        prompt = """Analyze this document image.

Step 1 — Quality: Is it too blurry/dark to read? If yes: is_blur=true, explain in blur_reason.
Step 2 — Classify into: [electricity_bill, water_bill, gas_bill, mobile_bill, bank_statement, invoice, receipt, marksheet, certificate, aadhaar, pan_card, passport, insurance, medical_report, prescription, legal_document, tax_document, resume, student_id_card]
OR generate a descriptive snake_case slug (e.g. 'math_notes'). Never use 'miscellaneous' or 'other'.
Extract key metadata fields. Write a short searchable summary. Extract visible text.

CRITICAL: Return ONLY a single top-level JSON object (never an array or list) with keys: is_blur, blur_reason, category, confidence, metadata, summary, text."""

        content_array = [{"type": "text", "text": prompt}]
        for b64_img in images_base64:
            content_array.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})

        try:
            response = await self.client.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[{"role": "user", "content": content_array}],
                response_format={"type": "json_object"},
                max_tokens=2500,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Vision Gemini API Error: {e}, trying fallback Qwen-VL...")
            try:
                response = await self.client.chat.completions.create(
                    model="qwen/qwen2.5-vl-72b-instruct",
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
                    "confidence": 0,
                    "metadata": {},
                    "summary": "Failed to process image during analysis.",
                    "text": ""
                }
                return json.dumps(fallback_json)
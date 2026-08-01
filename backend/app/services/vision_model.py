import base64
from openai import OpenAI
from app.core.config import settings

class VisionProcessor:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )

    async def analyze_text_document(self, extracted_text: str, document_name: str):
        prompt = f"""
        Analyze the following document text extracted from '{document_name}'.
        
        Step 1: Classification.
        Determine Document Category. If the document matches one of these predefined types, choose it: [electricity_bill, water_bill, gas_bill, mobile_bill, bank_statement, invoice, receipt, marksheet, certificate, aadhaar, pan_card, passport, insurance, medical_report, prescription, legal_document, tax_document, resume]. 
        Otherwise, generate a specific, descriptive category slug in snake_case (e.g., 'math_notes', 'programming_notes', 'academic_paper', 'rent_agreement', 'driver_license').
        DO NOT use generic categories like 'miscellaneous', 'other', or 'document'.
        Confidence Score: Evaluate confidence from 80 to 100.
        Metadata: Dynamically extract structured relevant fields (dates, amounts, reference numbers, names) into a dictionary.
        Summary: Generate a concise, searchable summary of the document.
        
        Return your response strictly as JSON with exactly these keys:
        'is_blur' (false), 'blur_reason' (null), 'category' (string), 'confidence' (integer), 'metadata' (dictionary), 'summary' (string), 'text' (string).
        """

        try:
            # Use fast text-only Qwen 7B or 72B text model
            response = self.client.chat.completions.create(
                model="qwen/qwen-2.5-7b-instruct",
                messages=[
                    {"role": "system", "content": "You are a document extraction assistant. Respond ONLY with valid JSON."},
                    {"role": "user", "content": f"{prompt}\n\nDOCUMENT TEXT:\n{extracted_text[:6000]}"}
                ],
                response_format={"type": "json_object"},
                max_tokens=1500,
                temperature=0.0,
            )
            result_str = response.choices[0].message.content
            # Ensure text field contains full extracted text
            import json
            data = json.loads(result_str)
            data["text"] = extracted_text
            return json.dumps(data)
        except Exception as e:
            print(f"Text Analysis API Error: {e}")
            fallback_json = {
                "category": "miscellaneous", 
                "confidence": 70, 
                "metadata": {}, 
                "summary": extracted_text[:300], 
                "text": extracted_text
            }
            import json
            return json.dumps(fallback_json)

    async def analyze_document(self, images_base64: list[str]):
        prompt = """
        Analyze this document image(s). 
        
        Step 1: Quality Check. Determine if the image is too blurry, dark, or low quality to reliably extract text or identify the document. 
        If it is illegible, set 'is_blur' to true and provide a short explanation in 'blur_reason'.
        
        Step 2: Classification (if not blurry). 
        Determine Document Category. If the document matches one of these predefined types, choose it: [electricity_bill, water_bill, gas_bill, mobile_bill, bank_statement, invoice, receipt, marksheet, certificate, aadhaar, pan_card, passport, insurance, medical_report, prescription, legal_document, tax_document, resume]. 
        Otherwise, you must generate a new specific, descriptive, and content-focused category slug in snake_case (e.g., 'math_notes', 'programming_notes', 'academic_paper', 'rent_agreement', 'driver_license') that best represents the document type.
        DO NOT use generic categories like 'miscellaneous', 'other', or 'document'. Always generate a descriptive label based on the specific content (e.g. for Chinese Remainder Theorem, use 'math_notes' or 'study_material').
        Confidence Score: Evaluate confidence from 0 to 100.
        Metadata: Dynamically extract structured relevant fields depending on the category.
        Summary: Generate a searchable summary of the document.
        Text: Extract the main text content.
        
        Return your response strictly as JSON with exactly these keys: 
        'is_blur' (boolean), 'blur_reason' (string or null), 'category' (string), 'confidence' (integer), 'metadata' (dictionary), 'summary' (string), 'text' (string).
        """
        
        content_array = [{"type": "text", "text": prompt}]
        for b64_img in images_base64:
            content_array.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})
        
        try:
            response = self.client.chat.completions.create(
                model="qwen/qwen2.5-vl-72b-instruct",
                messages=[{
                    "role": "user",
                    "content": content_array
                }],
                response_format={ "type": "json_object" },
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Vision API Error: {e}")
            fallback_json = {
                "category": "miscellaneous", 
                "confidence": 0, 
                "metadata": {}, 
                "summary": "Failed to extract text during processing.", 
                "text": ""
            }
            import json
            return json.dumps(fallback_json)
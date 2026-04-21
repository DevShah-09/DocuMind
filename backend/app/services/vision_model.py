import base64
from openai import OpenAI
from app.core.config import settings

class VisionProcessor:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )

    async def analyze_document(self, images_base64: list[str]):
        prompt = """
        Analyze this document image(s). 
        
        Step 1: Quality Check. Determine if the image is too blurry, dark, or low quality to reliably extract text or identify the document. 
        If it is illegible, set 'is_blur' to true and provide a short explanation in 'blur_reason'.
        
        Step 2: Classification (if not blurry). 
        Determine Document Category. Choose exactly one from: [electricity_bill, water_bill, gas_bill, mobile_bill, bank_statement, invoice, receipt, marksheet, certificate, aadhaar, pan_card, passport, insurance, medical_report, prescription, legal_document, tax_document, resume, miscellaneous].
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
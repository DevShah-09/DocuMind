import base64
from openai import OpenAI
from app.core.config import settings

class VisionProcessor:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )

    async def analyze_document(self, file_content: bytes):
        base64_image = base64.b64encode(file_content).decode('utf-8')
        
        # New prompt: "Describe this document in one or two words as a category"
        prompt = """
        Analyze this document image. 
        1. Give it a short, one-to-two word category name (e.g., 'Electricity Bill', 'College ID', 'Passport', 'Medical Report').
        2. Extract the main text content.
        Return as JSON: {"category": "...", "text": "..."}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="qwen/qwen2.5-vl-72b-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                response_format={ "type": "json_object" },
                max_tokens=2000, # Added to prevent OpenRouter from assuming 65k tokens and rejecting the request
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Vision API Error: {e}")
            # Fallback JSON so the categorization gracefully defaults to "Unsorted" instead of completely failing the upload
            return '{"category": "Unsorted", "text": "Failed to extract text during processing."}'
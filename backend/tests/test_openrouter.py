import os
import sys
import asyncio
from dotenv import load_dotenv

sys.path.insert(0, ".")
load_dotenv()

from app.services.vision_model import VisionProcessor

async def run_openrouter_test():
    try:
        vp = VisionProcessor()
        print("Testing OpenRouter API connection...")
        res = await vp.client.chat.completions.create(
            model="qwen/qwen2.5-vl-72b-instruct",
            messages=[{"role": "user", "content": "Hello, respond with exactly the word OK."}],
            max_tokens=10
        )
        print("API Response:", res.choices[0].message.content)
    except Exception as e:
        print("API Error:", e)

if __name__ == "__main__":
    asyncio.run(run_openrouter_test())

import os
from google import genai
from itertools import cycle
from dotenv import load_dotenv

# --- AI HANDLER ---
load_dotenv()

GEMINI_API_KEYS = [
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GEMINI_API_KEY2"),
        os.getenv("GEMINI_API_KEY3")
]

gemini_key_pool = cycle(GEMINI_API_KEYS)

def get_ai_client():
    current_key = next(gemini_key_pool)
    return genai.Client(api_key=current_key)





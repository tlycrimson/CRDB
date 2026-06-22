import os
import time
import asyncio
import httpx
import logging

from google import genai
from google.genai import errors, types
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# --- Constants ---
GEMINI_MODEL = "gemini-2.5-flash"
KEY_COOLDOWN_SECONDS = 15 * 3600  
MAX_TOKENS = 4096


FALLBACK_MODELS = [
    "openrouter/auto",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1:free",
]

GEMINI_SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
]

# --- Gemini Key Pool ---
_raw_keys = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY2"),
    os.getenv("GEMINI_API_KEY3"),
]

# { api_key: cooldown_until_timestamp }
_key_cooldowns: dict[str, float] = {}
_key_pool: list[str] = [k for k in _raw_keys if k]


def _get_available_key() -> str | None:
    """Return the first key not currently on cooldown, or None if all are rate-limited."""
    now = time.monotonic()
    for key in _key_pool:
        if now >= _key_cooldowns.get(key, 0):
            return key
    return None


def _mark_key_rate_limited(key: str):
    """Put a key on cooldown for KEY_COOLDOWN_SECONDS."""
    _key_cooldowns[key] = time.monotonic() + KEY_COOLDOWN_SECONDS
    logger.warning(
        f"Gemini key ...{key[-6:]} rate limited. "
        f"Cooling down for {KEY_COOLDOWN_SECONDS // 3600}h."
    )


def get_or_key() -> str | None:
    return os.getenv("OR_KEY")

class AIHandler:
    def __init__(self):
        self.switched = False

    # --- Main AI entry point ---
    async def generate_ai_response(self, system_instruction: str, query: str) -> str | None:
        """
        Try each available Gemini key in order. On 429/503, mark the key as
        rate-limited and try the next one. If all keys are exhausted, fall back
        to OpenRouter.
        """
        for key in _key_pool:
            now = time.monotonic()
            if now < _key_cooldowns.get(key, 0):
                logger.debug(f"Skipping key ...{key[-6:]} (on cooldown).")
                continue

            client = genai.Client(api_key=key)
            try:
                response = await client.aio.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=query,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,
                        safety_settings=GEMINI_SAFETY_SETTINGS,
                        max_output_tokens=MAX_TOKENS,
                    ),
                )
                if response.text:
                    if self.switched:
                        self.switched = False
                    return response.text

                return None

            except errors.APIError as e:
                if e.code in [429, 503]:
                    _mark_key_rate_limited(key)
                    logger.error(
                        f"Gemini key ...{key[-3:]} returned {e.code}. "
                        "Trying next key..."
                    )
                    continue
                else:
                    logger.error(f"Gemini API error (non-retryable): {e}")
                    raise

        # All Gemini keys exhausted — fall back to OpenRouter
        if not self.switched:
            logger.error("All Gemini keys rate limited or exhausted. Switching to OpenRouter fallback...")
            self.switched = True

        return await self._openrouter_fallback(system_instruction, query)


    async def _openrouter_fallback(self, system_instruction: str, user_question: str) -> str:
        """Try each OpenRouter fallback model in order."""
        openrouter_key = get_or_key()
        if not openrouter_key:
            logger.error("No OpenRouter API key configured.")
            return "```⚠️ Both primary and backup AI systems are currently unavailable.```"

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
        }

        for model in FALLBACK_MODELS:
            payload = {
                "model": model,
                "temperature": 0.3,
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_question},
                ],
            }
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
                    )

                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                elif response.status_code == 404:
                    logger.warning(f"OpenRouter model {model} not found, trying next...")
                    continue
                else:
                    logger.error(
                        f"OpenRouter error {response.status_code} for {model}: {response.text}"
                    )
                    continue

            except httpx.TimeoutException:
                logger.error(f"OpenRouter model {model} timed out, trying next...")
                continue
            except Exception as e:
                logger.error(f"OpenRouter fallback error for {model}: {e}")
                continue

        return "```⚠️ Both primary and backup AI systems are currently unavailable.```"

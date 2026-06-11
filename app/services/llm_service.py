from google import genai
from app.config import get_settings
from loguru import logger


class LLMService:
    def __init__(self):
        self.settings = get_settings()

        self.client = genai.Client(
            api_key=self.settings.google_api_key
        )

        self.model = self.settings.llm_model

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=f"{system_prompt}\n\n{prompt}"
            )

            return response.text

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise

    def generate_with_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.0
    ) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=f"{system_prompt}\n\n{prompt}"
            )

            return response.text

        except Exception as e:
            logger.error(f"LLM JSON generation error: {e}")
            raise
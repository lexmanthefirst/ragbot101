import os
from openai import AsyncOpenAI
from app.core.config import settings
from app.core.logging import logger


class LLMService:
    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Primary model
        self.primary_model = settings.OPENROUTER_MODEL
        
        # Fallback models (will try these if primary fails)
        self.fallback_models = [
            os.getenv("FALLBACK_MODEL_1", "meta-llama/llama-3.2-3b-instruct:free"),
            os.getenv("FALLBACK_MODEL_2", "google/gemma-2-9b-it:free"),
            os.getenv("FALLBACK_MODEL_3", "mistralai/mistral-7b-instruct:free"),
            os.getenv("FALLBACK_MODEL_4", "huggingfaceh4/zephyr-7b-beta:free"),
            os.getenv("FALLBACK_MODEL_5", "microsoft/phi-3.5-mini-instruct:free")
        ]
        
        self.site_url = os.getenv("SITE_URL", "http://localhost:8000")
        self.site_name = settings.PROJECT_NAME
        
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not set in environment variables")
        
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    async def get_embedding(self, text: str) -> list[float]:
        """
        Generate embeddings for the given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding
        """
        response = await self.client.embeddings.create(
            model="text-embedding-ada-002", 
            input=text
        )
        return response.data[0].embedding

    async def generate_answer(self, prompt: str) -> str:
        """
        Generate an answer using LLM with automatic fallback.
        
        Tries the primary model first, then falls back to alternative models
        if the primary fails.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If all models fail
        """
        models_to_try = [self.primary_model] + self.fallback_models
        last_error = None
        
        for idx, model in enumerate(models_to_try):
            try:
                if idx > 0:
                    logger.info(f"Trying fallback model {idx}: {model}")
                else:
                    logger.info(f"Using primary model: {model}")
                
                completion = await self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": self.site_url, 
                        "X-Title": self.site_name,
                    },
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
                
                answer = completion.choices[0].message.content
                
                if idx > 0:
                    logger.info(f"Successfully used fallback model {idx}: {model}")
                    
                return answer
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Model {model} failed: {str(e)}. "
                    f"{'Trying next fallback...' if idx < len(models_to_try) - 1 else 'No more fallbacks available.'}"
                )
                continue
        
        # If we get here, all models failed
        logger.error(f"All models failed. Last error: {str(last_error)}")
        raise Exception(f"All LLM models failed. Last error: {str(last_error)}")


llm_service = LLMService()

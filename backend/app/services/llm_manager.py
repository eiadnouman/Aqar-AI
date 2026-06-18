from langchain_groq import ChatGroq
from langchain_community.llms import HuggingFaceEndpoint
from app.core.config import settings
from app.core.logging import logger


class LLMManager:
    """
    Manages Large Language Model instances.
    Prioritizes Native Groq (openai/gpt-oss-120b) for flawless Arabic dialect logic.
    Safely falls back to a HuggingFace endpoint if Groq is unavailable.
    """
    
    def __init__(self):
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initializes the LLM, prioritizing Native Groq with fallback to Mixtral."""
        if getattr(settings, "groq_api_key", None):
            try:
                logger.info("Connecting to Native Groq (openai/gpt-oss-120b)...")
                llm = ChatGroq(
                    temperature=0.2,
                    model_name="openai/gpt-oss-120b",
                    groq_api_key=settings.groq_api_key.strip(),
                    max_tokens=settings.llm_max_tokens,
                    top_p=0.9,
                )
                logger.info("Connected to Groq successfully.")
                return llm
            except Exception as e:
                logger.error(f"Groq connection failed: {e}. Falling back to HuggingFace.")
        else:
            logger.warning("GROQ_API_KEY not set in environment. Trying HuggingFace fallback.")

        try:
            return self.get_hf_fallback()
        except Exception as e:
            logger.error(f"No LLM backend available: {e}")
            return None

    def get_hf_fallback(self):
        """Fallback to Mixtral via HuggingFace API."""
        hf_token = (settings.huggingfacehub_api_token or "").strip()
        if not hf_token:
            raise RuntimeError(
                "HUGGINGFACEHUB_API_TOKEN is required to use HuggingFace fallback when GROQ_API_KEY is unavailable."
            )

        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        logger.info(f"Initializing Fallback Model: {repo_id}")
        return HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.1,
            max_new_tokens=settings.llm_max_tokens,
            huggingfacehub_api_token=hf_token,
        )

    def get_llm(self):
        """Returns the active LLM instance."""
        if self.llm is None:
            raise RuntimeError("No LLM backend is configured.")
        return self.llm

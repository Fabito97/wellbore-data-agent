
"""
LLM Service - Unified interface for multiple LLM providers.

This service is STATELESS. It does not store conversation history.
Its only job is to execute a call to an LLM with a given context.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Iterator, List
from enum import Enum
from dataclasses import dataclass
import json

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from pydantic import SecretStr

from app.core.config import settings
from app.models.message import Message


logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """
    Structured response from LLM.
    """
    content: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    @property
    def has_token_info(self) -> bool:
        """Check if token usage info is available."""
        return self.total_tokens is not None


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    GEMINI = 'gemini'
    GROQ = "groq"
    # OPENAI = "openai"
    # ANTHROPIC = "anthropic"
    # AZURE = "azure"


class LLMService:
    """Stateless service for interacting with various LLM providers."""
    def __init__(
            self,
            provider: LLMProvider = LLMProvider.OLLAMA,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            api_key: Optional[SecretStr | str] = None,
            base_url: Optional[str] = None
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.model = model or self._get_default_model(provider)
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.api_key = api_key or ""
        self.base_url = base_url or self._get_default_base_url(provider)

        self.llm = self._initialize_provider()
        logger.info(
            f"LLMService initialized: provider={provider.value}, "
            f"model={self.model}, temperature={self.temperature}"
        )

    def _get_default_model(self, provider: LLMProvider) -> str:
        defaults = {
            LLMProvider.OLLAMA: settings.OLLAMA_MODEL,
            LLMProvider.HUGGINGFACE: "microsoft/Phi-3-mini-4k-instruct",
            LLMProvider.GEMINI: "gemini-1.5-flash",
            LLMProvider.GROQ: "llama-3.1-8b-instant",
        }
        return defaults.get(provider, "")

    def _get_default_base_url(self, provider: LLMProvider) -> str:
        defaults = {
            LLMProvider.OLLAMA: settings.OLLAMA_BASE_URL,
        }
        return defaults.get(provider)

    def _initialize_huggingface_provider(self):
        try:
            from langchain_huggingface import HuggingFacePipeline
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
        except ImportError:
            raise ImportError("HuggingFace support requires: pip install langchain-huggingface transformers torch")

        logger.info(f"Loading HuggingFace model: {self.model}")
        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=torch.float32,
            device_map=settings.EMBEDDING_DEVICE,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=settings.LLM_MAX_TOKENS,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            top_p=settings.LLM_TOP_P
        )
        return HuggingFacePipeline(pipeline=pipe)

    def _initialize_provider(self):
        if self.provider == LLMProvider.OLLAMA:
            return ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                num_ctx=settings.LLM_MAX_TOKENS,
            )
        elif self.provider == LLMProvider.HUGGINGFACE:
            return self._initialize_huggingface_provider()

        elif self.provider == LLMProvider.GEMINI:
            if not self.api_key:
                raise ValueError("Gemini requires an API key")

            from langchain_gemini import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=self.model,  # pick default
                api_key=self.api_key,
                temperature=self.temperature,
                max_output_tokens=settings.LLM_MAX_TOKENS,
            )

        elif self.provider == LLMProvider.GROQ:
            if not self.api_key:
                raise ValueError("Groq requires an API key")

            from langchain_groq import ChatGroq

            return ChatGroq(
                model=self.model,  # Groq defaults
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=settings.LLM_MAX_TOKENS,
            )

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _to_langchain_messages(self, messages: List[Message]) -> List[BaseMessage]:
        """Convert our Message model to LangChain's message types."""
        lc_messages = []
        for msg in messages:
            if msg.sender.lower() == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.sender.lower() == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.sender.lower() == "system":
                lc_messages.append(SystemMessage(content=msg.content))
        return lc_messages


    def generate(
            self,
            messages: Optional[List[Message]],
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM based on a list of messages.

        Args:
            messages: The history of messages in the conversation.
            system_prompt: System instructions (optional, prepended to messages).
            temperature: Override default temperature.

        Returns:
            LLMResponse with the generated text.
        """
        try:
            lc_messages = self._to_langchain_messages(messages)
            if system_prompt:
                lc_messages.insert(0, SystemMessage(content=system_prompt))

            llm = self.llm
            if temperature is not None:
                llm = self.llm.bind(temperature=temperature)

            logger.debug(f"Generating response for {len(lc_messages)} messages...")
            response = llm.invoke(lc_messages)
            content = response.content
            token_usage = getattr(response, 'response_metadata', {}).get('token_usage', {})

            result = LLMResponse(
                content=content,
                model=self.model,
                prompt_tokens=token_usage.get('prompt_tokens'),
                completion_tokens=token_usage.get('completion_tokens'),
                total_tokens=token_usage.get('total_tokens')
            )
            logger.debug(f"Response generated: {len(content)} chars")
            return result

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def stream_generate(
            self,
            messages: List[Message],
            system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        """
        Stream response token by token.

        Args:
            messages: The history of messages in the conversation.
            system_prompt: Optional system instructions.

        Yields:
            String tokens as they're generated.
        """
        lc_messages = self._to_langchain_messages(messages)
        if system_prompt:
            lc_messages.insert(0, SystemMessage(content=system_prompt))

        try:
            for chunk in self.llm.stream(lc_messages):
                if hasattr(chunk, 'content'):
                    yield chunk.content
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise


    def generate_structured(
            self,
            prompt: str,
            schema: Dict[str, Any],
            system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured output (JSON).

        Args:
            prompt: Extraction instructions
            schema: Expected output structure
            system_prompt: Optional system instructions

        Returns:
            Parsed JSON dictionary
        """
        # Build system prompt with schema
        schema_str = json.dumps(schema, indent=2)

        full_system = f"""Extract information and return it as valid JSON.

                    Output schema:
                    {schema_str}

                    Rules:
                    1. Return ONLY valid JSON, no other text
                    2. Use null for missing values
                    3. Follow the schema exactly
                    4. Extract values carefully from the provided text"""

        if system_prompt:
            full_system = system_prompt + "\n\n" + full_system

        messages: List[Message] = [
            Message(
            id=str(uuid.uuid4()),
            conversation_id=str(uuid.uuid4()),
            sender="user",
            content=prompt,
            timestamp=datetime.now()
            )
        ]
        # Generate response
        response = self.generate(messages=messages, system_prompt=full_system)

        # Parse JSON - LLMs sometimes add markdown fences: ```json ... ```
        #
        try:
            content = response.content.strip()

            # Remove markdown fences if present
            if content.startswith("```"):
                # Find JSON content between fences
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])  # Skip first and last line

            parsed = json.loads(content)
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response.content}")
            raise ValueError(f"LLM did not return valid JSON: {e}")



    def validate_connection(self) -> bool:
        """
        Test if the LLM is accessible and responsive.
        """
        try:
            logger.debug("Validating LLM connection...")
            # Construct a minimal list of messages for the LLM call
            test_messages = [
                SystemMessage(content="Respond with only the word 'Hi'"),
                HumanMessage(content="Hello")
            ]
            response = self.llm.invoke(test_messages)
            
            # Check if the response content is not empty and is a string
            is_valid = isinstance(response.content, str) and len(response.content) > 0
            if is_valid:
                logger.info("LLM connection validation successful.")
            else:
                logger.warning("LLM validation failed: received empty or invalid response.")
            return is_valid

        except Exception as e:
            logger.error(f"LLM validation failed with exception: {e}", exc_info=True)
            return False


# ==================== Module-level instance ====================

_llm_service_instance: Optional[LLMService] = None

def get_llm_service(
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
) -> LLMService:
    """
    Get a configured instance of the LLM service.
    For non-default providers, this will always create a new instance.
    The default provider instance is cached as a singleton.
    """
    global _llm_service_instance

    # Determine the provider from arguments or settings
    llm_provider = provider or settings.LLM_PROVIDER

    # If a non-default configuration is requested, create a new instance
    if llm_provider != LLMProvider.OLLAMA or (model or api_key):
        logger.info(f"Creating new LLMService instance for provider={llm_provider.value}")
        return LLMService(provider=llm_provider, model=model, api_key=api_key)

    # Otherwise, use the cached singleton for the default provider
    if _llm_service_instance is None:
        logger.info("Creating singleton LLMService instance for default provider.")
        _llm_service_instance = LLMService(provider=LLMProvider.OLLAMA)

    return _llm_service_instance

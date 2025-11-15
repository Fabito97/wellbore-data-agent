
"""
LLM Service - Unified interface for multiple LLM providers.

This service provides a clean, typed interface to interact with various LLMs:
- Ollama (local, default)
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Azure OpenAI
- Any LangChain-compatible provider
"""

import logging
from typing import Optional, Dict, Any, Iterator, List
from enum import Enum
from dataclasses import dataclass
import json

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import settings

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
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"


class LLMService:
    """Unified service for interacting with various LLM providers.  """
    def __init__(
            self,
            provider: LLMProvider = LLMProvider.OLLAMA,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None
    ):
        """
        Initialize LLM service with selected provider.

        Args:
            provider: Which LLM provider to use
            model: Model name (provider-specific)
            temperature: Sampling temperature
            api_key: API key for cloud providers
            base_url: Custom endpoint URL
        """
        self.provider = provider or settings.LLM_PROVIDER
        self.model = model or self._get_default_model(provider)
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.api_key = api_key if api_key is not None else settings.HF_TOKEN
        self.base_url = base_url or self._get_default_base_url(provider)

        # Initialize provider-specific client
        self.llm = self._initialize_provider()

        logger.info(
            f"LLMService initialized: provider={provider.value}, "
            f"model={self.model}, temperature={self.temperature}"
        )

    def _get_default_model(self, provider: LLMProvider) -> str:
        """Get default model for provider."""
        defaults = {
            LLMProvider.OLLAMA: settings.OLLAMA_MODEL,
            LLMProvider.HUGGINGFACE: "microsoft/Phi-3-mini-4k-instruct"
            # LLMProvider.HUGGINGFACE: "microsoft/phi-2"
        }
        return defaults.get(provider, "")

    def _get_default_base_url(self, provider: LLMProvider) -> str:
        """Get default base URL for provider."""
        defaults = {
            LLMProvider.OLLAMA: settings.OLLAMA_BASE_URL,
            LLMProvider.HUGGINGFACE: None
        }
        return defaults.get(provider)

    def _initialize_huggingface_provider(self):
        try:
            from langchain_huggingface import HuggingFacePipeline
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
        except ImportError:
            raise ImportError(
                "HuggingFace support requires: "
                "pip install langchain-huggingface transformers torch"
            )

        logger.info(f"Loading HuggingFace model: {self.model}")
        logger.info("⚠️  First run will download model (~2GB). This may take a few minutes.")

        # Load tokenizer and model - Downloads from HuggingFace Hub and cached in ~/.cache/huggingface/
        tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=True  # Some models need this
        )

        logger.info("Loading huggingface model. This will will take a few minutes.")
        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=torch.float32,  # CPU requires float32
            device_map=settings.EMBEDDING_DEVICE,  # Force CPU (hackathon constraint)
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Memory optimization
        )

        # Ensure pad token is set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Create pipeline - Handles tokenization, generation, decoding

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=settings.LLM_MAX_TOKENS,  # max_new_tokens: How much to generate
            temperature=self.temperature,
            do_sample=True if self.temperature > 0 else False,  # do_sample: Enable sampling (vs greedy)
            top_p=settings.LLM_TOP_P
        )

        # Wrap in LangChain interface
        return HuggingFacePipeline(pipeline=pipe)

    def _initialize_provider(self):
        """Initialize the appropriate LangChain chat model."""
        if self.provider == LLMProvider.OLLAMA or settings.OLLAMA_MODEL == str(LLMProvider.OLLAMA):
            return ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                num_ctx=settings.LLM_MAX_TOKENS,
            )

        elif self.provider == LLMProvider.HUGGINGFACE or settings.LLM_PROVIDER == str(LLMProvider.HUGGINGFACE):
            # Direct HuggingFace transformers (fully local)
            return self._initialize_huggingface_provider()

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: User's input/question
            system_prompt: System instructions (optional)
            temperature: Override default temperature
            max_tokens: Max tokens to generate

        Returns:
            LLMResponse with generated text
        """
        try:
            # Build messages
            messages = []

            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            messages.append(HumanMessage(content=prompt))

            # Override temperature if provided
            llm = self.llm
            if temperature is not None:
                llm = self.llm.bind(temperature=temperature)

            # Generate response
            logger.debug(f"Generating response for prompt: {prompt[:100]}...")

            response = llm.invoke(messages)

            # Extract content
            content = response.content

            # Token usage (if available)
            # Note: Ollama doesn't always provide token counts
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


    def generate_with_context(
            self,
            question: str,
            context: str,
            system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate response with provided context (RAG pattern).

        Args:
            question: User's question
            context: Retrieved context from documents
            system_prompt: Optional system instructions

        Returns:
            LLMResponse with answer
        """
        # Default system prompt for RAG
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant specialized in petroleum engineering and well analysis.
                            
                            Your task is to answer questions based ONLY on the provided context from well documents.
                            
                            Rules:
                            1. Answer only using information from the context
                            2. If the answer is not in the context, say "I cannot find this information in the provided documents"
                            3. Cite the source (document name and page number) when possible
                            4. Be concise and factual
                            5. If you're unsure, say so
                            
                            Context format: [Document name, Page X]
                            This shows where each piece of information comes from."""

        # Format prompt with context - Clear separation
        prompt = f"""Context from documents:
                
                {context}
                
                ---
                
                Question: {question}
                
                Answer based on the context above:"""

        return self.generate(prompt=prompt, system_prompt=system_prompt)


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

        # Generate response
        response = self.generate(prompt=prompt, system_prompt=full_system)

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


    def chat(
            self,
            messages: List[Dict[str, str]],
            system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Multi-turn conversation.

        Args:
            messages: List of conversation turns
            system_prompt: Optional system instructions

        Returns:
            LLMResponse with next turn
        """
        # Convert to LangChain messages
        lc_messages = []

        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))

        # Generate response
        response = self.llm.invoke(lc_messages)

        return LLMResponse(
            content=response.content,
            model=self.model
        )


    def stream_generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        """
        Stream response token by token - Tokens appear as generated (immediate feedback)

        Args:
            prompt: User's input
            system_prompt: Optional system instructions

        Yields:
            String tokens as they're generated
        """
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        try:
            for chunk in self.llm.stream(messages):
                # Each chunk is a message delta
                if hasattr(chunk, 'content'):
                    yield chunk.content

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise

    def validate_connection(self) -> bool:
        """
        Test if Ollama is accessible and model is available.

        Returns:
            True if LLM is ready, False otherwise
        """
        try:
            # Simple test prompt
            response = self.generate(
                prompt="Hello",
                system_prompt="Respond with 'Hi'"
            )

            return len(response.content) > 0

        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return False


# ==================== Module-level instance ====================

_llm_service_instance: Optional[LLMService] = None


def get_llm_service(
        provider: Optional[LLMProvider] = LLMProvider.OLLAMA,
        model: Optional[str] = None,
        api_key: Optional[str] = None,

) -> LLMService:
    """
    Get or create LLM service instance - Use Ollama (default)

    Args:
        provider: Which LLM provider
        model: Optional model override

    Returns:
        Configured LLM service
    """
    global _llm_service_instance
    llm_provider =  provider or LLMProvider(settings.LLM_PROVIDER)
    print(f"Using {str(llm_provider)} to process user requests")

    if provider != (LLMProvider.OLLAMA or model is not None) and api_key is not None:
        return LLMService(provider=provider, model=model, api_key=api_key)


    # For non-default providers, always create new instance
    if provider != LLMProvider.OLLAMA or model is not None:
        return LLMService(provider=llm_provider, model=model)

    # For default (Ollama), use singleton
    if _llm_service_instance is None:
        _llm_service_instance = LLMService(provider=llm_provider)

    return _llm_service_instance

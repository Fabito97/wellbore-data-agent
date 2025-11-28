from app.core.config import settings


def validate_required_settings() -> list[str]:
    """
    Validate that all required external dependencies are configured.

    Returns:
        List of validation errors (empty if all valid)
    """
    errors = []

    if not settings.LLM_PROVIDER:
        errors.append("LLM_PROVIDER is required")

    # Check Ollama config
    if settings.LLM_PROVIDER == "ollama":
        if not settings.OLLAMA_BASE_URL:
            errors.append("OLLAMA_BASE_URL is required")
        if not settings.OLLAMA_MODEL:
            errors.append("OLLAMA_MODEL is required")

    elif settings.LLM_PROVIDER == "huggingface":
        if not settings.HF_TOKEN:
            errors.append("HF_TOKEN is required")

    elif settings.LLM_PROVIDER == "gemini":
        if not settings.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is required")

    elif settings.LLM_PROVIDER == "groq":
        if not settings.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is required")

        # Embedding config (common requirement)
    if not settings.EMBEDDING_MODEL:
        errors.append("EMBEDDING_MODEL is required")

    return errors


def validate_ollama_connection() -> bool:
    """
    Check if Ollama is running and accessible.

    Returns:
        bool: True if Ollama is accessible, False otherwise
    """
    import httpx

    try:
        response = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def validate_ollama_model() -> bool:
    """
    Check if the configured model is available in Ollama.

    Returns:
        bool: True if model exists, False otherwise
    """
    import httpx

    try:
        response = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            return settings.OLLAMA_MODEL in model_names
        return False
    except Exception:
        return False


if __name__ == "__main__":
    """Test configuration loading."""
    print("=" * 60)
    print("WELLBORE AI AGENT - Configuration Test")
    print("=" * 60)

    # Check for validation errors
    errors = settings.validate_required_settings()
    if errors:
        print("\n❌ Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
        print("\n💡 Please check your .env file")
        exit(1)

    # Display key settings
    print(f"\n📱 Application:")
    print(f"   Name: {settings.APP_NAME}")
    print(f"   Version: {settings.APP_VERSION}")
    print(f"   Debug: {settings.DEBUG}")

    print(f"\n🤖 LLM Settings:")
    print(f"   Ollama URL: {settings.OLLAMA_BASE_URL}")
    print(f"   HF Token: {settings.HF_TOKEN[:5] + ("*" * 10)}")
    print(f"   Model: {settings.OLLAMA_MODEL}")
    print(f"   Temperature: {settings.LLM_TEMPERATURE}")

    print(f"\n🔤 Embedding Settings:")
    print(f"   Model: {settings.EMBEDDING_MODEL}")
    print(f"   Dimension: {settings.EMBEDDING_DIMENSION}")

    print(f"\n📁 Storage Paths:")
    print(f"   Upload Dir: {settings.UPLOAD_DIR}")
    print(f"   Vector DB: {settings.VECTOR_DB_DIR}")

    print(f"\n🔍 RAG Settings:")
    print(f"   Chunk Size: {settings.CHUNK_SIZE}")
    print(f"   Top K: {settings.RETRIEVAL_TOP_K}")

    print(f"\n🔌 Validating Ollama Connection...")
    if validate_ollama_connection():
        print("   ✅ Ollama is running")

        if validate_ollama_model():
            print(f"   ✅ Model '{settings.OLLAMA_MODEL}' is available")
        else:
            print(f"   ⚠️  Model '{settings.OLLAMA_MODEL}' not found")
            print(f"   💡 Run: ollama pull {settings.OLLAMA_MODEL}")
    else:
        print("   ❌ Ollama is not accessible")
        print("   💡 Run: ollama serve")

    print("\n" + "=" * 60)

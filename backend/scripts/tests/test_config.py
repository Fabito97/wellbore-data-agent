"""
Test script to validate configuration and environment setup.

Run this after installing requirements to ensure everything is configured correctly.

Usage:
    python scripts/test_config.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.validation.validators import  validate_ollama_connection, validate_ollama_model


def test_configuration():
    """Test basic configuration loading."""
    print("🧪 Testing Configuration Loading...")

    tests = {
        "APP_NAME is set": settings.APP_NAME is not None,
        "OLLAMA_MODEL is set": settings.OLLAMA_MODEL is not None,
        "EMBEDDING_MODEL is set": settings.EMBEDDING_MODEL is not None,
        "Data directories exist": settings.DATA_DIR.exists(),
        "Upload directory exists": settings.UPLOAD_DIR.exists(),
        "Vector DB directory exists": settings.VECTOR_DB_DIR.exists(),
    }

    passed = sum(tests.values())
    total = len(tests)

    for test_name, result in tests.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")

    print(f"\n  Result: {passed}/{total} tests passed\n")
    return passed == total


def test_ollama():
    """Test Ollama connection and model availability."""
    print("🤖 Testing Ollama Setup...")

    # Test connection
    print(f"  Checking connection to {settings.OLLAMA_BASE_URL}...")
    if validate_ollama_connection():
        print("  ✅ Ollama is running and accessible")

        # Test model
        print(f"  Checking if model '{settings.OLLAMA_MODEL}' is available...")
        if validate_ollama_model():
            print(f"  ✅ Model '{settings.OLLAMA_MODEL}' is available")
            return True
        else:
            print(f"  ❌ Model '{settings.OLLAMA_MODEL}' not found")
            print(f"\n  💡 To fix this, run:")
            print(f"     ollama pull {settings.OLLAMA_MODEL}")
            return False
    else:
        print("  ❌ Ollama is not accessible")
        print("\n  💡 To fix this:")
        print("     1. Install Ollama from https://ollama.ai")
        print("     2. Start the service: ollama serve")
        return False


def test_imports():
    """Test that all required packages can be imported."""
    print("📦 Testing Package Imports...")

    packages = {
        "fastapi": "FastAPI",
        "pydantic": "Pydantic",
        "langchain": "LangChain",
        "langgraph": "LangGraph",
        "chromadb": "ChromaDB",
        "sentence_transformers": "Sentence Transformers",
        "pdfplumber": "pdfplumber",
        "httpx": "HTTPX",
    }

    failed = []

    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - Not installed")
            failed.append(package)

    if failed:
        print(f"\n  ⚠️  Missing packages: {', '.join(failed)}")
        print("  💡 Run: pip install -r requirements.txt")
        return False

    print(f"\n  Result: All packages installed correctly\n")
    return True


def test_embedding_model():
    """Test that embedding model can be loaded."""
    print("🔤 Testing Embedding Model...")

    try:
        from sentence_transformers import SentenceTransformer

        print(f"  Loading {settings.EMBEDDING_MODEL}...")
        model = SentenceTransformer(settings.EMBEDDING_MODEL)

        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)

        if len(embedding) == settings.EMBEDDING_DIMENSION:
            print(f"  ✅ Model loaded successfully (dimension: {len(embedding)})")
            return True
        else:
            print(f"  ❌ Unexpected embedding dimension: {len(embedding)} (expected {settings.EMBEDDING_DIMENSION})")
            return False

    except Exception as e:
        print(f"  ❌ Failed to load embedding model: {e}")
        print("\n  💡 The model will download on first use. Ensure you have internet connection.")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("  WELLBORE AI AGENT - Environment Setup Test")
    print("=" * 70)
    print()

    results = {
        "Configuration": test_configuration(),
        "Package Imports": test_imports(),
        "Ollama Setup": test_ollama(),
        "Embedding Model": test_embedding_model(),
    }

    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name}")

    all_passed = all(results.values())

    print()
    if all_passed:
        print("🎉 All tests passed! Your environment is ready.")
        print("\n📝 Next steps:")
        print("   1. Copy .env.example to .env and customize if needed")
        print("   2. Run: uvicorn app.main:app --reload")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
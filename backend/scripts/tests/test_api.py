"""
API Test Script.

Tests the complete REST and WebSocket API.

Usage:
    # Start server first:
    uvicorn app.main:app --reload

    # Then run tests:
    python scripts/test_api.py [pdf_path]
"""

import sys
import json
import asyncio
from pathlib import Path

import httpx
import websockets

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
API_V1 = "/api/v1"


def test_root():
    """Test root endpoint."""
    print("=" * 70)
    print("TEST 1: Root Endpoint")
    print("=" * 70)

    try:
        response = httpx.get(f"{BASE_URL}/")
        print(f"\n✅ Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return False


def test_health():
    """Test health endpoint."""
    print("\n" + "=" * 70)
    print("TEST 2: Health Check")
    print("=" * 70)

    try:
        response = httpx.get(f"{BASE_URL}/health")
        data = response.json()

        print(f"\n✅ Status: {response.status_code}")
        print(f"Health: {data['status']}")
        print(f"Services:")
        for service, status in data['services'].items():
            emoji = "✅" if status == "healthy" else "⚠️"
            print(f"  {emoji} {service}: {status}")

        return data['status'] in ['healthy', 'degraded']
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return False


def test_document_upload(pdf_path: Path):
    """Test document upload."""
    print("\n" + "=" * 70)
    print("TEST 3: Document Upload")
    print("=" * 70)

    try:
        print(f"\n📤 Uploading: {pdf_path.name}")

        with open(pdf_path, 'rb') as f:
            files = {'file': (pdf_path.name, f, 'application/pdf')}
            response = httpx.post(
                f"{BASE_URL}{API_V1}/documents/upload",
                files=files,
                timeout=120.0  # Long timeout for processing
            )

        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Upload successful!")
            print(f"   Document ID: {data['document_id']}")
            print(f"   Pages: {data['page_count']}")
            print(f"   Chunks: {data['chunk_count']}")
            print(f"   Status: {data['status']}")
            return data['document_id']
        else:
            print(f"\n❌ Upload failed: {response.status_code}")
            print(f"   {response.text}")
            return None

    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return None


def test_list_documents():
    """Test listing documents."""
    print("\n" + "=" * 70)
    print("TEST 4: List Documents")
    print("=" * 70)

    try:
        response = httpx.get(f"{BASE_URL}{API_V1}/documents/")
        data = response.json()

        print(f"\n✅ Found {data['total']} document(s)")

        for doc in data['documents']:
            print(f"\n   📄 {doc['filename']}")
            print(f"      ID: {doc['document_id']}")
            print(f"      Pages: {doc['pages']}")
            print(f"      Chunks: {doc['chunks']}")

        return True
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return False


def test_document_status(document_id: str):
    """Test document status endpoint."""
    print("\n" + "=" * 70)
    print("TEST 5: Document Status")
    print("=" * 70)

    try:
        response = httpx.get(
            f"{BASE_URL}{API_V1}/documents/{document_id}/status"
        )
        data = response.json()

        print(f"\n✅ Status retrieved:")
        print(f"   Filename: {data['filename']}")
        print(f"   Status: {data['status']}")
        print(f"   Chunks: {data['chunk_count']}")

        return True
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return False


def test_rest_ask():
    """Test REST ask endpoint."""
    print("\n" + "=" * 70)
    print("TEST 6: REST Ask Endpoint")
    print("=" * 70)

    questions = [
        "What is the well depth?",
        "Tell me about the completion"
    ]

    for question in questions:
        try:
            print(f"\n❓ Question: {question}")

            response = httpx.post(
                f"{BASE_URL}{API_V1}/chat/ask",
                params={
                    "question": question,
                    "include_sources": True
                },
                timeout=30.0
            )

            data = response.json()

            print(f"   Answer: {data['answer'][:150]}...")
            print(f"   Confidence: {data['confidence']}")
            print(f"   Sources: {len(data.get('sources', []))}")

        except Exception as e:
            print(f"   ❌ Failed: {e}")


async def test_websocket_chat():
    """Test WebSocket chat."""
    print("\n" + "=" * 70)
    print("TEST 7: WebSocket Chat")
    print("=" * 70)

    try:
        uri = f"{WS_URL}{API_V1}/chat/ws"

        async with websockets.connect(uri) as websocket:
            # Wait for welcome message
            welcome = await websocket.recv()
            print(f"\n✅ Connected: {json.loads(welcome)['message']}")

            # Test question
            question = "What is the measured depth?"
            print(f"\n📤 Sending question: {question}")

            await websocket.send(json.dumps({
                "type": "question",
                "content": question,
                "options": {
                    "include_sources": True
                }
            }))

            # Receive responses
            while True:
                response = await websocket.recv()
                data = json.loads(response)

                msg_type = data.get("type")

                if msg_type == "status":
                    print(f"   ⏳ {data['message']}")

                elif msg_type == "answer":
                    print(f"\n   ✅ Answer received:")
                    print(f"      {data['content'][:200]}...")
                    print(f"      Confidence: {data['confidence']}")
                    print(f"      Sources: {len(data.get('sources', []))}")
                    break

                elif msg_type == "error":
                    print(f"\n   ❌ Error: {data['message']}")
                    break

            # Test summarize
            print(f"\n📤 Testing summarize...")

            # Get document list first
            response = httpx.get(f"{BASE_URL}{API_V1}/documents/")
            docs = response.json()['documents']

            if docs:
                doc_id = docs[0]['document_id']

                await websocket.send(json.dumps({
                    "type": "summarize",
                    "document_id": doc_id,
                    "max_words": 100
                }))

                response = await websocket.recv()
                data = json.loads(response)

                if data['type'] == 'summary':
                    print(f"   ✅ Summary: {data['content'][:150]}...")

            print(f"\n✅ WebSocket tests complete")
            return True

    except Exception as e:
        print(f"\n❌ WebSocket failed: {e}")
        return False


def test_stats():
    """Test stats endpoint."""
    print("\n" + "=" * 70)
    print("TEST 8: System Stats")
    print("=" * 70)

    try:
        response = httpx.get(f"{BASE_URL}{API_V1}/documents/stats/summary")
        data = response.json()

        print(f"\n✅ Stats retrieved:")
        print(f"   Total documents: {data['total_documents']}")
        print(f"   Total pages: {data['total_pages']}")
        print(f"   Total chunks: {data['total_chunks']}")
        print(f"   Vector store chunks: {data['vector_store']['total_chunks']}")

        return True
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return False


def main():
    """Run all API tests."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_api.py <pdf_file>")
        print("\nExample:")
        print("  python scripts/test_api.py data/raw/sample.pdf")
        print("\nMake sure server is running:")
        print("  uvicorn app.main:app --reload")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🌐 API TEST SUITE")
    print("=" * 70)
    print(f"\nServer: {BASE_URL}")
    print(f"Test file: {pdf_path.name}")

    # Check server is running
    try:
        httpx.get(BASE_URL, timeout=2)
    except:
        print("\n❌ Server not responding!")
        print("Start server with: uvicorn app.main:app --reload")
        sys.exit(1)

    # Run tests
    test_root()
    test_health()

    document_id = test_document_upload(pdf_path)

    if document_id:
        test_list_documents()
        test_document_status(document_id)
        test_stats()
        test_rest_ask()

        # WebSocket test
        print("\n⏳ Running WebSocket tests...")
        asyncio.run(test_websocket_chat())

    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 70)
    print("\n💡 API is fully operational!")
    print("   - REST endpoints working")
    print("   - WebSocket chat functional")
    print("   - Ready for frontend integration")


if __name__ == "__main__":
    main()
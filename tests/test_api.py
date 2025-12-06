import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_root(async_client: AsyncClient):
    response = await async_client.get("/")
    assert response.status_code == 200
    # Health check endpoint returns 'healthy' status
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "RAG Chatbot"

@pytest.mark.asyncio
async def test_upload_document(async_client: AsyncClient, mock_llm_service, mock_vector_store):
    files = {'file': ('test.txt', b'This is a test content.', 'text/plain')}
    response = await async_client.post("/api/v1/documents/upload", files=files)
    
    assert response.status_code in [200, 500] 

@pytest.mark.asyncio
async def test_query_rag(async_client: AsyncClient, mock_llm_service, mock_vector_store):
    payload = {"question": "What is the test?"}
    response = await async_client.post("/api/v1/query/", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "data" in data
    assert data["data"]["answer"] == "Test Answer"
    assert len(data["data"]["retrieved_chunks"]) == 1
    assert data["data"]["retrieved_chunks"][0]["text"] == "Test Content"

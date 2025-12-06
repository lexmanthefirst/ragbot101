import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app
from unittest.mock import AsyncMock, MagicMock, patch
import os

os.environ["USE_LOCAL_EMBEDDINGS"] = "false"

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Setup test environment variables"""
    monkeypatch.setenv("USE_LOCAL_EMBEDDINGS", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

@pytest.fixture
def mock_vector_store(monkeypatch):
    """Mock vector store to avoid ChromaDB during tests"""
    mock = MagicMock()
    mock.query.return_value = {
        'ids': [['1']],
        'documents': [['Test Content']],
        'metadatas': [[{'source': 'test.txt'}]]
    }
    mock.add_documents = MagicMock()
    
    # Patch at the module level before imported
    import app.services.vector_store
    monkeypatch.setattr(app.services.vector_store, "get_vector_store", lambda: mock)
    
    # Patch QueryService
    from app.services.query_service import QueryService
    original_init = QueryService.__init__
    
    def mock_init(self):
        from app.services.llm_service import llm_service
        self.llm = llm_service
        self.vector_store = mock
    
    monkeypatch.setattr(QueryService, "__init__", mock_init)
    return mock

@pytest.fixture
def mock_llm_service(monkeypatch):
    """Mock LLM service to avoid API calls during tests"""
    
    # Mock llm_service singleton
    from app.services import llm_service
    
    # Create async mock methods
    async def mock_get_embedding(text):
        return [0.1] * 384  # Local model dimension
    
    async def mock_generate_answer(prompt):
        return "Test Answer"
    
    monkeypatch.setattr(llm_service.llm_service, "get_embedding", mock_get_embedding)
    monkeypatch.setattr(llm_service.llm_service, "generate_answer", mock_generate_answer)

@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

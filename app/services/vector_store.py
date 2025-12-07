import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from app.core.config import settings

class VectorStore:
    """Base class for vector stores with standardized similarity scores."""
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str], embeddings: List[List[float]]):
        raise NotImplementedError
    
    def query(self, query_embeddings: List[List[float]], n_results: int = 5) -> Dict[str, Any]:
        """
        Query the vector store.
        
        Returns:
            Dict with 'documents', 'metadatas', 'ids', and 'similarities' (0-1 range)
        """
        raise NotImplementedError

class ChromaVectorStore(VectorStore):
    """ChromaDB implementation with automatic similarity score conversion."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
        # Use cosine similarity (recommended for text embeddings)
        # Range: -1 to 1, where 1 = identical, 0 = orthogonal, -1 = opposite
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # Cosine similarity metric
        )
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str], embeddings: List[List[float]]):
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_embeddings: List[List[float]], n_results: int = 5) -> Dict[str, Any]:
        """
        Query ChromaDB and convert distances to similarities.
        
        ChromaDB with cosine metric returns: distance = 1 - cosine_similarity
        We convert to: similarity = 1 - distance
        """
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )
        
        # Convert ChromaDB distances to standardized similarities
        if 'distances' in results and results['distances']:
            results['similarities'] = [
                [1.0 - d for d in dist_list]  # Cosine: similarity = 1 - distance
                for dist_list in results['distances']
            ]
        else:
            # Fallback if no distances returned
            results['similarities'] = [[0.0] * len(results.get('documents', [[]])[0])]
        
        return results

def get_vector_store() -> VectorStore:
    """Factory function to get the configured vector store."""
    if settings.VECTOR_DB_TYPE == "chroma":
        return ChromaVectorStore()
    # Placeholder for other vector stores
    raise NotImplementedError(f"Vector store '{settings.VECTOR_DB_TYPE}' not supported")

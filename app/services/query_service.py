from typing import List, Tuple

from app.schemas.query import QueryData, RetrievalChunk
from app.services.llm_service import llm_service
from app.services.vector_store import get_vector_store
from app.core.logging import logger


class QueryService:
    """Service for processing RAG queries."""
    
    def __init__(self):
        self.llm = llm_service
        self.vector_store = get_vector_store()
    
    async def process_query(self, question: str) -> QueryData:
        """
        Process a RAG query through the complete pipeline.
        
        Pipeline steps:
        1. Generate embedding for the question
        2. Retrieve similar document chunks from vector store
        3. Build context from retrieved chunks
        4. Generate answer using LLM
        
        Args:
            question: User's question
            
        Returns:
            QueryData: Answer and retrieved chunks
            
        Raises:
            Exception: If any step in the pipeline fails
        """
        logger.info("Processing RAG query: question='%s'", question[:100])
        
        # Step 1: Generate embedding
        query_embedding = await self._generate_embedding(question)
        
        # Step 2: Retrieve similar chunks
        retrieved_chunks, context_parts = await self._retrieve_chunks(query_embedding)
        
        # Step 3: Check if we have context
        if not context_parts:
            logger.warning("No relevant documents found for query")
            return QueryData(
                answer="I don't have enough information to answer this question. Please upload relevant documents first.",
                retrieved_chunks=[]
            )
        
        # Step 4: Build prompt and generate answer
        context = "\n\n".join(context_parts)
        prompt = self._build_rag_prompt(context, question)
        answer = await self._generate_answer(prompt)
        
        logger.info(
            "RAG query completed: chunks_used=%d, answer_length=%d",
            len(retrieved_chunks),
            len(answer)
        )
        
        return QueryData(
            answer=answer,
            retrieved_chunks=retrieved_chunks
        )
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
            
        Raises:
            Exception: If embedding generation fails
        """
        logger.info("Generating embedding for question")
        embedding = await self.llm.get_embedding(text)
        logger.info("Embedding generated successfully")
        return embedding
    
    async def _retrieve_chunks(
        self,
        query_embedding: List[float],
        n_results: int = 3
    ) -> Tuple[List[RetrievalChunk], List[str]]:
        """
        Retrieve similar document chunks from vector store.
        
        Args:
            query_embedding: Embedding vector for the query
            n_results: Number of results to retrieve
            
        Returns:
            Tuple of (retrieved_chunks, context_parts)
            
        Raises:
            Exception: If vector search fails
        """
        logger.info("Querying vector store for top %d relevant chunks", n_results)
        
        # Request distances from ChromaDB for similarity scores
        results = self.vector_store.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        logger.info("Vector search completed successfully")
        
        retrieved_chunks: List[RetrievalChunk] = []
        context_parts: List[str] = []
        
        if results and results.get('documents'):
            documents = results['documents'][0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]  # Get distance metrics
            
            num_chunks = len(documents)
            logger.info("Retrieved %d relevant chunk(s)", num_chunks)
            
            for i, doc_text in enumerate(documents):
                meta = metadatas[i] if i < len(metadatas) else {}
                source = meta.get('source', 'unknown')
                
                # Convert distance to similarity score
                # ChromaDB uses L2 (Euclidean) distance: lower = more similar
                # Convert to 0-1 scale where 1 = perfect match, 0 = no match
                distance = distances[i] if i < len(distances) else float('inf')
                similarity = 1.0 / (1.0 + distance) if distance != float('inf') else 0.0
                
                retrieved_chunks.append(RetrievalChunk(
                    text=doc_text,
                    source=source,
                    similarity_score=round(similarity, 4)  # Round to 4 decimal places
                ))
                
                context_parts.append(f"Source: {source}\nContent: {doc_text}")
        
        return retrieved_chunks, context_parts
    
    def _build_rag_prompt(self, context: str, question: str) -> str:
        """
        Build a RAG prompt with context and question.
        
        Args:
            context: Retrieved context from documents
            question: User's question
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Answer the user question based on the following context. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
        return prompt
    
    async def _generate_answer(self, prompt: str) -> str:
        """
        Generate answer using LLM.
        
        Args:
            prompt: The RAG prompt
            
        Returns:
            Generated answer
            
        Raises:
            Exception: If answer generation fails
        """
        logger.info("Generating answer with LLM")
        answer = await self.llm.generate_answer(prompt)
        logger.info("Answer generated successfully (length: %d chars)", len(answer))
        return answer

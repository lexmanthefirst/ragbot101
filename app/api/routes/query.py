from fastapi import APIRouter, HTTPException, status

from app.schemas.query import QueryRequest, QueryResponse
from app.services.query_service import QueryService
from app.core.logging import logger

router = APIRouter()


@router.post(
    "/",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query RAG System",
    description="Ask a question and get an answer based on uploaded documents using retrieval-augmented generation (RAG)."
)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    This endpoint implements a complete RAG pipeline:
    1. Embeds the user's question using OpenRouter
    2. Retrieves semantically similar document chunks from ChromaDB
    3. Constructs a prompt with retrieved context
    4. Generates an answer using the LLM with fallback models
    
    Args:
        request: QueryRequest containing the user's question
    
    Returns:
        QueryResponse: Generated answer with retrieved context chunks
    
    Raises:
        HTTPException: 500 if any step in the pipeline fails
    """
    logger.info("RAG query endpoint called: question='%s'", request.question[:100])
    
    service = QueryService()
    
    try:
        # Process query through the service layer
        query_data = await service.process_query(request.question)
        
        return QueryResponse(
            status="success",
            message="Query processed successfully",
            data=query_data
        )
        
    except Exception as e:
        logger.error(
            "RAG query processing failed: %s",
            str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Failed to process query",
                "details": {"error": str(e)}
            }
        )

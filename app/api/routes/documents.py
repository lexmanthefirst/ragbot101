import uuid
from typing import List

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.services.ingestion import IngestionService
from app.schemas.document import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDetailResponse,
    DocumentData,
)
from app.models.document import Document
from app.core.logging import logger

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Document",
    description="Upload and process a document (PDF, DOCX, or TXT). The document will be chunked, embedded, and stored in the vector database."
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and process a document for RAG system.
    
    This endpoint accepts PDF, DOCX, or TXT files, extracts text content,
    splits it into chunks, generates embeddings, and stores them in the
    vector database for later retrieval.
    
    Args:
        file: The uploaded file (PDF, DOCX, or TXT)
        db: Database session dependency
    
    Returns:
        DocumentUploadResponse: Document metadata with upload status
    
    Raises:
        HTTPException: 500 if document processing fails
    """
    logger.info(
        "Document upload requested: filename='%s', content_type='%s', size=%d bytes",
        file.filename,
        file.content_type,
        file.size if file.size else 0
    )
    
    service = IngestionService(db)
    
    try:
        doc = await service.process_file(file)
        await db.commit()
        
        logger.info(
            "Document uploaded successfully: id='%s', chunks=%d",
            doc.id,
            doc.chunk_count
        )
        
        return DocumentUploadResponse(
            status="success",
            message="Document uploaded and processed successfully",
            data=DocumentData.model_validate(doc)
        )
        
    except ValueError as e:
        # Invalid file type
        logger.warning("Invalid file type uploaded: %s", str(e))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": str(e),
                "details": {"filename": file.filename}
            }
        )
        
    except Exception as e:
        logger.error(
            "Error uploading document '%s': %s",
            file.filename,
            str(e),
            exc_info=True
        )
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "An error occurred while processing the document",
                "details": {"error": str(e)}
            }
        )


@router.get(
    "/",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Documents",
    description="Retrieve a list of all uploaded documents, ordered by creation date (newest first)."
)
async def list_documents(db: AsyncSession = Depends(get_db)):
    """
    List all uploaded documents for the system.
    
    Returns documents ordered by creation date in descending order,
    with metadata including file size, chunk count, and timestamps.
    
    Args:
        db: Database session dependency
    
    Returns:
        DocumentListResponse: List of all documents with metadata
    
    Raises:
        HTTPException: 500 if database query fails
    """
    logger.info("Listing all documents")
    
    try:
        result = await db.execute(
            select(Document).order_by(Document.created_at.desc())
        )
        documents = result.scalars().all()
        
        logger.info("Retrieved %d document(s)", len(documents))
        
        return DocumentListResponse(
            status="success",
            message=f"Retrieved {len(documents)} document(s)",
            data=[DocumentData.model_validate(doc) for doc in documents]
        )
        
    except Exception as e:
        logger.error(
            "Error listing documents: %s",
            str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "An error occurred while retrieving documents",
                "details": {"error": str(e)}
            }
        )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Document Details",
    description="Retrieve detailed information about a specific document by its ID."
)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information for a specific document.
    
    Retrieves document metadata including filename, content type,
    file size, number of chunks, and timestamps.
    
    Args:
        document_id: UUID of the document to retrieve
        db: Database session dependency
    
    Returns:
        DocumentDetailResponse: Document metadata
    
    Raises:
        HTTPException: 404 if document not found
        HTTPException: 500 if database query fails
    """
    logger.info("Retrieving document: id='%s'", document_id)
    
    try:
        result = await db.execute(
            select(Document).where(Document.id == str(document_id))
        )
        doc = result.scalar_one_or_none()
        
        if not doc:
            logger.warning("Document not found: id='%s'", document_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "error",
                    "message": "Document not found",
                    "details": {"document_id": str(document_id)}
                }
            )
        
        logger.info("Document retrieved successfully: id='%s'", document_id)
        
        return DocumentDetailResponse(
            status="success",
            message="Document retrieved successfully",
            data=DocumentData.model_validate(doc)
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(
            "Error retrieving document '%s': %s",
            document_id,
            str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "An error occurred while retrieving the document",
                "details": {"error": str(e)}
            }
        )

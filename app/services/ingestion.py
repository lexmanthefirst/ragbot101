import uuid
from typing import List
from fastapi import UploadFile
from app.services.vector_store import get_vector_store
from app.services.llm_service import llm_service
from app.models.document import Document
from sqlalchemy.ext.asyncio import AsyncSession
from pypdf import PdfReader
import docx
import io

class IngestionService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.vector_store = get_vector_store()

    async def process_file(self, file: UploadFile) -> Document:
        content = await file.read()
        file_size = len(content)
        
        # 1. Extract Text
        text = await self._extract_text(content, file.filename, file.content_type)
        
        # 2. Chunk Text
        chunks = self._chunk_text(text)
        
        # 3. Create Document Record
        doc = Document(
            filename=file.filename,
            content_type=file.content_type,
            file_size=file_size,
            chunk_count=len(chunks)
        )
        self.db.add(doc)
        await self.db.flush()
        
        # 4. Embed and Store
        embeddings = []
        ids = []
        metadatas = []
        documents_for_vector_db = []
        
        for i, chunk in enumerate(chunks):
            embedding = await llm_service.get_embedding(chunk)
            embeddings.append(embedding)
            
            chunk_id = f"{doc.id}_{i}"
            ids.append(chunk_id)
            metadatas.append({
                "document_id": str(doc.id),
                "chunk_index": i,
                "source": file.filename
            })
            documents_for_vector_db.append(chunk)

        if documents_for_vector_db:
            self.vector_store.add_documents(
                documents=documents_for_vector_db,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )

        return doc

    async def _extract_text(self, content: bytes, filename: str, content_type: str) -> str:
        if content_type == "application/pdf" or filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
            
        elif content_type == "text/plain" or filename.endswith(".txt"):
            return content.decode("utf-8")
        
        else:
            raise ValueError(f"Unsupported file type: {content_type}")

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        # Simple character-based chunking with overlap
        if not text:
            return []
            
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks

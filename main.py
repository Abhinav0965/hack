from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

from services.document_service import DocumentService
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService
from services.vector_service import VectorService

load_dotenv()

app = FastAPI(title="HackRX Document Q&A API")
security = HTTPBearer()

# Initialize services
document_service = DocumentService()
embedding_service = EmbeddingService()
llm_service = LLMService()
vector_service = VectorService()

class QueryRequest(BaseModel):
    documents: str  # URL to document
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != os.getenv("BEARER_TOKEN"):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_document_queries(
    request: QueryRequest,
    token: HTTPAuthorizationCredentials = Depends(verify_token)
):
    try:
        # 1. Download and parse document
        document_text = await document_service.download_and_parse(request.documents)
        
        # 2. Chunk document into clauses
        chunks = document_service.chunk_document(document_text)
        
        # 3. Generate embeddings and store in Pinecone
        chunk_ids = await vector_service.store_chunks(chunks)
        
        # 4. Process each question
        answers = []
        for question in request.questions:
            # Retrieve relevant chunks
            relevant_chunks = await vector_service.semantic_search(question, top_k=5)
            
            # Generate answer using LLM
            answer = await llm_service.generate_answer(question, relevant_chunks)
            answers.append(answer)
        
        return QueryResponse(answers=answers)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
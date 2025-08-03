import pinecone
import os
from typing import List, Dict
from services.embedding_service import EmbeddingService
import uuid

class VectorService:
    def __init__(self):
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        self.index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))
        self.embedding_service = EmbeddingService()
    
    async def store_chunks(self, chunks: List[Dict[str, str]]) -> List[str]:
        """Store document chunks in Pinecone"""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = await self.embedding_service.get_embeddings(texts)
        
        vectors = []
        chunk_ids = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "section_id": chunk["section_id"],
                    **chunk["metadata"]
                }
            })
        
        self.index.upsert(vectors)
        return chunk_ids
    
    async def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks using semantic similarity"""
        query_embedding = await self.embedding_service.get_single_embedding(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            {
                "text": match["metadata"]["text"],
                "score": match["score"],
                "section_id": match["metadata"]["section_id"]
            }
            for match in results["matches"]
        ]
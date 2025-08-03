import openai
import os
from typing import List
import numpy as np

class EmbeddingService:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        response = await openai.Embedding.acreate(
            model="text-embedding-ada-002",
            input=texts
        )
        return [item['embedding'] for item in response['data']]
    
    async def get_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        embeddings = await self.get_embeddings([text])
        return embeddings[0]
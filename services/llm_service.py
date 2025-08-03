import openai
import os
from typing import List, Dict

class LLMService:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    async def generate_answer(self, question: str, relevant_chunks: List[Dict]) -> str:
        """Generate answer using GPT-4 based on retrieved chunks"""
        
        context = "\n\n".join([
            f"Section {chunk['section_id']}: {chunk['text']}"
            for chunk in relevant_chunks
        ])
        
        prompt = f"""
Given the user question and the following retrieved policy clauses, answer in plain English:

Question: {question}

Retrieved Clauses:
{context}

Instructions:
1. Decide if the policy covers the scenario or query described; give a clear Yes/No/Amount as needed.
2. Reference (quote or paraphrase) the clause(s) from the policy which support your answer.
3. Your answer must only use the retrieved clauses as evidence.
4. If the information is insufficient, state that clearly.

Answer:"""

        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a policy analysis expert. Provide accurate, evidence-based answers citing specific clauses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
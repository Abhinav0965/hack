import aiohttp
import PyPDF2
from docx import Document
import io
from typing import List, Dict
import re

class DocumentService:
    async def download_and_parse(self, url: str) -> str:
        """Download document from URL and extract text"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download document: {response.status}")
                
                content = await response.read()
                content_type = response.headers.get('content-type', '')
                
                if 'pdf' in content_type or url.lower().endswith('.pdf'):
                    return self._parse_pdf(content)
                elif 'word' in content_type or url.lower().endswith(('.docx', '.doc')):
                    return self._parse_docx(content)
                else:
                    # Assume text/email
                    return content.decode('utf-8')
    
    def _parse_pdf(self, content: bytes) -> str:
        """Extract text from PDF"""
        pdf_file = io.BytesIO(content)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def _parse_docx(self, content: bytes) -> str:
        """Extract text from Word document"""
        doc_file = io.BytesIO(content)
        doc = Document(doc_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def chunk_document(self, text: str) -> List[Dict[str, str]]:
        """Chunk document into logical sections"""
        # Split by headers, bullet points, or paragraphs
        sections = re.split(r'\n\s*(?=\d+\.|\w+\.|\•|\-)', text)
        
        chunks = []
        for i, section in enumerate(sections):
            if len(section.strip()) > 50:  # Filter out very short sections
                chunks.append({
                    "text": section.strip(),
                    "section_id": f"section_{i}",
                    "metadata": {"section_number": i}
                })
    ##
        return chunks
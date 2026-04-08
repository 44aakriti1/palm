from pathlib import Path
from io import BytesIO
import PyPDF2


class FileExtractor:
    
    @staticmethod
    def extract_text(file_content: bytes, filename: str) -> str:
        file_ext = Path(filename).suffix.lower()
        
        if file_ext == ".pdf":
            return FileExtractor._extract_pdf(file_content)
        elif file_ext == ".txt":
            return FileExtractor._extract_txt(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Only .pdf and .txt supported.")
    
    @staticmethod
    def _extract_pdf(content: bytes) -> str:
        text = ""
        pdf_file = BytesIO(content)
        reader = PyPDF2.PdfReader(pdf_file)
        
        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"
            
        return text.strip()
    
    @staticmethod
    def _extract_txt(content: bytes) -> str:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1")

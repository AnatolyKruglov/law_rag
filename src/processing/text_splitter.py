from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from config.settings import settings

class TextSplitter:
    """Handles document splitting with various strategies"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    def split_documents(self, documents: List[Document], method: str = "recursive") -> List[Document]:
        """Split documents into chunks"""
        if method == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
        else:
            splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        
        return splitter.split_documents(documents)
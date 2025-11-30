import os
from typing import List
from langchain.schema import Document

# PDF functionality (commented but kept)
"""
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredFileLoader
)
"""

from .consultant_plus_loader import ConsultantPlusLoader

class DocumentLoader:
    """Main document loader that can use both PDFs and Consultant Plus"""
    
    def __init__(self, source: str = None, use_consultant_plus: bool = True):
        self.source = source
        self.use_consultant_plus = use_consultant_plus
        self.consultant_loader = ConsultantPlusLoader()
    
    def load_documents(self, query: str = None) -> List[Document]:
        """Load documents from Consultant Plus based on query"""
        if self.use_consultant_plus and query:
            return self.consultant_loader.load_documents(query)
        
        # PDF functionality (commented but kept for reference)
        """
        if not self.source or not os.path.exists(self.source):
            raise ValueError(f"Document path does not exist: {self.source}")
        
        if os.path.isdir(self.source):
            loader = DirectoryLoader(
                self.source,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
        else:
            if self.source.endswith('.pdf'):
                loader = PyPDFLoader(self.source)
            else:
                loader = UnstructuredFileLoader(self.source)
        
        return loader.load()
        """
        
        raise ValueError("Either provide a query for Consultant Plus or enable PDF loading with valid source")

    def load_documents_from_query(self, query: str) -> List[Document]:
        """Convenience method to load documents from Consultant Plus using query"""
        return self.consultant_loader.load_documents(query)
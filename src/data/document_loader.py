import os
from typing import List
from langchain.schema import Document
from config.settings import settings, SearchMode

from .consultant_plus_loader import ConsultantPlusLoader
from .pptx_loader import PPTXLoader

class DocumentLoader:
    """Main document loader that supports multiple sources"""
    
    def __init__(self, use_consultant_plus: bool = None, use_pptx: bool = None):
        # Determine which sources to use based on settings
        if use_consultant_plus is None:
            use_consultant_plus = settings.SEARCH_MODE in [SearchMode.CONSULTANT_ONLY, SearchMode.BOTH]
        
        if use_pptx is None:
            use_pptx = settings.SEARCH_MODE in [SearchMode.PPTX_ONLY, SearchMode.BOTH]
        
        self.use_consultant_plus = use_consultant_plus
        self.use_pptx = use_pptx
        
        # Initialize loaders
        if self.use_consultant_plus:
            self.consultant_loader = ConsultantPlusLoader()
        
        if self.use_pptx:
            self.pptx_loader = PPTXLoader()
    
    def load_documents_from_query(self, query: str) -> List[Document]:
        """Load documents from all configured sources based on query"""
        all_documents = []
        
        # Load from Consultant Plus if enabled
        if self.use_consultant_plus:
            print("  Поиск на Консультант+")
            consultant_docs = self.consultant_loader.load_documents(query)
            print(f"  Нашлось {len(consultant_docs)} страниц на Консультант+")
            all_documents.extend(consultant_docs)
        
        # Load from PPTX files if enabled
        if self.use_pptx:
            print("  Поиск PPTX файлов...")
            pptx_docs = self.pptx_loader.load_documents_from_query(query)
            print(f"  Нашлось {len(pptx_docs)} PPTX файлов")
            all_documents.extend(pptx_docs)
        
        # print(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def load_documents(self, query: str = None) -> List[Document]:
        """Alias for load_documents_from_query for backward compatibility"""
        return self.load_documents_from_query(query)
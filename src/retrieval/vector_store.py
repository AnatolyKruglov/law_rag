from langchain.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Optional
import logging
from config.settings import settings
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages FAISS vector store operations with optimized batch processing"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_store = None
    
    def create_vector_store(self, documents: List[Document], batch_size: int = 5):  # Reduced batch size
        """Create FAISS vector store from documents with batching"""
        try:
            if len(documents) > batch_size:
                # Process in smaller batches to avoid rate limits
                logger.info(f"Creating vector store with {len(documents)} documents in batches of {batch_size}")
                
                # Initialize with first batch
                first_batch = documents[:batch_size]
                self.vector_store = FAISS.from_documents(first_batch, self.embeddings)
                logger.info(f"Initialized vector store with first {len(first_batch)} documents")
                
                # Add remaining documents in batches
                for i in tqdm(range(batch_size, len(documents), batch_size)):
                    batch = documents[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(documents) - 1) // batch_size + 1
                    # logger.info(f"Adding batch {batch_num}/{total_batches} ({len(batch)} documents)")
                    
                    self.vector_store.add_documents(batch)
                    
                    # More aggressive rate limiting between batches
                    if i + batch_size < len(documents):
                        import time
                        time.sleep(2)  # Increased to 2 seconds between batches
            else:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            logger.info(f"Created vector store with {len(documents)} documents")
            return self.vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def save_vector_store(self, path: str):
        """Save vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(path)
            logger.info(f"Vector store saved to {path}")
    
    def load_vector_store(self, path: str):
        """Load vector store from disk"""
        try:
            self.vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Vector store loaded from {path}")
            return self.vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def get_retriever(self, search_type: str = "similarity", **kwargs):
        """Get retriever from vector store"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        search_kwargs = {**settings.SEARCH_KWARGS, **kwargs}
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
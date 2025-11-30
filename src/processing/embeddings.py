from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from config.settings import settings
import logging
import time
import threading

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embedding generation with aggressive rate limiting"""
    
    def __init__(self):
        self.embeddings = YandexGPTEmbeddings(
            folder_id=settings.FOLDER_ID,
            api_key=settings.API_KEY
        )
        self.last_request_time = 0
        self.min_interval = 0.5  # Increased to 500ms between requests (2 requests per second)
        self.lock = threading.Lock()
    
    def get_embeddings(self):
        """Get embeddings instance"""
        return self.embeddings
    
    def _rate_limited_embed(self, texts):
        """Apply rate limiting to embedding requests"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                logger.info(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
            return self.embeddings.embed_documents(texts)
    
    def embed_documents(self, documents: list):
        """Embed a list of documents with rate limiting"""
        try:
            texts = [doc.page_content for doc in documents]
            
            # Process in even smaller batches
            batch_size = 3  # Further reduced batch size
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(texts) - 1) // batch_size + 1
                logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
                
                batch_embeddings = self._rate_limited_embed(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                # Longer delay between batches
                if i + batch_size < len(texts):
                    logger.info("Sleeping between batches...")
                    time.sleep(1.5)  # Increased delay
            
            return all_embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise e
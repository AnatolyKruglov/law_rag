import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Configuration settings for the RAG pipeline"""
    
    # Yandex Cloud credentials
    FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
    API_KEY = os.getenv("YANDEX_API_KEY")
    
    # Document processing
    CHUNK_SIZE = 2500  #1500
    CHUNK_OVERLAP = 500  #250
    
    # Model settings
    TEMPERATURE = 0.3
    MAX_TOKENS = 8000

    # Retrieval settings
    SEARCH_KWARGS = {"k": 10}  # Number of documents to retrieve
    SCORE_THRESHOLD = 0.7  # Minimum similarity score

settings = Settings()
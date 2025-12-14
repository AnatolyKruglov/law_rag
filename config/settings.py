import os
from dotenv import load_dotenv
from enum import Enum

load_dotenv()

class SearchMode(Enum):
    """Режимы поиска для RAG системы"""
    CONSULTANT_ONLY = "consultant_only"
    PPTX_ONLY = "pptx_only"
    BOTH = "both"

class DocumentType(Enum):
    """Типы документов для оптимизации чанкинга"""
    PPTX = "pptx"
    CONSULTANT = "consultant"
    MIXED = "mixed"

class Settings:
    """Configuration settings for the RAG pipeline"""
    
    # Yandex Cloud credentials
    FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
    API_KEY = os.getenv("YANDEX_API_KEY")
    
    # Общие настройки чанкинга (по умолчанию)
    CHUNK_SIZE = 2500
    CHUNK_OVERLAP = 500
    
    # Оптимизированные настройки для PPTX
    PPTX_CHUNK_SIZE = 1200  # Меньше, так как слайды обычно короче
    PPTX_CHUNK_OVERLAP = 300  # Больше пересечение для сохранения контекста между слайдами
    
    # Настройки для Consultant Plus (более длинные документы)
    CONSULTANT_CHUNK_SIZE = 3000
    CONSULTANT_CHUNK_OVERLAP = 600
    
    # Model settings
    TEMPERATURE = 0.3
    MAX_TOKENS = 8000

    # Retrieval settings
    SEARCH_KWARGS = {"k": 10}  # Number of documents to retrieve
    SCORE_THRESHOLD = 0.7  # Minimum similarity score
    
    # Search mode configuration
    SEARCH_MODE = SearchMode.BOTH  # consultant_only, pptx_only, both
    
    # PPTX folder path
    PPTX_FOLDER_PATH = r"D:\Документы\MIPT\право"
    
    # Chunking strategy
    PPTX_CHUNK_BY_SLIDE = True  # Создавать чанки по слайдам
    MIN_SLIDE_CHUNK_SIZE = 200  # Минимальный размер чанка для слайда
    MAX_SLIDE_CHUNK_SIZE = 2000  # Максимальный размер чанка для слайда

settings = Settings()
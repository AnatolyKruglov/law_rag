import logging
from typing import List

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_pipeline.log')
        ]
    )

def format_sources(source_documents: List) -> str:
    """Format source documents for display"""
    if not source_documents:
        return "No sources found"
    
    sources = []
    for i, doc in enumerate(source_documents, 1):
        source_info = f"Source {i}:"
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            source_info += f" {doc.metadata['source']}"
        if hasattr(doc, 'metadata') and 'page' in doc.metadata:
            source_info += f" (Page {doc.metadata['page']})"
        sources.append(source_info)
    
    return "\n".join(sources)
import logging
from typing import List
import faiss

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    
    # Настройка логгера Faiss
    faiss_logger = logging.getLogger('faiss')
    faiss_logger.setLevel(logging.WARNING)  # Только WARNING и выше
    faiss_logger.propagate = False  # Не передавать выше
    
    # Настройка loader отдельно
    faiss_loader_logger = logging.getLogger('faiss.loader')
    faiss_loader_logger.setLevel(logging.ERROR)  # Только ERROR и выше
    faiss_loader_logger.propagate = False
    
    # Основная настройка логгера
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_pipeline.log')
        ]
    )
    
    # Отключаем лишние логи от других библиотек
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

def format_sources(source_documents: List) -> str:
    """Format source documents for display"""
    if not source_documents:
        return "No sources found"
    
    sources = []
    consultant_count = 0
    pptx_count = 0
    other_count = 0
    
    for i, doc in enumerate(source_documents, 1):
        source_info = f"🔗 Источник {i}:"
        
        if hasattr(doc, 'metadata'):
            # Determine source type
            source_type = doc.metadata.get('type', 'unknown')
            chunk_type = doc.metadata.get('chunk_type', 'standard')
            
            if 'consultant' in doc.metadata.get('source', '').lower() or source_type == 'consultant':
                source_type_display = "Консультант Плюс"
                consultant_count += 1
            elif source_type == 'pptx' or doc.metadata.get('source', '').endswith('.pptx'):
                source_type_display = "PPTX"
                pptx_count += 1
                
                # Добавляем информацию о слайде для PPTX
                if 'slide_number' in doc.metadata:
                    source_type_display += f" (Слайд {doc.metadata['slide_number']})"
                elif 'slide_numbers' in doc.metadata:
                    source_type_display += f" (Слайды {doc.metadata['slide_numbers']})"
                
                if chunk_type != 'standard':
                    source_type_display += f" [{chunk_type}]"
            else:
                source_type_display = "Другой"
                other_count += 1
            
            source_info += f" {source_type_display}"
            
            if 'source' in doc.metadata:
                # Shorten long paths
                source = doc.metadata['source']
                if len(source) > 80:
                    if '\\' in source:
                        # Для Windows путей показываем только имя файла
                        filename = source.split('\\')[-1]
                        source_info += f" - {filename}"
                    else:
                        source_info += f" - ...{source[-60:]}"
                else:
                    source_info += f" - {source}"
            
            if 'title' in doc.metadata and doc.metadata['title']:
                title = doc.metadata['title'][:40]
                if len(doc.metadata['title']) > 40:
                    title += "..."
                source_info += f" - {title}"
            
            # Показываем размер чанка
            if hasattr(doc, 'page_content'):
                source_info += f" ({len(doc.page_content)} символов)"
        
        sources.append(source_info)
    
    # Add summary
    summary = f"\n📊 Итого источников: Консультант Плюс - {consultant_count}, PPTX - {pptx_count}, другие - {other_count}"
    sources.append(summary)
    
    return "\n".join(sources)

def analyze_document_stats(documents: List) -> dict:
    """Анализирует статистику документов для рекомендаций по чанкингу"""
    if not documents:
        return {}
    
    stats = {
        'total_documents': len(documents),
        'total_characters': 0,
        'avg_length': 0,
        'min_length': float('inf'),
        'max_length': 0,
        'type_distribution': {},
        'recommendations': []
    }
    
    for doc in documents:
        content_length = len(doc.page_content)
        stats['total_characters'] += content_length
        stats['min_length'] = min(stats['min_length'], content_length)
        stats['max_length'] = max(stats['max_length'], content_length)
        
        doc_type = doc.metadata.get('type', 'unknown')
        if doc_type not in stats['type_distribution']:
            stats['type_distribution'][doc_type] = 0
        stats['type_distribution'][doc_type] += 1
    
    stats['avg_length'] = stats['total_characters'] / len(documents) if documents else 0
    
    # Генерируем рекомендации
    if stats['avg_length'] < 1000:
        stats['recommendations'].append("Документы короткие. Используйте меньшие чанки (800-1200 символов).")
    elif stats['avg_length'] > 5000:
        stats['recommendations'].append("Документы длинные. Используйте бóльшие чанки (3000-4000 символов).")
    
    pptx_count = stats['type_distribution'].get('pptx', 0)
    if pptx_count > 0:
        stats['recommendations'].append(f"Найдено {pptx_count} PPTX файлов. Используйте стратегию чанкинга по слайдам.")
    
    return stats
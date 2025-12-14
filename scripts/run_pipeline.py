#!/usr/bin/env python3
"""
Main script to run the RAG pipeline with multiple sources
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.document_loader import DocumentLoader
from src.processing.text_splitter import TextSplitter, DocumentType
from src.processing.embeddings import EmbeddingManager
from src.retrieval.vector_store import VectorStoreManager
from src.generation.qa_chain import QASystem
from src.utils.helpers import setup_logging, format_sources, analyze_document_stats
from config.settings import settings, SearchMode

def analyze_documents(documents):
    """Анализирует документы и рекомендует настройки чанкинга"""
    if not documents:
        return None
    
    pptx_docs = [doc for doc in documents if doc.metadata.get('type') == 'pptx']
    consultant_docs = [doc for doc in documents if doc.metadata.get('type') != 'pptx']
    
    stats = {
        'total': len(documents),
        'pptx_count': len(pptx_docs),
        'consultant_count': len(consultant_docs),
        'avg_pptx_length': 0,
        'avg_consultant_length': 0
    }
    
    if pptx_docs:
        total_length = sum(len(doc.page_content) for doc in pptx_docs)
        stats['avg_pptx_length'] = total_length / len(pptx_docs)
    
    if consultant_docs:
        total_length = sum(len(doc.page_content) for doc in consultant_docs)
        stats['avg_consultant_length'] = total_length / len(consultant_docs)
    
    # print(f"\n📊 Анализ документов:")
    # print(f"   Всего документов: {stats['total']}")
    # print(f"   PPTX документов: {stats['pptx_count']}")
    # print(f"   Консультант+ документов: {stats['consultant_count']}")
    
    # if stats['pptx_count'] > 0:
    #     print(f"   Средняя длина PPTX: {stats['avg_pptx_length']:.0f} символов")
    
    # if stats['consultant_count'] > 0:
    #     print(f"   Средняя длина Консультант+: {stats['avg_consultant_length']:.0f} символов")
    
    return stats

def main(questions):
    setup_logging()
    
    try:        
        if settings.SEARCH_MODE == SearchMode.CONSULTANT_ONLY:
            print("📄 Используется только Consultant Plus")
            loader = DocumentLoader(use_consultant_plus=True, use_pptx=False)
            document_type = DocumentType.CONSULTANT
        elif settings.SEARCH_MODE == SearchMode.PPTX_ONLY:
            print("📊 Используются только PPTX файлы")
            loader = DocumentLoader(use_consultant_plus=False, use_pptx=True)
            document_type = DocumentType.PPTX
        else:  # BOTH
            print("🔗 Используются оба источника: Consultant Plus и PPTX файлы")
            loader = DocumentLoader(use_consultant_plus=True, use_pptx=True)
            document_type = DocumentType.MIXED
        
        for question in questions:
            print(f"❓ Вопрос: {question}")
            
            # 1. Load documents from configured sources
            print("📥 Загрузка документов...")
            documents = loader.load_documents_from_query(question)
            
            if not documents:
                print("⚠️  Не найдено документов по данному запросу")
                continue
                
            print(f"✅ Найдено {len(documents)} документов")
            
            # Анализируем документы и показываем рекомендации
            stats = analyze_documents(documents)
            
            # 2. Split documents с оптимизацией
            print("✂️  Обработка документов...")
            
            # Определяем оптимальные параметры
            splitter = TextSplitter(document_type=document_type)
            
            # Получаем рекомендации
            recommendations = splitter.get_optimal_settings(
                document_type.value,
                stats['avg_pptx_length'] if stats else None
            )
            
            # print(f"   Рекомендуемые настройки:")
            # print(f"   - Размер чанка: {recommendations['chunk_size']}")
            # print(f"   - Перекрытие: {recommendations['chunk_overlap']}")
            # print(f"   - Стратегия: {recommendations['strategy']}")
            # print(f"   - Обоснование: {recommendations['reasoning']}")
            
            chunks = splitter.split_documents(documents)
            # print(f"✅ Создано {len(chunks)} чанков из {len(documents)} документов")
            
            # Показываем статистику чанков
            if chunks:
                avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
                min_chunk = min(len(chunk.page_content) for chunk in chunks)
                max_chunk = max(len(chunk.page_content) for chunk in chunks)
                
                # print(f"   Статистика чанков:")
                # print(f"   - Средний размер: {avg_chunk_size:.0f} символов")
                # print(f"   - Минимальный: {min_chunk} символов")
                # print(f"   - Максимальный: {max_chunk} символов")
                
                # Предупреждение если чанки слишком большие или маленькие
                if avg_chunk_size > 2500:
                    print("   ⚠️  Чанки слишком большие. Рекомендуется уменьшить CHUNK_SIZE")
                elif avg_chunk_size < 500:
                    print("   ⚠️  Чанки слишком маленькие. Рекомендуется увеличить CHUNK_SIZE")
            
            # 3. Create embeddings and vector store with retry logic
            print("🧠 Создание эмбеддингов и векторного хранилища...")
            embedding_manager = EmbeddingManager()
            embeddings = embedding_manager.get_embeddings()
            
            vector_manager = VectorStoreManager(embeddings)
            
            # Add retry logic for vector store creation
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Используем меньший batch_size для PPTX
                    batch_size = 2 if document_type == DocumentType.PPTX else 3
                    vector_store = vector_manager.create_vector_store(chunks, batch_size=batch_size)
                    break
                except Exception as e:
                    if "rate quota limit exceed" in str(e) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 15  # 15, 30, 45 seconds
                        print(f"⏳ Лимит запросов, ждем {wait_time} секунд перед повторной попыткой {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise
            
            # 4. Create QA system
            print("🤖 Инициализация QA системы...")
            retriever = vector_manager.get_retriever()
            qa_system = QASystem(retriever)
            
            # 5. Make query         
            result = qa_system.query(question)
            
            print(f"\n📝 Ответ:")
            print(result['answer'])
            
            print(f"\n📚 Источники:")
            print(format_sources(result['source_documents']))
            
    except Exception as e:
        print(f"❌ Ошибка в пайплайне: {e}")
        raise

if __name__ == "__main__":
    main(
        questions=[
            "трудовой кодекс отпуск",
        ],
    )
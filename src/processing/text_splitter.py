from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document
from typing import List, Optional
from config.settings import settings, DocumentType
import re

class TextSplitter:
    """Handles document splitting with various strategies optimized for different document types"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, document_type: DocumentType = DocumentType.MIXED):
        self.document_type = document_type
        
        # Устанавливаем размеры чанков в зависимости от типа документа
        if document_type == DocumentType.PPTX:
            self.chunk_size = chunk_size or settings.PPTX_CHUNK_SIZE
            self.chunk_overlap = chunk_overlap or settings.PPTX_CHUNK_OVERLAP
            self.separators = ["\n\nСлайд", "\n\n", "\n", ". ", "! ", "? ", " ", ""]
        elif document_type == DocumentType.CONSULTANT:
            self.chunk_size = chunk_size or settings.CONSULTANT_CHUNK_SIZE
            self.chunk_overlap = chunk_overlap or settings.CONSULTANT_CHUNK_OVERLAP
            self.separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        else:  # MIXED или по умолчанию
            self.chunk_size = chunk_size or settings.CHUNK_SIZE
            self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
            self.separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    
    def split_documents(self, documents: List[Document], method: str = "recursive") -> List[Document]:
        """Split documents into chunks with type-specific optimization"""
        
        # Если есть PPTX документы и включена оптимизация по слайдам
        pptx_docs = [doc for doc in documents if doc.metadata.get('type') == 'pptx']
        other_docs = [doc for doc in documents if doc.metadata.get('type') != 'pptx']
        
        all_chunks = []
        
        # Обрабатываем PPTX документы с особой стратегией
        if pptx_docs and settings.PPTX_CHUNK_BY_SLIDE:
            all_chunks.extend(self._split_pptx_by_slides(pptx_docs))
        
        # Обрабатываем остальные документы стандартным способом
        if other_docs:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators
            )
            all_chunks.extend(splitter.split_documents(other_docs))
        
        return all_chunks
    
    def _split_pptx_by_slides(self, pptx_documents: List[Document]) -> List[Document]:
        """Специальная обработка PPTX документов по слайдам"""
        chunks = []
        
        for doc in pptx_documents:
            content = doc.page_content
            metadata = doc.metadata.copy()
            
            # Разделяем по слайдам (предполагается, что слайды разделены маркерами)
            slide_pattern = r'(?:Слайд \d+:|^|\n\n)(.*?)(?=(?:Слайд \d+:|$))'
            slides = re.findall(slide_pattern, content, re.DOTALL)
            
            if not slides:
                # Если не нашли разделение по слайдам, используем стандартное разделение
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=settings.PPTX_CHUNK_SIZE,
                    chunk_overlap=settings.PPTX_CHUNK_OVERLAP,
                    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
                )
                doc_chunks = splitter.split_documents([doc])
                chunks.extend(doc_chunks)
                continue
            
            # Обрабатываем каждый слайд
            for i, slide_content in enumerate(slides):
                slide_content = slide_content.strip()
                if not slide_content:
                    continue
                
                # Если слайд слишком длинный, разбиваем его дальше
                if len(slide_content) > settings.MAX_SLIDE_CHUNK_SIZE:
                    slide_chunks = self._split_long_slide(slide_content, i+1, metadata)
                    chunks.extend(slide_chunks)
                elif len(slide_content) >= settings.MIN_SLIDE_CHUNK_SIZE:
                    # Создаем чанк из одного слайда
                    slide_metadata = metadata.copy()
                    slide_metadata['slide_number'] = i + 1
                    slide_metadata['chunk_type'] = 'full_slide'
                    slide_metadata['original_slide_length'] = len(slide_content)
                    
                    chunk = Document(
                        page_content=slide_content,
                        metadata=slide_metadata
                    )
                    chunks.append(chunk)
                else:
                    # Если слайд слишком короткий, объединяем со следующим
                    if i < len(slides) - 1:
                        next_slide = slides[i + 1].strip()
                        combined_content = slide_content + "\n\n" + next_slide
                        
                        if len(combined_content) <= settings.MAX_SLIDE_CHUNK_SIZE:
                            slide_metadata = metadata.copy()
                            slide_metadata['slide_numbers'] = f"{i+1}-{i+2}"
                            slide_metadata['chunk_type'] = 'combined_slides'
                            
                            chunk = Document(
                                page_content=combined_content,
                                metadata=slide_metadata
                            )
                            chunks.append(chunk)
                            # Пропускаем следующий слайд, так как мы его уже объединили
                            slides[i + 1] = ""
        
        return chunks
    
    def _split_long_slide(self, slide_content: str, slide_num: int, metadata: dict) -> List[Document]:
        """Разбивает длинный слайд на несколько чанков"""
        chunks = []
        
        # Используем рекурсивное разделение для длинных слайдов
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.PPTX_CHUNK_SIZE,
            chunk_overlap=settings.PPTX_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Создаем временный документ для разделения
        temp_doc = Document(page_content=slide_content, metadata=metadata)
        slide_chunks = splitter.split_documents([temp_doc])
        
        # Обновляем метаданные для каждого чанка
        for i, chunk in enumerate(slide_chunks):
            chunk_metadata = chunk.metadata.copy()
            chunk_metadata['slide_number'] = slide_num
            chunk_metadata['chunk_type'] = 'slide_part'
            chunk_metadata['part_number'] = i + 1
            chunk_metadata['total_parts'] = len(slide_chunks)
            
            chunks.append(Document(
                page_content=chunk.page_content,
                metadata=chunk_metadata
            ))
        
        return chunks
    
    def get_optimal_settings(self, document_type: str, avg_content_length: int = None) -> dict:
        """Возвращает оптимальные настройки для конкретного типа документов"""
        
        recommendations = {
            'pptx': {
                'chunk_size': settings.PPTX_CHUNK_SIZE,
                'chunk_overlap': settings.PPTX_CHUNK_OVERLAP,
                'strategy': 'by_slide' if settings.PPTX_CHUNK_BY_SLIDE else 'recursive',
                'reasoning': 'PPTX слайды обычно содержат 50-500 слов. Меньшие чанки с бóльшим перекрытием сохраняют контекст между слайдами.'
            },
            'consultant': {
                'chunk_size': settings.CONSULTANT_CHUNK_SIZE,
                'chunk_overlap': settings.CONSULTANT_CHUNK_OVERLAP,
                'strategy': 'recursive',
                'reasoning': 'Юридические документы длинные и структурированные. Большие чанки сохраняют контекст статей.'
            },
            'mixed': {
                'chunk_size': settings.CHUNK_SIZE,
                'chunk_overlap': settings.CHUNK_OVERLAP,
                'strategy': 'adaptive',
                'reasoning': 'Баланс между разными типами документов.'
            }
        }
        
        if avg_content_length:
            # Динамическая корректировка на основе средней длины
            if document_type == 'pptx':
                if avg_content_length < 500:
                    recommendations['pptx']['chunk_size'] = 800
                    recommendations['pptx']['chunk_overlap'] = 250
                elif avg_content_length > 2000:
                    recommendations['pptx']['chunk_size'] = 1500
                    recommendations['pptx']['chunk_overlap'] = 400
        
        return recommendations.get(document_type, recommendations['mixed'])
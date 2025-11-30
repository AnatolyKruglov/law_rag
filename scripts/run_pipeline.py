#!/usr/bin/env python3
"""
Main script to run the RAG pipeline
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.document_loader import DocumentLoader
from src.processing.text_splitter import TextSplitter
from src.processing.embeddings import EmbeddingManager
from src.retrieval.vector_store import VectorStoreManager
from src.generation.qa_chain import QASystem
from src.utils.helpers import setup_logging, format_sources
from config.settings import settings

def main(questions):
    setup_logging()
    
    try:
        # Initialize document loader for Consultant Plus
        print("Initializing Consultant Plus loader...")
        loader = DocumentLoader(use_consultant_plus=True)
        
        for question in questions:
            print(f"\n{'='*50}")
            print(f"Question: {question}")
            print(f"{'='*50}")
            
            # 1. Load documents from Consultant Plus based on question
            print("Searching Consultant Plus...")
            documents = loader.load_documents_from_query(question)
            
            if not documents:
                print("No documents found for this query")
                continue
                
            print(f"Found {len(documents)} relevant documents")
            
            # 2. Split documents
            print("Processing documents...")
            splitter = TextSplitter()
            chunks = splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks from {len(documents)} documents")
            
            # 3. Create embeddings and vector store with retry logic
            print("Creating embeddings and vector store...")
            embedding_manager = EmbeddingManager()
            embeddings = embedding_manager.get_embeddings()
            
            vector_manager = VectorStoreManager(embeddings)
            
            # Add retry logic for vector store creation
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vector_store = vector_manager.create_vector_store(chunks, batch_size=3)  # Small batch size
                    break
                except Exception as e:
                    if "rate quota limit exceed" in str(e) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 15  # 15, 30, 45 seconds
                        print(f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise
            
            # 4. Create QA system
            print("Initializing QA system...")
            retriever = vector_manager.get_retriever()
            qa_system = QASystem(retriever)
            
            # 5. Make query
            # system_prompt = "Ты специалист по российскому праву. Подробно объясни ответ, указав ссылки на источники"
            system_prompt = "Ты специалист по российскому праву. Твоя задача: ответить на вопрос юзера, подробно объяснить ответ, дать пояснения простым языком (при необходимости - предоставить понятный алгоритм действий). Обязательно указать ссылки на источники"

            
            result = qa_system.query(question, system_prompt)
            print(f"Answer: {result['answer']}")
            print(f"\nSources:\n{format_sources(result['source_documents'])}")
            
    except Exception as e:
        print(f"Error in pipeline: {e}")
        raise

if __name__ == "__main__":
    main(
        questions=[
            "трудовой кодекс отпуск",
        ],
    )
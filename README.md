# RAG Law Assistant

A Retrieval-Augmented Generation (RAG) pipeline using YandexGPT, FAISS and Консультант+ Russian law database (via their site).

## Features

- Document loading and preprocessing from PDFs
- Intelligent text splitting with multiple strategies
- Vector embeddings using YandexGPT
- FAISS vector store for efficient similarity search
- Customizable QA system with source citation

## Setup and run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run:
```bash
python scripts/run_pipeline.py
```

or from notebook


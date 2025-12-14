# RAG Law Assistant

A Retrieval-Augmented Generation (RAG) pipeline using YandexGPT, FAISS and Консультант+ Russian law database (via their site)
<img width="1961" height="1165" alt="image" src="https://github.com/user-attachments/assets/16964623-a68f-4135-b787-d4eefd727fba" />

## Features

- Document loading and preprocessing from PowerPoint presentations
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

or in notebook run.ipynb
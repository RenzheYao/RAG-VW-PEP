hardware: CPU only ram 16gb
Local LLM used: Qwen 3 -4 B Q6  /  Gemma4 E2B Q4

Local RAG System Technical Summary: Volkswagen PEP Guidelines

1. Initial Testing & Chunking Strategy (rag_vw_pep.py)

Initial Validation: An initial test was successfully conducted using the baseline rag_vw_pep.py script to ingest and process Volkswagen PEP PDF guidelines (such as the Lieferantenleitfaden).

Chunking Strategy: The pipeline splits long-form technical guidelines into overlapping text fragments (chunks) ranging between 500 to 1,000 characters. This ensures that process steps, quality gates, and milestone definitions retain complete semantic context during vector lookup.

2. Model Performance Optimization (Qwen3 4B → Gemma 4 E2B)

Performance Bottleneck: The initial test model, Qwen3 4B, exhibited excessive latency during inference on the laptop's CPU-only hardware (16GB RAM).

Model Pivot: The system was migrated to Gemma 4 E2B as the active local LLM via Ollama. Its lower parameter count and optimized edge execution significantly increase token generation speed, making local interactive querying viable.

3. Web Stack & Current Architecture (rag_vw_pep_flask.py)

Application Framework: The production script (rag_vw_pep_flask.py) uses Flask as the backend service layer combined with a Gradio web interface for user interaction.

Current Vector Store: ChromaDB is configured inside .vector_db on drive D:\ to handle persistent embeddings and document metadata without using C: drive storage.

4. Scaling Plan (Transition to Faiss)

Scalability: If the document repository expands substantially and causes query overhead, the vector retrieval layer is planned to migrate from ChromaDB to Faiss (Facebook AI Similarity Search).

Archived Implementation: The preliminary script configured for Faiss has been stored in the archiv directory for future migration when needed.
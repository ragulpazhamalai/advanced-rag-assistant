# Advanced RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) project built with LangChain, OpenAI, ChromaDB, BM25, Streamlit, and advanced retrieval techniques.

This project started as a basic PDF question-answering system and was progressively upgraded into a conversational, evaluated, structured RAG assistant.

---

## Project Overview

This application allows users to ask questions from a PDF knowledge base.

It combines multiple retrieval techniques to improve answer quality, reduce irrelevant context, support follow-up questions, and provide structured responses with confidence, sources, and supporting evidence.

The project also includes an evaluation framework to measure retrieval and answer quality.

---

## Key Features

- PDF ingestion and chunking
- OpenAI embeddings
- Chroma vector database
- Parent-child retrieval
- Hybrid search
  - Vector search
  - BM25 keyword search
- Reciprocal Rank Fusion (RRF)
- Multi-query retrieval
- Cross-query fusion
- Contextual compression
- Conversational memory
- Query rewriting for follow-up questions
- Structured output using Pydantic
- Confidence scoring
- Source attribution
- Supporting evidence
- Suggested follow-up questions
- RAG evaluation framework
- Streamlit chat interface

---

## Architecture

```text
                         USER
                          │
                          ▼
                Conversation Memory
                          │
                          ▼
                   Query Rewriting
                          │
                          ▼
                 Multi-Query Generator
                          │
                          ▼
             ┌─────────────────────────┐
             │     Hybrid Retrieval    │
             │                         │
             │  Vector Search + BM25   │
             └────────────┬────────────┘
                          │
                          ▼
                    RRF Fusion
                          │
                          ▼
                Cross-Query Fusion
                          │
                          ▼
                 Child Chunk Results
                          │
                          ▼
                  Parent Retrieval
                          │
                          ▼
              Contextual Compression
                          │
                          ▼
                 Relevant Context
                          │
                          ▼
               Structured LLM Output
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
          Answer      Confidence     Sources
             │
             ├──── Supporting Evidence
             │
             └──── Suggested Questions

## Development Journey

### Phase 1 — Basic RAG
PDF loading, chunking, embeddings, Chroma vector search, and LLM answer generation.

### Phase 2 — Parent Retrieval
Added parent-child chunking to improve retrieval precision while preserving broader context.

### Phase 3 — Hybrid Search
Combined vector search with BM25 keyword retrieval using Reciprocal Rank Fusion (RRF).

### Phase 4 — Multi-Query Retrieval
Generated multiple search queries and fused results across queries to improve recall.

### Phase 5 — Contextual Compression
Reduced irrelevant parent context before sending it to the final LLM.

### Phase 6 — Conversational Memory
Added history-aware query rewriting for follow-up questions such as:
`What is PCA?` → `What are the cons?`

### Phase 7 — Structured Output
Returned answer, confidence, sources, supporting evidence, and follow-up questions using Pydantic.

### Phase 8 — RAG Evaluation
Evaluated retrieval and generation using Hit@K, MRR, correctness, groundedness, context relevance, and refusal tests.

### Phase 9 — Streamlit UI
Built an interactive chat interface with confidence, source attribution, supporting evidence, and session management.
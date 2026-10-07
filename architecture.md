# Advanced RAG AI Research Assistant — Architecture

## 1. Project Goal

Build a production-oriented AI Research & Document Intelligence Assistant that can answer questions over user-provided documents using advanced Retrieval-Augmented Generation (RAG).

Core retrieval and conversation capabilities:

- Parent-child document retrieval
- Hybrid search: dense vector search + keyword/BM25 search
- Multi-query retrieval / query expansion
- Contextual compression
- Conversational memory
- Source attribution and structured responses
- Modular architecture so individual retrieval components can be replaced or evaluated independently

---

## 2. High-Level Architecture

```text
                           ┌──────────────────────┐
                           │        USER          │
                           │ Upload + Ask Question│
                           └──────────┬───────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
                 ▼                                         ▼
        DOCUMENT INGESTION                         QUERY / CHAT FLOW
                 │                                         │
                 ▼                                         ▼
        Document Loader                          Conversation Memory
                 │                                         │
                 ▼                                         ▼
         Text Cleaning                              Query Rewriting
                 │                                         │
                 ▼                                         ▼
      Create Parent Documents                  Multi-Query Generation
                 │                                         │
                 ▼                                ┌────────┴────────┐
       Create Child Chunks                       │                 │
                 │                               ▼                 ▼
        ┌────────┴─────────┐              Original Query      Expanded Queries
        │                  │                       │                 │
        ▼                  ▼                       └────────┬────────┘
   Vector Index        Keyword Index                         ▼
  (Embeddings)            (BM25)                     Hybrid Retrieval
        │                  │                       ┌────────┴────────┐
        │                  │                       │                 │
        └──────────────────┘                       ▼                 ▼
                                             Dense Search       Keyword Search
                                                      \           /
                                                       \         /
                                                        ▼       ▼
                                                          Fusion
                                                            │
                                                            ▼
                                                   Candidate Child Chunks
                                                            │
                                                            ▼
                                                    Parent Retrieval
                                                            │
                                                            ▼
                                                Contextual Compression
                                                            │
                                                            ▼
                                                     Final Context
                                                            │
                                                            ▼
                                                      Prompt Builder
                                                            │
                                                            ▼
                                                         LLM
                                                            │
                                                            ▼
                                                   Structured Response
                                                            │
                                          ┌─────────────────┴─────────────────┐
                                          ▼                                   ▼
                                      User Answer                         Sources
                                          │
                                          ▼
                                  Save Conversation
```

---

## 3. Design Principles

### 3.1 Modular components
Each major operation should have its own module/class. Avoid putting ingestion, retrieval, memory, prompting, and UI in one file.

### 3.2 Retrieval before generation
The LLM should receive retrieved evidence rather than relying only on its parametric knowledge.

### 3.3 Child chunks for precision, parent documents for context
Small child chunks are optimized for search. Larger parent documents are returned to preserve surrounding context.

### 3.4 Hybrid retrieval
Dense retrieval captures semantic similarity. Keyword retrieval improves exact-term, IDs, names, numbers, and domain-specific phrase matching.

### 3.5 Query expansion
A user query may be ambiguous or too narrow. Generate multiple search formulations to improve recall.

### 3.6 Compression before generation
Reduce retrieved context to information relevant to the current question to control token usage and reduce distraction.

### 3.7 Explicit conversation state
Conversation history should be associated with a session ID and passed to the query-understanding and generation stages where needed.

### 3.8 Structured outputs
Return predictable fields for UI rendering, evaluation, logging, and downstream applications.

---

## 4. Project Structure

```text
advanced-rag-assistant/
│
├── app.py                         # Streamlit entry point
├── config.py                      # Environment/configuration
├── requirements.txt
├── .env                           # Secrets; never commit
├── .gitignore
├── README.md
├── architecture.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py              # PDF/DOCX/TXT loading
│   │   ├── cleaner.py             # Text normalization
│   │   └── chunker.py             # Parent/child chunk creation
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── vector_store.py        # Chroma/Qdrant adapter
│   │   ├── keyword_store.py       # BM25 keyword index
│   │   └── metadata_store.py      # Parent/child mapping + metadata
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_retriever.py    # Dense retrieval
│   │   ├── keyword_retriever.py   # BM25 retrieval
│   │   ├── hybrid_retriever.py    # Fusion / ranking
│   │   ├── multi_query.py         # Query generation
│   │   ├── parent_retriever.py    # Child -> parent expansion
│   │   └── compressor.py          # Contextual compression
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   └── conversation.py        # Session-based chat memory
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── prompts.py             # Prompt templates
│   │   └── response.py            # Pydantic response schemas
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py           # Retrieval/answer evaluation
│   │
│   └── assistant.py               # Orchestrates complete RAG pipeline
│
├── data/
│   ├── documents/                 # Uploaded source files
│   └── chroma/                    # Local vector DB during development
│
└── tests/
    ├── test_ingestion.py
    ├── test_retrieval.py
    ├── test_memory.py
    └── test_assistant.py
```

---

## 5. Ingestion Pipeline

### 5.1 Load documents

Function:

```python
load_documents(file_path: str) -> list[Document]
```

Responsibilities:

- Detect supported file type
- Extract text
- Preserve source metadata
- Preserve useful page/section metadata where available

Flow:

```text
File
 ↓
Loader
 ↓
LangChain Document objects
```

### 5.2 Clean documents

Function:

```python
clean_documents(documents: list[Document]) -> list[Document]
```

Responsibilities:

- Normalize whitespace
- Remove obvious extraction noise
- Preserve meaningful headings
- Preserve metadata

Avoid aggressive cleaning that destroys document structure.

### 5.3 Create parent documents

Function:

```python
create_parent_documents(documents: list[Document]) -> list[ParentDocument]
```

A parent should represent a meaningful larger context such as a section, page group, or logical document region.

Each parent receives a stable ID:

```text
parent_id = P000123
```

### 5.4 Create child chunks

Function:

```python
create_child_chunks(parent_documents) -> list[Document]
```

Each child stores:

```text
child_id
parent_id
source
page
text
```

Conceptual structure:

```text
Parent P001
 ├── Child C001
 ├── Child C002
 ├── Child C003
 └── Child C004
```

Child chunks are indexed for retrieval.

### 5.5 Indexing

The child chunks are written to two retrieval systems:

```text
                    Child Chunks
                         │
                ┌────────┴────────┐
                ▼                 ▼
          Embedding Model      BM25 Index
                │                 │
                ▼                 ▼
          Vector Database     Keyword Index
```

The parent store keeps the mapping needed to retrieve the larger parent context.

---

## 6. Query Pipeline

### 6.1 Receive user query

Function:

```python
answer_question(question: str, session_id: str) -> ResearchResponse
```

### 6.2 Load conversation memory

Function:

```python
get_conversation_context(session_id: str)
```

The latest relevant conversation turns are used to understand references such as:

- "the second point"
- "that model"
- "what about its limitations?"

Do not blindly send the entire conversation indefinitely. Use a bounded window and/or summarization strategy.

### 6.3 Rewrite ambiguous conversational query

Function:

```python
rewrite_query(question, conversation_context) -> str
```

Example:

```text
Previous conversation:
User: What are the components of RAG?
Assistant: Retrieval, knowledge store, generation...

User: How does the second component work?

Rewritten search query:
How does the knowledge store component of a RAG system work?
```

### 6.4 Generate multiple queries

Function:

```python
generate_multi_queries(query: str, n: int = 3) -> list[str]
```

Example:

```text
Original:
How does RAG improve LLM accuracy?

Expanded:
1. How does retrieval improve LLM accuracy?
2. How does RAG reduce hallucinations?
3. How does external knowledge improve LLM responses?
```

The multi-query stage should improve recall, not invent unrelated search topics.

---

## 7. Hybrid Retrieval

Function:

```python
hybrid_search(queries: list[str], top_k: int) -> list[RetrievedChunk]
```

### 7.1 Dense retrieval

```text
Query
 ↓
Embedding
 ↓
Vector similarity search
 ↓
Top-K child chunks
```

Strengths:

- Semantic similarity
- Paraphrases
- Conceptual questions

Weaknesses:

- Exact IDs/numbers/codes may be weaker
- Similar meaning can sometimes produce false positives

### 7.2 Keyword retrieval

```text
Query
 ↓
Token matching / BM25
 ↓
Top-K child chunks
```

Strengths:

- Exact names
- IDs
- Product codes
- Technical terms
- Rare words

### 7.3 Fusion

Combine dense and keyword candidates.

Recommended initial design:

```text
Vector results
      +
BM25 results
      ↓
Deduplicate by child_id
      ↓
Normalize scores
      ↓
Weighted fusion / Reciprocal Rank Fusion
      ↓
Final ranked child chunks
```

A first implementation can use Reciprocal Rank Fusion (RRF) because it is easy to reason about and does not require directly comparable score scales.

---

## 8. Parent Document Retrieval

Function:

```python
retrieve_parent_documents(child_results) -> list[ParentDocument]
```

Flow:

```text
Hybrid retrieval
     ↓
Child C031
     ↓
parent_id = P010
     ↓
Retrieve Parent P010
```

If several children map to the same parent, deduplicate the parent.

Recommended metadata:

```text
parent_id
child_id
source
page
section
chunk_index
```

---

## 9. Contextual Compression

Function:

```python
compress_context(question, parent_documents) -> list[CompressedDocument]
```

Flow:

```text
Parent documents
       ↓
LLM-based relevance extraction / compression
       ↓
Question-focused passages
       ↓
Compact context
```

Goals:

- Remove irrelevant paragraphs
- Keep facts needed to answer the question
- Preserve source metadata
- Reduce prompt size

Compression must not rewrite facts into unsupported claims. Prefer extraction-oriented compression over free-form summarization when factual traceability is important.

---

## 10. Prompt Construction

Function:

```python
build_prompt(question, conversation, compressed_context) -> PromptValue
```

The prompt should establish:

1. The assistant's role
2. Grounding rules
3. Retrieved source context
4. Relevant conversation history
5. The current question
6. Output schema requirements

Core grounding instruction:

```text
Answer using the retrieved evidence.
Do not invent information that is not supported by the evidence.
If the evidence is insufficient, explicitly state that the documents do not provide enough information.
```

---

## 11. Structured Response

Recommended schema:

```python
class ResearchResponse(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"]
    sources: list[str]
    supporting_evidence: list[str]
    follow_up_questions: list[str]
```

Potential future fields:

```text
retrieved_chunks
retrieval_method
query_variants
latency_ms
```

These are useful for evaluation/debugging but should not necessarily be displayed to the end user.

---

## 12. Conversational Memory

Function interface:

```python
get_history(session_id)
save_turn(session_id, question, answer)
clear_session(session_id)
```

Session flow:

```text
session_id
   ↓
Conversation history
   ↓
Query rewriting
   ↓
Retrieval / answer generation
   ↓
Save new turn
```

Initial implementation:

- In-memory sessions for development
- Bounded recent-message window

Production-ready option:

- Redis/PostgreSQL-backed conversation store

Memory should be treated separately from the document knowledge base.

---

## 13. End-to-End Assistant Orchestrator

Main class:

```python
class AdvancedRAGAssistant:
```

Primary methods:

```python
add_documents(...)
answer_question(...)
clear_session(...)
get_session_messages(...)
get_sources(...)
```

Internal query flow:

```python
answer_question(question, session_id)
    │
    ├── get_conversation_context()
    ├── rewrite_query()
    ├── generate_multi_queries()
    ├── hybrid_search()
    ├── retrieve_parent_documents()
    ├── compress_context()
    ├── build_prompt()
    ├── generate_structured_response()
    └── save_turn()
```

---

## 14. Recommended Class Responsibilities

### DocumentLoader
Loads raw files and produces normalized `Document` objects.

### DocumentChunker
Creates parent documents and child chunks, assigning IDs and metadata.

### VectorStore
Stores embeddings and performs dense retrieval.

### KeywordRetriever
Maintains BM25/keyword retrieval.

### HybridRetriever
Runs both retrieval methods and fuses rankings.

### MultiQueryGenerator
Creates alternative search queries.

### ParentRetriever
Maps child results to larger parent contexts.

### ContextCompressor
Extracts only question-relevant information from parent contexts.

### ConversationMemory
Maintains per-session message history.

### ResponseGenerator
Builds prompts and returns a validated Pydantic response.

### AdvancedRAGAssistant
Coordinates the full workflow.

---

## 15. Data Model

### Document metadata

```python
{
    "source": "research_paper.pdf",
    "page": 12,
    "parent_id": "P00042",
    "child_id": "C00167",
    "section": "Methodology",
    "chunk_index": 3,
    "indexed_at": "2026-09-07T09:00:00"
}
```

### Retrieval result

```python
class RetrievalResult(BaseModel):
    child_id: str
    parent_id: str
    source: str
    page: int | None
    content: str
    score: float | None
    retriever: str
```

---

## 16. Error Handling

The application should gracefully handle:

- Unsupported file types
- Empty documents
- Failed text extraction
- Missing API keys
- Embedding API failures
- Vector DB failures
- LLM timeouts
- Invalid structured output
- Empty retrieval results
- Compression failures
- Session/memory errors

User-facing behavior should be clear and non-technical. Internal logs should retain enough information for debugging.

---

## 17. Security and Configuration

Secrets belong in `.env` and should never be committed.

Example:

```text
OPENAI_API_KEY=...
```

Use configuration variables for:

```text
LLM model
Embedding model
Chunk size
Chunk overlap
Top-K values
Hybrid weights / RRF parameters
Compression limits
Memory window
Vector DB path
```

---

## 18. Retrieval Configuration — Initial Baseline

Start with simple, measurable defaults and tune later.

```text
Parent size:          ~2000–4000 tokens
Child size:           ~300–600 tokens
Child overlap:        ~50–100 tokens
Dense top-k:          8–12
BM25 top-k:           8–12
Hybrid candidate k:   10–20
Parent candidates:    4–8
Compressed context:   question-focused
Conversation window:  last 6–10 messages
Multi-query count:    3–4
```

These are starting points, not fixed optimal values. Tune them against an evaluation dataset.

---

## 19. Evaluation Strategy

The project should include evaluation rather than relying only on subjective testing.

### Retrieval metrics

- Recall@K
- Precision@K where ground truth is available
- MRR
- Hit Rate

### Generation metrics

- Faithfulness / groundedness
- Answer relevance
- Citation correctness
- Completeness

### System metrics

- End-to-end latency
- Retrieval latency
- LLM latency
- Token usage
- Cost per question

### Compare retrieval strategies

At minimum evaluate:

```text
A. Vector only
B. BM25 only
C. Hybrid
D. Hybrid + Multi-Query
E. Hybrid + Multi-Query + Parent Retrieval
F. Full pipeline + Compression
```

The goal is to demonstrate whether each advanced component actually improves the system.

---

## 20. Observability / Debugging

For development, expose a debug mode showing:

```text
Original query
Rewritten query
Generated query variants
Vector results
BM25 results
Fusion ranking
Selected parent documents
Compressed context
Final sources
Latency
```

This is essential for understanding why a RAG system retrieves the wrong information.

In production, use structured logs instead of printing raw debugging information to the UI.

---

## 21. Streamlit UI

The initial UI should contain:

```text
┌──────────────────────────────────────────────┐
│        AI Research & Document Assistant      │
├──────────────────────────────────────────────┤
│ Upload documents                             │
│ [PDF] [DOCX] [TXT]                           │
│                                              │
│ Indexed sources: 5                           │
├──────────────────────────────────────────────┤
│ Chat                                         │
│                                              │
│ User: How does RAG reduce hallucinations?    │
│                                              │
│ Assistant: ...                               │
│                                              │
│ Sources:                                      │
│  • rag_survey.pdf, page 4                    │
│                                              │
│ Confidence: HIGH                             │
├──────────────────────────────────────────────┤
│ Ask a question...                      [Send]│
└──────────────────────────────────────────────┘
```

A developer/debug panel can optionally show retrieval details.

---

## 22. Development Phases

### Phase 1 — Foundation

```text
Project setup
↓
Config
↓
Document loading
↓
Chunking
↓
Embeddings
↓
Vector DB
↓
Basic RAG
```

### Phase 2 — Parent Retrieval

```text
Parent IDs
↓
Child IDs
↓
Child retrieval
↓
Parent expansion
```

### Phase 3 — Hybrid Search

```text
Vector search
+
BM25
↓
Fusion
```

### Phase 4 — Multi-Query

```text
Conversation-aware query
↓
Multiple query variants
↓
Hybrid retrieval
```

### Phase 5 — Contextual Compression

```text
Parent documents
↓
Relevant passage extraction
↓
Compact context
```

### Phase 6 — Conversation Memory

```text
Session store
↓
History-aware query rewriting
↓
History-aware answer generation
```

### Phase 7 — Structured Output

```text
Pydantic response
↓
Sources
Confidence
Evidence
Follow-ups
```

### Phase 8 — Evaluation

```text
Create question/ground-truth dataset
↓
Measure retrieval
↓
Measure answer quality
↓
Tune pipeline
```

### Phase 9 — UI & Deployment

```text
Streamlit
↓
Logging
↓
Docker
↓
Deploy
```

---

## 23. Final End-to-End Flow

```text
DOCUMENT SIDE

Files
 ↓
Load
 ↓
Clean
 ↓
Parent Documents
 ↓
Child Chunks
 ↓
 ┌─────────────────────┐
 │                     │
 ▼                     ▼
Embeddings           BM25
 │                     │
 ▼                     ▼
Vector DB          Keyword Index


QUERY SIDE

User Question
 ↓
Conversation Memory
 ↓
Query Rewrite
 ↓
Multi-Query Generation
 ↓
 ┌────────────────────────────┐
 │                            │
 ▼                            ▼
Vector Search             BM25 Search
 │                            │
 └─────────────┬──────────────┘
               ▼
          Rank Fusion
               ↓
       Child Chunk Results
               ↓
        Parent Retrieval
               ↓
    Contextual Compression
               ↓
        Final Evidence
               ↓
          Prompt Builder
               ↓
              LLM
               ↓
      Structured Response
               ↓
     Save Conversation
               ↓
             User
```

---

## 24. Portfolio / Interview Positioning

Project title:

**Advanced RAG AI Research & Document Intelligence Assistant**

Suggested one-line description:

> Built a modular RAG system combining parent-child retrieval, hybrid dense + BM25 search, multi-query expansion, contextual compression, conversational memory, structured responses, and retrieval evaluation for grounded document question answering.

Interview topics this architecture should prepare you to explain:

- Why chunking matters
- Why child chunks and parent documents are separated
- Dense vs keyword retrieval
- Why hybrid search improves recall/precision trade-offs
- How MultiQueryRetriever improves retrieval recall
- How contextual compression reduces irrelevant context and token usage
- How conversational memory supports follow-up questions
- How structured output improves application reliability
- How to evaluate retrieval independently from generation
- How to identify where a RAG system is failing

---

## 25. Implementation Rule

Build and test one layer at a time.

Do not start with the full advanced pipeline.

Recommended sequence:

```text
Basic RAG
 → Parent Retrieval
 → Hybrid Search
 → Multi-Query
 → Compression
 → Memory
 → Structured Output
 → Evaluation
 → UI
 → Deployment
```

Every phase should have a working test before adding the next retrieval capability.

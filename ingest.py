"""
Phase 6: Conversational Memory
--------------------------------

Pipeline:

Conversation History
        ↓
Query Rewriting
        ↓
Multi-Query Generation
        ↓
Vector Search + BM25
        ↓
RRF Fusion
        ↓
Cross-Query Fusion
        ↓
Parent Retrieval
        ↓
Contextual Compression
        ↓
Final LLM Answer
        ↓
Save Conversation
"""

from pathlib import Path
import json
import re
from collections import defaultdict

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rank_bm25 import BM25Okapi


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHROMA_DIR = BASE_DIR / "chroma_db"
PARENT_STORE_PATH = BASE_DIR / "parent_store.json"
CHILD_STORE_PATH = BASE_DIR / "child_store.json"

# IMPORTANT:
# Keep this exactly the same as your working Phase 5
# Chroma collection name.
COLLECTION_NAME = "phase5_contextual_compression_rag"

# Retrieval settings
VECTOR_K = 6
BM25_K = 6

# Number of hybrid child results kept per generated query
PER_QUERY_CHILD_K = 4

# Number of children after cross-query fusion
FINAL_CHILD_K = 5

# RRF constant
RRF_K = 60

# Number of alternative search queries
NUM_GENERATED_QUERIES = 3

# Maximum messages kept in conversation memory
MAX_HISTORY_MESSAGES = 10


# ============================================================
# BASIC UTILITIES
# ============================================================

def tokenize(text: str) -> list[str]:
    """
    Convert text into simple tokens for BM25.

    Example:
        "What is Q2?" -> ["what", "is", "q2"]
    """

    return re.findall(
        r"\b[a-zA-Z0-9_]+\b",
        text.lower()
    )


def get_child_id(doc: Document) -> str:
    """
    Return the unique child ID.
    """

    child_id = doc.metadata.get("child_id")

    if child_id:
        return child_id

    # Fallback ID
    return (
        f"{doc.metadata.get('source', '')}:"
        f"{doc.metadata.get('page', '')}:"
        f"{doc.page_content[:80]}"
    )


# ============================================================
# VECTOR DATABASE
# ============================================================

def get_vectorstore() -> Chroma:
    """
    Connect to the existing Phase 5 Chroma database.

    No new ingestion is required.
    """

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


# ============================================================
# LOAD DOCUMENT STORES
# ============================================================

def load_json_store(path: Path) -> dict:
    """
    Load a JSON-based store.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_children() -> list[Document]:
    """
    Load child chunks for BM25.
    """

    data = load_json_store(
        CHILD_STORE_PATH
    )

    children = []

    for item in data.values():

        children.append(
            Document(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )
        )

    return children


# ============================================================
# CONVERSATION MEMORY
# ============================================================

MEMORY_FILE = (
    BASE_DIR / "conversation_memory.json"
)


def load_memory_store() -> dict:
    """
    Load all sessions.

    Example:

    {
        "session1": [
            {
                "role": "user",
                "content": "What is RAG?"
            },
            {
                "role": "assistant",
                "content": "RAG..."
            }
        ]
    }
    """

    if not MEMORY_FILE.exists():
        return {}

    return json.loads(
        MEMORY_FILE.read_text(
            encoding="utf-8"
        )
    )


def save_memory_store(
    memory_store: dict
) -> None:
    """
    Persist conversation memory.
    """

    MEMORY_FILE.write_text(
        json.dumps(
            memory_store,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def get_session_history(
    session_id: str
) -> list[dict]:
    """
    Get history for one session.
    """

    memory_store = load_memory_store()

    return memory_store.get(
        session_id,
        []
    )


def save_conversation_turn(
    session_id: str,
    user_question: str,
    assistant_answer: str,
) -> None:
    """
    Save a user + assistant turn.
    """

    memory_store = load_memory_store()

    history = memory_store.setdefault(
        session_id,
        []
    )

    history.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": assistant_answer,
        }
    )

    # Keep only recent messages
    memory_store[session_id] = (
        history[-MAX_HISTORY_MESSAGES:]
    )

    save_memory_store(
        memory_store
    )


def clear_session(
    session_id: str
) -> None:
    """
    Remove one session.
    """

    memory_store = load_memory_store()

    if session_id in memory_store:
        del memory_store[session_id]

    save_memory_store(
        memory_store
    )


def format_history(
    history: list[dict]
) -> str:
    """
    Convert memory into prompt text.
    """

    if not history:
        return "No previous conversation."

    lines = []

    for message in history[
        -MAX_HISTORY_MESSAGES:
    ]:

        role = message["role"].upper()

        lines.append(
            f"{role}: {message['content']}"
        )

    return "\n".join(lines)


# ============================================================
# PHASE 6 — QUERY REWRITING
# ============================================================

def rewrite_query(
    question: str,
    history: list[dict],
) -> str:
    """
    Rewrite an ambiguous follow-up into a
    standalone retrieval query.

    Example:

    Previous:
        User: What is supervised learning?

    New:
        How is the second one different?

    Rewritten:
        How is unsupervised learning different
        from supervised learning?
    """

    # No history means nothing to rewrite
    if not history:
        return question

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a query rewriting assistant for a
document retrieval system.

Rewrite the user's latest question into one
standalone retrieval query.

Rules:

1. Use conversation history to understand
   references such as:
   - it
   - this
   - that
   - the second one
   - the previous model
   - the above concept

2. Preserve the user's intent.

3. Include important technical terms that
   are known from the conversation.

4. Do not answer the question.

5. Do not add information that is not
   supported by the conversation.

6. Return ONLY the rewritten query.
""",
            ),
            (
                "human",
                """
Conversation history:

{history}

Latest user question:

{question}

Standalone retrieval query:
""",
            ),
        ]
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    rewritten = chain.invoke(
        {
            "history": format_history(
                history
            ),
            "question": question,
        }
    ).strip()

    if not rewritten:
        return question

    return rewritten


# ============================================================
# MULTI-QUERY GENERATION
# ============================================================

def generate_multi_queries(
    question: str,
) -> list[str]:
    """
    Generate alternative search formulations.

    Original question is always retained.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a search-query generation assistant.

Generate exactly 3 alternative search
queries for the given question.

Rules:

1. Preserve the original intent.
2. Use different wording.
3. Keep important technical terms.
4. Do not answer the question.
5. Return only one query per line.
6. Do not add numbering or explanations.
""",
            ),
            (
                "human",
                "{question}"
            ),
        ]
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    raw = chain.invoke(
        {
            "question": question
        }
    )

    generated_queries = []

    for line in raw.splitlines():

        query = line.strip()

        # Remove accidental:
        # 1.
        # 2)
        # -
        # *
        query = re.sub(
            r"^\s*[\d\-\*\.\)]\s*",
            "",
            query
        )

        if (
            query
            and query not in generated_queries
        ):
            generated_queries.append(
                query
            )

    generated_queries = (
        generated_queries[
            :NUM_GENERATED_QUERIES
        ]
    )

    queries = [question]

    for query in generated_queries:

        if query not in queries:
            queries.append(query)

    return queries


# ============================================================
# VECTOR SEARCH
# ============================================================

def vector_search(
    question: str,
    k: int = VECTOR_K,
):
    """
    Semantic retrieval.

    Lower distance = better match.
    """

    vectorstore = get_vectorstore()

    return (
        vectorstore
        .similarity_search_with_score(
            question,
            k=k
        )
    )


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    question: str,
    k: int = BM25_K,
):
    """
    Keyword retrieval using BM25.

    Important:
    documents with BM25 <= 0 are discarded.
    """

    children = load_children()

    if not children:
        return []

    corpus = [
        tokenize(
            child.page_content
        )
        for child in children
    ]

    query_tokens = tokenize(
        question
    )

    if not query_tokens:
        return []

    bm25 = BM25Okapi(
        corpus
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(children)),
        key=lambda index:
            scores[index],
        reverse=True,
    )

    results = []

    for index in ranked_indices:

        score = float(
            scores[index]
        )

        # Phase 3 refinement
        # Ignore zero-overlap documents
        if score <= 0:
            continue

        results.append(
            (
                children[index],
                score
            )
        )

        if len(results) >= k:
            break

    return results


# ============================================================
# RRF FUSION
# ============================================================

def rrf_fusion(
    vector_docs,
    bm25_docs,
):
    """
    Combine Vector + BM25 rankings.

    We use ranking positions, not raw scores.
    """

    fused_scores = defaultdict(float)

    documents = {}

    contributions = defaultdict(dict)

    # -----------------------------
    # Vector results
    # -----------------------------

    for rank, doc in enumerate(
        vector_docs,
        start=1,
    ):

        doc_id = get_child_id(
            doc
        )

        documents[doc_id] = doc

        contribution = (
            1 / (RRF_K + rank)
        )

        fused_scores[doc_id] += (
            contribution
        )

        contributions[
            doc_id
        ]["vector"] = contribution

    # -----------------------------
    # BM25 results
    # -----------------------------

    for rank, doc in enumerate(
        bm25_docs,
        start=1,
    ):

        doc_id = get_child_id(
            doc
        )

        documents[doc_id] = doc

        contribution = (
            1 / (RRF_K + rank)
        )

        fused_scores[doc_id] += (
            contribution
        )

        contributions[
            doc_id
        ]["bm25"] = contribution

    # -----------------------------
    # Sort
    # -----------------------------

    ranked = sorted(
        fused_scores.items(),
        key=lambda item:
            item[1],
        reverse=True,
    )

    return [
        (
            documents[doc_id],
            score,
            contributions[doc_id]
        )
        for doc_id, score in ranked
    ]


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    question: str,
):
    """
    Vector + BM25 + RRF.
    """

    vector_results = vector_search(
        question
    )

    bm25_results = bm25_search(
        question
    )

    fused_results = rrf_fusion(
        [
            doc
            for doc, _ in vector_results
        ],
        [
            doc
            for doc, _ in bm25_results
        ],
    )

    return (
        vector_results,
        bm25_results,
        fused_results[
            :PER_QUERY_CHILD_K
        ],
    )


# ============================================================
# CROSS-QUERY FUSION
# ============================================================

def cross_query_fusion(
    per_query_results,
):
    """
    Combine results from multiple
    generated queries.

    A document appearing in:
        Q1
        Q2
        Q3
        Q4

    receives multiple RRF contributions.
    """

    fused_scores = defaultdict(float)

    documents = {}

    appearances = defaultdict(set)

    for query_index, results in enumerate(
        per_query_results,
        start=1,
    ):

        query_name = (
            f"Q{query_index}"
        )

        for rank, (
            doc,
            _score,
            _contributions,
        ) in enumerate(
            results,
            start=1,
        ):

            doc_id = get_child_id(
                doc
            )

            documents[doc_id] = doc

            contribution = (
                1 / (RRF_K + rank)
            )

            fused_scores[doc_id] += (
                contribution
            )

            appearances[
                doc_id
            ].add(query_name)

    ranked = sorted(
        fused_scores.items(),
        key=lambda item:
            item[1],
        reverse=True,
    )

    return [
        (
            documents[doc_id],
            score,
            sorted(
                appearances[doc_id]
            ),
        )
        for doc_id, score in ranked
    ]


# ============================================================
# PARENT RETRIEVAL
# ============================================================

def retrieve_parents(
    retrieval_question: str,
):
    """
    Multi-query → Hybrid → Cross-query
    fusion → Parent retrieval.
    """

    queries = generate_multi_queries(
        retrieval_question
    )

    per_query_results = []

    for query in queries:

        (
            _vector_results,
            _bm25_results,
            fused_results,
        ) = hybrid_search(
            query
        )

        per_query_results.append(
            fused_results
        )

    # Cross-query fusion
    final_children = (
        cross_query_fusion(
            per_query_results
        )
    )

    final_children = (
        final_children[
            :FINAL_CHILD_K
        ]
    )

    # Load parent store
    parent_store = load_json_store(
        PARENT_STORE_PATH
    )

    parents = []

    seen_parent_ids = set()

    for (
        child,
        _score,
        _appearances,
    ) in final_children:

        parent_id = (
            child.metadata.get(
                "parent_id"
            )
        )

        if not parent_id:
            continue

        if parent_id in seen_parent_ids:
            continue

        parent_data = (
            parent_store.get(
                parent_id
            )
        )

        if not parent_data:
            continue

        parent = Document(
            page_content=parent_data[
                "page_content"
            ],
            metadata=parent_data[
                "metadata"
            ],
        )

        parents.append(
            parent
        )

        seen_parent_ids.add(
            parent_id
        )

    return (
        queries,
        per_query_results,
        final_children,
        parents,
    )


# ============================================================
# CONTEXTUAL COMPRESSION
# ============================================================

def compress_documents(
    question: str,
    parent_documents: list[Document],
):
    """
    Extract only query-relevant portions
    from parent documents.
    """

    if not parent_documents:
        return []

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    compressor = (
        LLMChainExtractor.from_llm(
            llm
        )
    )

    compressed_documents = (
        compressor.compress_documents(
            parent_documents,
            question,
        )
    )

    # Remove empty results
    compressed_documents = [
        doc
        for doc in compressed_documents
        if doc.page_content.strip()
    ]

    return compressed_documents


# ============================================================
# COMPRESSION METRICS
# ============================================================

def calculate_compression(
    original_documents,
    compressed_documents,
):
    before = sum(
        len(doc.page_content)
        for doc in original_documents
    )

    after = sum(
        len(doc.page_content)
        for doc in compressed_documents
    )

    if before == 0:
        return (
            before,
            after,
            0.0
        )

    reduction = (
        1 - (
            after / before
        )
    ) * 100

    return (
        before,
        after,
        reduction
    )


# ============================================================
# FINAL ANSWER
# ============================================================

def generate_answer(
    question: str,
    context: str,
    history: list[dict],
) -> str:
    """
    Generate final answer using:
        - conversation history
        - compressed document context
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a document question-answering assistant.

Answer ONLY from the provided compressed
document context.

Use conversation history only to understand
the current conversation and references.

Do not use outside knowledge.

If the document context does not contain
the answer, say that the answer was not
found in the documents.

Mention source names when possible.

Conversation history:
{history}

Compressed document context:
{context}
""",
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )

    response = (
        prompt
        | llm
    ).invoke(
        {
            "history": format_history(
                history
            ),
            "context": context,
            "question": question,
        }
    )

    return response.content


# ============================================================
# FORMAT CONTEXT
# ============================================================

def format_context(
    documents: list[Document],
):
    if not documents:
        return (
            "No relevant compressed "
            "context was found."
        )

    sections = []

    for index, doc in enumerate(
        documents,
        start=1,
    ):

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        sections.append(
            f"""
[Compressed Source {index}: {source}]

{doc.page_content}
"""
        )

    return "\n\n---\n\n".join(
        sections
    )


# ============================================================
# DEBUG OUTPUT
# ============================================================

def print_debug(
    session_id,
    original_question,
    history,
    rewritten_question,
    queries,
    final_children,
    parents,
    compressed_documents,
):
    print("\n" + "=" * 90)
    print(
        "PHASE 6 — CONVERSATIONAL MEMORY"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Session
    # --------------------------------------------------------

    print("\nSESSION")
    print("-" * 90)

    print(
        session_id
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    print("\nCONVERSATION HISTORY")
    print("-" * 90)

    if not history:
        print(
            "No previous conversation."
        )

    else:

        for message in history[
            -MAX_HISTORY_MESSAGES:
        ]:

            print(
                f"{message['role'].upper()}: "
                f"{message['content']}"
            )

    # --------------------------------------------------------
    # Original question
    # --------------------------------------------------------

    print("\nORIGINAL QUESTION")
    print("-" * 90)

    print(
        original_question
    )

    # --------------------------------------------------------
    # Rewritten question
    # --------------------------------------------------------

    print("\nREWRITTEN RETRIEVAL QUERY")
    print("-" * 90)

    print(
        rewritten_question
    )

    # --------------------------------------------------------
    # Generated queries
    # --------------------------------------------------------

    print("\nGENERATED SEARCH QUERIES")
    print("-" * 90)

    for index, query in enumerate(
        queries,
        start=1,
    ):

        print(
            f"Q{index}: {query}"
        )

    # --------------------------------------------------------
    # Cross-query results
    # --------------------------------------------------------

    print("\nCROSS-QUERY TOP CHILDREN")
    print("-" * 90)

    for rank, (
        doc,
        score,
        appearances,
    ) in enumerate(
        final_children,
        start=1,
    ):

        print(
            f"Rank {rank} | "
            f"RRF={score:.6f} | "
            f"appeared_in={appearances}"
        )

        print(
            f"  child="
            f"{doc.metadata.get('child_id')}"
        )

        print(
            f"  parent="
            f"{doc.metadata.get('parent_id')}"
        )

    # --------------------------------------------------------
    # Parent documents
    # --------------------------------------------------------

    print("\nPARENT CONTEXT")
    print("-" * 90)

    for index, parent in enumerate(
        parents,
        start=1,
    ):

        print(
            f"Parent {index} | "
            f"chars="
            f"{len(parent.page_content)} | "
            f"source="
            f"{parent.metadata.get('source', 'Unknown')}"
        )

    # --------------------------------------------------------
    # Compression
    # --------------------------------------------------------

    before, after, reduction = (
        calculate_compression(
            parents,
            compressed_documents,
        )
    )

    print("\nCOMPRESSION")
    print("-" * 90)

    print(
        f"Before    : {before} characters"
    )

    print(
        f"After     : {after} characters"
    )

    print(
        f"Reduction : {reduction:.2f}%"
    )


# ============================================================
# ONE CONVERSATIONAL TURN
# ============================================================

def ask(
    question: str,
    session_id: str,
):
    """
    Execute one conversational turn.
    """

    # --------------------------------------------------------
    # 1. Load memory
    # --------------------------------------------------------

    history = get_session_history(
        session_id
    )

    # --------------------------------------------------------
    # 2. Rewrite follow-up
    # --------------------------------------------------------

    rewritten_question = rewrite_query(
        question,
        history,
    )

    # --------------------------------------------------------
    # 3. Advanced retrieval
    # --------------------------------------------------------

    (
        queries,
        _per_query_results,
        final_children,
        parents,
    ) = retrieve_parents(
        rewritten_question
    )

    # --------------------------------------------------------
    # 4. Compression
    # --------------------------------------------------------

    compressed_documents = (
        compress_documents(
            rewritten_question,
            parents,
        )
    )

    context = format_context(
        compressed_documents
    )

    # --------------------------------------------------------
    # 5. Debug
    # --------------------------------------------------------

    print_debug(
        session_id,
        question,
        history,
        rewritten_question,
        queries,
        final_children,
        parents,
        compressed_documents,
    )

    # --------------------------------------------------------
    # 6. Show context
    # --------------------------------------------------------

    print("\n" + "=" * 90)

    print(
        "COMPRESSED CONTEXT SENT TO LLM"
    )

    print("=" * 90)

    print(
        context
    )

    print(
        f"\nFinal context characters: "
        f"{len(context)}"
    )

    # --------------------------------------------------------
    # 7. Final answer
    # --------------------------------------------------------

    answer = generate_answer(
        question,
        context,
        history,
    )

    print("\n" + "=" * 90)
    print(
        "FINAL ANSWER"
    )
    print("=" * 90)

    print(
        answer
    )

    # --------------------------------------------------------
    # 8. Save memory
    # --------------------------------------------------------

    save_conversation_turn(
        session_id,
        question,
        answer,
    )

    print(
        "\nConversation turn saved."
    )

    return answer


# ============================================================
# MAIN CHAT LOOP
# ============================================================

if __name__ == "__main__":

    required_paths = [
        CHROMA_DIR,
        PARENT_STORE_PATH,
        CHILD_STORE_PATH,
    ]

    if not all(
        path.exists()
        for path in required_paths
    ):

        raise SystemExit(
            "Phase 5 index not found. "
            "Use your existing working index."
        )

    print("\n" + "=" * 90)

    print(
        "PHASE 6 — CONVERSATIONAL RAG ASSISTANT"
    )

    print("=" * 90)

    session_id = input(
        "\nSession ID [default]: "
    ).strip()

    if not session_id:
        session_id = "default"

    print(
        "\nCommands:"
    )

    print(
        "  /history  -> show conversation history"
    )

    print(
        "  /clear    -> clear current session"
    )

    print(
        "  /exit     -> exit"
    )

    # --------------------------------------------------------
    # Chat loop
    # --------------------------------------------------------

    while True:

        question = input(
            "\nUser: "
        ).strip()

        if not question:
            continue

        # -----------------------------
        # Exit
        # -----------------------------

        if question.lower() == "/exit":

            print(
                "Goodbye."
            )

            break

        # -----------------------------
        # Clear
        # -----------------------------

        if question.lower() == "/clear":

            clear_session(
                session_id
            )

            print(
                f"Session '{session_id}' cleared."
            )

            continue

        # -----------------------------
        # History
        # -----------------------------

        if question.lower() == "/history":

            history = get_session_history(
                session_id
            )

            if not history:

                print(
                    "No conversation history."
                )

            else:

                print(
                    "\nConversation history:"
                )

                for message in history:

                    print(
                        f"{message['role'].upper()}: "
                        f"{message['content']}"
                    )

            continue

        # -----------------------------
        # Normal question
        # -----------------------------

        ask(
            question,
            session_id,
        )
from pathlib import Path
import json
import re
from collections import defaultdict
from typing import List, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
PARENT_STORE_PATH = BASE_DIR / "parent_store.json"
CHILD_STORE_PATH = BASE_DIR / "child_store.json"
MEMORY_FILE = BASE_DIR / "conversation_memory.json"

COLLECTION_NAME = "phase5_contextual_compression_rag"

VECTOR_K = 6
BM25_K = 6
PER_QUERY_CHILD_K = 4
FINAL_CHILD_K = 5
RRF_K = 60
NUM_GENERATED_QUERIES = 3
MAX_HISTORY_MESSAGES = 10


class ResearchResponse(BaseModel):
    answer: str = Field(
        description="Answer based only on retrieved document context."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="High if directly supported, medium if partially inferred, low if weak or missing."
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Source document names actually supporting the answer."
    )
    key_quotes: List[str] = Field(
        default_factory=list,
        description="Short supporting excerpts from retrieved context."
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Useful next questions relevant to the topic."
    )


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


def get_child_id(doc: Document) -> str:
    cid = doc.metadata.get("child_id")
    if cid:
        return cid
    return (
        f"{doc.metadata.get('source', '')}:"
        f"{doc.metadata.get('page', '')}:"
        f"{doc.page_content[:80]}"
    )


def load_json_store(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_children() -> list[Document]:
    data = load_json_store(CHILD_STORE_PATH)
    return [
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )
        for item in data.values()
    ]


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=str(CHROMA_DIR),
    )


def load_memory_store() -> dict:
    if not MEMORY_FILE.exists():
        return {}
    return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))


def save_memory_store(store: dict) -> None:
    MEMORY_FILE.write_text(
        json.dumps(store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_session_history(session_id: str) -> list[dict]:
    return load_memory_store().get(session_id, [])


def save_conversation_turn(
    session_id: str,
    user_question: str,
    assistant_answer: str,
) -> None:
    store = load_memory_store()
    history = store.setdefault(session_id, [])

    history.append({"role": "user", "content": user_question})
    history.append({"role": "assistant", "content": assistant_answer})

    store[session_id] = history[-MAX_HISTORY_MESSAGES:]
    save_memory_store(store)


def clear_session(session_id: str) -> None:
    store = load_memory_store()
    store.pop(session_id, None)
    save_memory_store(store)


def format_history(history: list[dict]) -> str:
    if not history:
        return "No previous conversation."
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in history[-MAX_HISTORY_MESSAGES:]
    )


def rewrite_query(question: str, history: list[dict]) -> str:
    if not history:
        return question

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Rewrite the latest conversational question into one standalone retrieval query.
Use history to resolve omitted subjects, pronouns, and topic continuity.
Do not answer. Do not add unsupported information.
Return only the rewritten query.""",
            ),
            (
                "human",
                """Conversation history:
{history}

Latest question:
{question}

Standalone retrieval query:""",
            ),
        ]
    )

    result = (prompt | llm | StrOutputParser()).invoke(
        {
            "history": format_history(history),
            "question": question,
        }
    ).strip()

    return result or question


def generate_multi_queries(question: str) -> list[str]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Generate exactly 3 alternative search queries.
Preserve intent, vary wording, keep important technical terms.
Do not answer. Return one query per line with no numbering.""",
            ),
            ("human", "{question}"),
        ]
    )

    raw = (prompt | llm | StrOutputParser()).invoke({"question": question})

    generated = []
    for line in raw.splitlines():
        query = re.sub(r"^\s*[\d\-\*\.\)]\s*", "", line.strip())
        if query and query not in generated:
            generated.append(query)

    queries = [question]
    for query in generated[:NUM_GENERATED_QUERIES]:
        if query not in queries:
            queries.append(query)

    return queries


def vector_search(question: str, k: int = VECTOR_K):
    return get_vectorstore().similarity_search_with_score(question, k=k)


def bm25_search(question: str, k: int = BM25_K):
    children = load_children()
    if not children:
        return []

    corpus = [tokenize(doc.page_content) for doc in children]
    query_tokens = tokenize(question)
    if not query_tokens:
        return []

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(children)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        results.append((children[index], score))
        if len(results) >= k:
            break

    return results


def rrf_fusion(vector_docs, bm25_docs):
    fused_scores = defaultdict(float)
    documents = {}
    contributions = defaultdict(dict)

    for method_name, ranked_docs in (
        ("vector", vector_docs),
        ("bm25", bm25_docs),
    ):
        for rank, doc in enumerate(ranked_docs, start=1):
            doc_id = get_child_id(doc)
            documents[doc_id] = doc
            contribution = 1 / (RRF_K + rank)
            fused_scores[doc_id] += contribution
            contributions[doc_id][method_name] = contribution

    ranked = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (documents[doc_id], score, contributions[doc_id])
        for doc_id, score in ranked
    ]


def hybrid_search(question: str):
    vector_results = vector_search(question)
    bm25_results = bm25_search(question)

    fused_results = rrf_fusion(
        [doc for doc, _ in vector_results],
        [doc for doc, _ in bm25_results],
    )

    return fused_results[:PER_QUERY_CHILD_K]


def cross_query_fusion(per_query_results):
    fused_scores = defaultdict(float)
    documents = {}
    appearances = defaultdict(set)

    for query_index, results in enumerate(per_query_results, start=1):
        query_name = f"Q{query_index}"

        for rank, (doc, _score, _contributions) in enumerate(
            results, start=1
        ):
            doc_id = get_child_id(doc)
            documents[doc_id] = doc

            contribution = 1 / (RRF_K + rank)
            fused_scores[doc_id] += contribution
            appearances[doc_id].add(query_name)

    ranked = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (documents[doc_id], score, sorted(appearances[doc_id]))
        for doc_id, score in ranked
    ]


def retrieve_parents(retrieval_question: str):
    queries = generate_multi_queries(retrieval_question)

    per_query_results = [
        hybrid_search(query)
        for query in queries
    ]

    final_children = cross_query_fusion(per_query_results)[:FINAL_CHILD_K]

    parent_store = load_json_store(PARENT_STORE_PATH)

    parents = []
    seen_parent_ids = set()

    for child, _score, _appearances in final_children:
        parent_id = child.metadata.get("parent_id")

        if not parent_id or parent_id in seen_parent_ids:
            continue

        parent_data = parent_store.get(parent_id)
        if not parent_data:
            continue

        parents.append(
            Document(
                page_content=parent_data["page_content"],
                metadata=parent_data["metadata"],
            )
        )

        seen_parent_ids.add(parent_id)

    return queries, final_children, parents


def compress_documents(
    question: str,
    parents: list[Document],
):
    if not parents:
        return []

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)

    compressed = compressor.compress_documents(parents, question)

    return [
        doc
        for doc in compressed
        if doc.page_content.strip()
    ]


def format_context(documents: list[Document]) -> str:
    if not documents:
        return "No relevant compressed context was found."

    sections = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown")
        sections.append(
            f"[Source {index}: {source}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(sections)


def generate_structured_answer(
    question: str,
    context: str,
    history: list[dict],
) -> ResearchResponse:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    structured_llm = llm.with_structured_output(ResearchResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a document-grounded research assistant.

Answer only from the compressed document context.

Conversation history may help understand the dialogue,
but it is not an independent factual source.

Rules:
1. If the answer is not in context, say so and set confidence="low".
2. confidence="high" if directly supported.
3. confidence="medium" if a small inference is needed.
4. sources must contain only filenames that support the answer.
5. key_quotes must contain short supporting excerpts from context, max 3.
6. follow_up_questions should contain 2-3 useful relevant next questions.
7. Do not use outside knowledge.

Conversation history:
{history}

Compressed document context:
{context}
""",
            ),
            ("human", "{question}"),
        ]
    )

    return (prompt | structured_llm).invoke(
        {
            "history": format_history(history),
            "context": context,
            "question": question,
        }
    )


def print_structured_response(response: ResearchResponse):
    print("\n" + "=" * 90)
    print("STRUCTURED RESPONSE")
    print("=" * 90)

    print("\nANSWER")
    print("-" * 90)
    print(response.answer)

    print("\nCONFIDENCE")
    print("-" * 90)
    print(response.confidence.upper())

    print("\nSOURCES")
    print("-" * 90)
    if response.sources:
        for source in response.sources:
            print(f"- {source}")
    else:
        print("No sources returned.")

    print("\nKEY QUOTES")
    print("-" * 90)
    if response.key_quotes:
        for quote in response.key_quotes:
            print(f'- "{quote}"')
    else:
        print("No supporting quotes returned.")

    print("\nFOLLOW-UP QUESTIONS")
    print("-" * 90)
    for q in response.follow_up_questions:
        print(f"- {q}")


def ask(question: str, session_id: str) -> ResearchResponse:
    history = get_session_history(session_id)

    rewritten_question = rewrite_query(
        question,
        history,
    )

    queries, final_children, parents = retrieve_parents(
        rewritten_question
    )

    compressed_documents = compress_documents(
        rewritten_question,
        parents,
    )

    context = format_context(
        compressed_documents
    )

    print("\n" + "=" * 90)
    print("PHASE 7 — STRUCTURED CONVERSATIONAL RAG")
    print("=" * 90)

    print("\nORIGINAL QUESTION")
    print("-" * 90)
    print(question)

    print("\nREWRITTEN QUERY")
    print("-" * 90)
    print(rewritten_question)

    print("\nGENERATED SEARCH QUERIES")
    print("-" * 90)
    for index, query in enumerate(queries, start=1):
        print(f"Q{index}: {query}")

    print("\nFINAL CHILDREN")
    print("-" * 90)
    for rank, (doc, score, appearances) in enumerate(
        final_children,
        start=1,
    ):
        print(
            f"Rank {rank} | RRF={score:.6f} | "
            f"queries={appearances}"
        )
        print(f"child={doc.metadata.get('child_id')}")

    print("\nCOMPRESSED CONTEXT")
    print("-" * 90)
    print(context)

    response = generate_structured_answer(
        question,
        context,
        history,
    )

    print_structured_response(response)

    # Only save the natural answer text in memory.
    save_conversation_turn(
        session_id,
        question,
        response.answer,
    )

    return response


if __name__ == "__main__":
    required_paths = [
        CHROMA_DIR,
        PARENT_STORE_PATH,
        CHILD_STORE_PATH,
    ]

    if not all(path.exists() for path in required_paths):
        raise SystemExit(
            "Existing RAG index not found. "
            "Copy this rag.py into your working Phase 6 project "
            "or point the paths/collection name to your existing index."
        )

    print("\n" + "=" * 90)
    print("PHASE 7 — STRUCTURED RAG ASSISTANT")
    print("=" * 90)

    session_id = input("\nSession ID [default]: ").strip()
    if not session_id:
        session_id = "default"

    print("\nCommands: history, clear, exit")

    while True:
        question = input("\nUser: ").strip()

        if not question:
            continue

        command = question.lower().strip()

        if command in {"exit", "/exit", "quit", "/quit"}:
            print("Goodbye.")
            break

        if command in {"clear", "/clear"}:
            clear_session(session_id)
            print(f"Session '{session_id}' cleared.")
            continue

        if command in {"history", "/history"}:
            history = get_session_history(session_id)

            if not history:
                print("No conversation history.")
            else:
                print("\nConversation history:")
                for message in history:
                    print(
                        f"{message['role'].upper()}: "
                        f"{message['content']}"
                    )
            continue

        ask(question, session_id)

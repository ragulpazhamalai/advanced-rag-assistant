"""
Phase 8 — Evaluation for the Phase 7 RAG pipeline.

This file imports your existing `rag.py`.

It evaluates:
- Retrieval Hit@K
- MRR
- Keyword coverage
- Source correctness
- Refusal correctness
- LLM judged correctness
- LLM judged groundedness
- LLM judged context relevance
"""

from pathlib import Path
import json
from statistics import mean
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import rag


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
EVAL_SET_PATH = BASE_DIR / "evaluation_set.json"


# ============================================================
# EVALUATION SCHEMAS
# ============================================================

class JudgeResult(BaseModel):
    correctness: int = Field(
        ge=1,
        le=5,
        description="How correct the answer is compared with expected answer."
    )

    groundedness: int = Field(
        ge=1,
        le=5,
        description="How strongly the answer is supported by retrieved context."
    )

    context_relevance: int = Field(
        ge=1,
        le=5,
        description="How relevant the retrieved context is to the question."
    )

    verdict: Literal[
        "pass",
        "partial",
        "fail"
    ]

    explanation: str


# ============================================================
# LOAD TEST SET
# ============================================================

def load_evaluation_set():
    return json.loads(
        EVAL_SET_PATH.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# SIMPLE METRICS
# ============================================================

def keyword_coverage(
    answer: str,
    expected_keywords: list[str],
) -> float:
    """
    Percentage of expected keywords appearing in answer.
    """

    if not expected_keywords:
        return 1.0

    answer_lower = answer.lower()

    matches = sum(
        1
        for keyword in expected_keywords
        if keyword.lower() in answer_lower
    )

    return (
        matches
        / len(expected_keywords)
    )


def source_correct(
    sources: list[str],
    expected_source: str,
) -> bool:
    if not expected_source:
        return len(sources) == 0

    return any(
        expected_source.lower()
        == source.lower()
        for source in sources
    )


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def retrieval_metrics(
    question: str,
    expected_source: str,
):
    """
    Run the retrieval stack and calculate source-based Hit@K and MRR.

    Since your current test data uses one PDF, source is used as the
    relevance label.

    Later we can upgrade this to expected child IDs / parent IDs.
    """

    rewritten = question

    queries, final_children, parents = (
        rag.retrieve_parents(
            rewritten
        )
    )

    if not expected_source:
        return {
            "hit_at_k": None,
            "mrr": None,
            "retrieved_children": len(
                final_children
            ),
        }

    hit = False
    reciprocal_rank = 0.0

    for rank, (
        doc,
        _score,
        _appearances,
    ) in enumerate(
        final_children,
        start=1,
    ):
        source = doc.metadata.get(
            "source",
            ""
        )

        if source.lower() == (
            expected_source.lower()
        ):
            hit = True

            if reciprocal_rank == 0:
                reciprocal_rank = (
                    1 / rank
                )

    return {
        "hit_at_k": int(hit),
        "mrr": reciprocal_rank,
        "retrieved_children": len(
            final_children
        ),
    }


# ============================================================
# LLM JUDGE
# ============================================================

def judge_answer(
    question: str,
    expected_answer: str,
    actual_answer: str,
    context: str,
    should_be_answerable: bool,
):
    """
    LLM-as-judge evaluation.

    This does not replace deterministic metrics.
    It complements them.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    structured_llm = (
        llm.with_structured_output(
            JudgeResult
        )
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are evaluating a RAG system.

Score each dimension from 1 to 5.

Correctness:
5 = answer matches expected answer very well
1 = incorrect or contradicts expected answer

Groundedness:
5 = claims are directly supported by context
1 = answer relies on unsupported information

Context relevance:
5 = retrieved context directly answers the question
1 = context is unrelated

If should_be_answerable is false:
A correct refusal such as "not found in the documents"
should receive high correctness.

Do not reward outside knowledge.

Return a strict structured judgment.
""",
            ),
            (
                "human",
                """
Question:
{question}

Expected answer:
{expected_answer}

Should be answerable:
{should_be_answerable}

Retrieved context:
{context}

Actual answer:
{actual_answer}
""",
            ),
        ]
    )

    return (
        prompt
        | structured_llm
    ).invoke(
        {
            "question": question,
            "expected_answer": (
                expected_answer
            ),
            "should_be_answerable": (
                should_be_answerable
            ),
            "context": context,
            "actual_answer": (
                actual_answer
            ),
        }
    )


# ============================================================
# ONE TEST CASE
# ============================================================

def evaluate_case(
    case: dict,
    case_number: int,
):
    question = case[
        "question"
    ]

    expected_answer = case.get(
        "expected_answer",
        ""
    )

    expected_keywords = case.get(
        "expected_keywords",
        []
    )

    expected_source = case.get(
        "expected_source",
        ""
    )

    should_be_answerable = case.get(
        "should_be_answerable",
        True
    )

    # Use a fresh session so evaluation questions
    # do not influence each other.
    session_id = (
        f"evaluation_case_{case_number}"
    )

    rag.clear_session(
        session_id
    )

    history = []

    rewritten_question = (
        rag.rewrite_query(
            question,
            history,
        )
    )

    (
        queries,
        final_children,
        parents,
    ) = rag.retrieve_parents(
        rewritten_question
    )

    compressed_documents = (
        rag.compress_documents(
            rewritten_question,
            parents,
        )
    )

    context = rag.format_context(
        compressed_documents
    )

    response = (
        rag.generate_structured_answer(
            question,
            context,
            history,
        )
    )

    # --------------------------------------------
    # Deterministic metrics
    # --------------------------------------------

    keyword_score = (
        keyword_coverage(
            response.answer,
            expected_keywords,
        )
    )

    source_score = (
        source_correct(
            response.sources,
            expected_source,
        )
    )

    # Correct refusal for unanswerable question
    refusal_correct = None

    if not should_be_answerable:
        low_confidence = (
            response.confidence
            == "low"
        )

        no_sources = (
            len(response.sources)
            == 0
        )

        refusal_words = [
            "not found",
            "not available",
            "does not contain",
            "cannot answer",
        ]

        refused = any(
            phrase in (
                response.answer.lower()
            )
            for phrase in refusal_words
        )

        refusal_correct = (
            low_confidence
            and no_sources
            and refused
        )

    # --------------------------------------------
    # Retrieval metrics
    # --------------------------------------------

    retrieval = (
        retrieval_metrics(
            question,
            expected_source,
        )
    )

    # --------------------------------------------
    # LLM judge
    # --------------------------------------------

    judge = judge_answer(
        question=question,
        expected_answer=expected_answer,
        actual_answer=response.answer,
        context=context,
        should_be_answerable=(
            should_be_answerable
        ),
    )

    result = {
        "question": question,
        "answer": response.answer,
        "confidence": (
            response.confidence
        ),
        "sources": (
            response.sources
        ),
        "keyword_coverage": (
            round(
                keyword_score,
                3
            )
        ),
        "source_correct": (
            source_score
        ),
        "refusal_correct": (
            refusal_correct
        ),
        "hit_at_k": (
            retrieval[
                "hit_at_k"
            ]
        ),
        "mrr": (
            retrieval[
                "mrr"
            ]
        ),
        "correctness": (
            judge.correctness
        ),
        "groundedness": (
            judge.groundedness
        ),
        "context_relevance": (
            judge.context_relevance
        ),
        "verdict": (
            judge.verdict
        ),
        "judge_explanation": (
            judge.explanation
        ),
        "context_chars": len(
            context
        ),
    }

    return result


# ============================================================
# SUMMARY
# ============================================================

def safe_mean(
    values,
):
    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return mean(values)


def print_summary(results):
    print("\n")
    print("=" * 90)
    print("PHASE 8 — EVALUATION SUMMARY")
    print("=" * 90)

    avg_correctness = safe_mean(
        [
            r["correctness"]
            for r in results
        ]
    )

    avg_groundedness = safe_mean(
        [
            r["groundedness"]
            for r in results
        ]
    )

    avg_context_relevance = safe_mean(
        [
            r["context_relevance"]
            for r in results
        ]
    )

    avg_keyword = safe_mean(
        [
            r["keyword_coverage"]
            for r in results
        ]
    )

    avg_hit = safe_mean(
        [
            r["hit_at_k"]
            for r in results
        ]
    )

    avg_mrr = safe_mean(
        [
            r["mrr"]
            for r in results
        ]
    )

    print(
        f"Average correctness       : "
        f"{avg_correctness:.2f}/5"
        if avg_correctness is not None
        else "Average correctness       : N/A"
    )

    print(
        f"Average groundedness      : "
        f"{avg_groundedness:.2f}/5"
        if avg_groundedness is not None
        else "Average groundedness      : N/A"
    )

    print(
        f"Average context relevance : "
        f"{avg_context_relevance:.2f}/5"
        if avg_context_relevance is not None
        else "Average context relevance : N/A"
    )

    print(
        f"Average keyword coverage  : "
        f"{avg_keyword:.2%}"
        if avg_keyword is not None
        else "Average keyword coverage  : N/A"
    )

    print(
        f"Retrieval Hit@K           : "
        f"{avg_hit:.2%}"
        if avg_hit is not None
        else "Retrieval Hit@K           : N/A"
    )

    print(
        f"Mean Reciprocal Rank      : "
        f"{avg_mrr:.3f}"
        if avg_mrr is not None
        else "Mean Reciprocal Rank      : N/A"
    )

    refusal_results = [
        r["refusal_correct"]
        for r in results
        if r[
            "refusal_correct"
        ] is not None
    ]

    if refusal_results:
        print(
            f"Correct refusal rate      : "
            f"{mean(refusal_results):.2%}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    evaluation_set = (
        load_evaluation_set()
    )

    results = []

    for index, case in enumerate(
        evaluation_set,
        start=1,
    ):
        print("\n")
        print("=" * 90)
        print(
            f"TEST CASE {index}"
        )
        print("=" * 90)

        print(
            f"Question: "
            f"{case['question']}"
        )

        result = evaluate_case(
            case,
            index,
        )

        results.append(
            result
        )

        print(
            f"\nAnswer: "
            f"{result['answer']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']}"
        )

        print(
            f"Correctness: "
            f"{result['correctness']}/5"
        )

        print(
            f"Groundedness: "
            f"{result['groundedness']}/5"
        )

        print(
            f"Context relevance: "
            f"{result['context_relevance']}/5"
        )

        print(
            f"Keyword coverage: "
            f"{result['keyword_coverage']:.2%}"
        )

        print(
            f"Source correct: "
            f"{result['source_correct']}"
        )

        if (
            result["hit_at_k"]
            is not None
        ):
            print(
                f"Hit@K: "
                f"{result['hit_at_k']}"
            )

            print(
                f"MRR: "
                f"{result['mrr']:.3f}"
            )

        if (
            result[
                "refusal_correct"
            ]
            is not None
        ):
            print(
                f"Correct refusal: "
                f"{result['refusal_correct']}"
            )

        print(
            f"Judge verdict: "
            f"{result['verdict']}"
        )

    # Save raw results
    output_path = (
        BASE_DIR
        / "evaluation_results.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print_summary(
        results
    )

    print(
        f"\nSaved detailed results to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()

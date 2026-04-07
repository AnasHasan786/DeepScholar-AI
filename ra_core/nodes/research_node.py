"""
research_node.py
────────────────
Handles RAG-based synthesis from uploaded document context only.
Web search has been intentionally removed; all answers are grounded
exclusively in retrieved documents and the model's own knowledge.
"""

import os
from typing import TypedDict, List, Annotated, Tuple

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages

from prompts.research_prompts import (
    GATEKEEPER_SYSTEM_PROMPT,
    REJECTION_MESSAGE,
    RESEARCH_SYSTEM_PROMPT,
    RESEARCH_SYNTHESIS_PROMPT,
    CONTEXT_SUFFICIENCY_PROMPT,
)


# ──────────────────────────────────────────────
# State Schema
# ──────────────────────────────────────────────

class ResearchAssistant(TypedDict):
    messages:       Annotated[List[BaseMessage], add_messages]
    question:       str
    intent:         str
    context:        str
    response:       dict
    sources:        List[str]
    evaluation:     dict
    iteration:      int
    retrieved_docs: List[Document]


# ──────────────────────────────────────────────
# Context Helpers
# ──────────────────────────────────────────────

def build_context_from_docs(docs: List[Document]) -> Tuple[str, List[str]]:
    """Serialize retrieved documents into a single context string with source labels."""
    if not docs:
        return "", []

    context_parts: List[str] = []
    sources:       List[str] = []
    seen:          set       = set()

    for doc in docs:
        meta  = doc.metadata or {}
        fname = meta.get("source_file") or os.path.basename(meta.get("source", "Document"))
        page  = meta.get("page", 0) + 1
        label = f"{fname} (p. {page})"

        context_parts.append(f"[{label}]\n{doc.page_content.strip()}")

        if label not in seen:
            sources.append(label)
            seen.add(label)

    return "\n\n".join(context_parts), sources


def is_context_sufficient(context: str, question: str, llm) -> bool:
    """
    Ask the LLM whether the context contains enough information to answer
    the question. Falls back to a length heuristic on parse failure.
    """
    if not context or len(context.strip()) < 200:
        return False

    try:
        result = llm.invoke([
            SystemMessage(content=(
                "You are a strict binary evaluator. "
                "Reply with ONLY the single word YES or NO — no punctuation, no explanation."
            )),
            HumanMessage(content=CONTEXT_SUFFICIENCY_PROMPT.format(
                context=context[:4000],
                question=question,
            )),
        ])
        reply = result.content.strip().upper()
        if reply.startswith("YES"):
            return True
        if reply.startswith("NO"):
            return False
        return len(context.strip()) > 500
    except Exception:
        return len(context.strip()) > 500


# ──────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────

def classify_intent_node(state: ResearchAssistant, llm) -> dict:
    """
    Gatekeeper: decides whether the question is research-relevant.
    Returns early with a rejection message for off-topic queries.
    """
    question = state.get("question") or (
        state["messages"][-1].content if state["messages"] else ""
    )

    result = llm.invoke([
        SystemMessage(content=GATEKEEPER_SYSTEM_PROMPT),
        HumanMessage(content=f"QUESTION: {question}"),
    ])

    if "UNRELATED" in result.content.upper():
        return {
            "messages": [AIMessage(content=REJECTION_MESSAGE)],
            "intent":   "rejected",
            "response": {"answer": REJECTION_MESSAGE},
        }

    return {"intent": "research_verified"}


def research_node(state: ResearchAssistant, llm) -> dict:
    """
    Core synthesis node.
    1. Builds context from retrieved documents.
    2. Falls back to model knowledge when context is sparse.
    3. Synthesises a grounded, precise answer.
    """
    if state.get("intent") == "rejected":
        return state

    question = state.get("question") or (
        state["messages"][-1].content if state["messages"] else ""
    )

    # ── Build document context ──────────────────
    doc_context, doc_sources = build_context_from_docs(
        state.get("retrieved_docs", [])[:4]
    )

    context_available = bool(doc_context and is_context_sufficient(doc_context, question, llm))

    if context_available:
        context_note = "Answer using the provided document context as your primary source."
    else:
        context_note = (
            "No document context was retrieved for this question. "
            "Answer from your own knowledge and clearly state that no paper context was available."
        )

    # ── Synthesise answer ───────────────────────
    answer = llm.invoke([
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(content=RESEARCH_SYNTHESIS_PROMPT.format(
            context=doc_context or "No document context available.",
            context_note=context_note,
            question=question,
        )),
    ])

    full_context = (
        f"--- DOCUMENT CONTEXT ---\n{doc_context or 'NONE'}"
    )

    return {
        "messages": [AIMessage(content=answer.content)],
        "response": {"answer": answer.content},
        "context":  full_context,
        "sources":  doc_sources,
        "intent":   "research_synthesis",
    }
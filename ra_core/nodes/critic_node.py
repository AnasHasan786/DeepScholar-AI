"""
critic_node.py
──────────────
Audits the research response for grounding, accuracy, and completeness.
Returns a structured evaluation dict that controls the improvement loop.
"""

import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from nodes.research_node import ResearchAssistant
from prompts.critic_prompts import CRITIC_SYSTEM_PROMPT, CRITIC_REPORT_PROMPT


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

_FALLBACK_EVALUATION = {
    "relevance_score":      0,
    "grounded_score":       0,
    "completeness_score":   0,
    "hallucination_detected": True,
    "feedback": (
        "Auditor failed to produce a parseable report. "
        "Forcing re-generation as a safety measure."
    ),
    "improve": True,
}


def _parse_evaluation(text: str) -> Optional[dict]:
    """Extract and parse the JSON evaluation block from LLM output."""
    try:
        cleaned = re.sub(r"```json|```", "", text).strip()
        start, end = cleaned.find("{"), cleaned.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(cleaned[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# ──────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────

def critic_node(state: ResearchAssistant, llm) -> dict:
    """
    Audits the latest response against the document context.

    Sets evaluation["improve"] = True  → triggers the improver.
    Sets evaluation["improve"] = False → the workflow ends.

    Hard-stops the loop at iteration 3 to prevent runaway token usage.
    """
    question  = state.get("question") or (
        state["messages"][-1].content if state["messages"] else ""
    )
    context   = (state.get("context") or "").strip()
    response  = (state.get("response") or {}).get("answer", "")
    iteration = state.get("iteration", 0)

    # ── Guard: nothing to audit ──────────────────
    if not context or not response:
        return {
            "evaluation": {**_FALLBACK_EVALUATION, "improve": False,
                           "feedback": "Insufficient state data for audit."},
            "iteration":  iteration,
        }

    # ── Run audit ────────────────────────────────
    try:
        result = llm.invoke([
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=CRITIC_REPORT_PROMPT.format(
                context=context,
                question=question,
                response=response,
            )),
        ])
        evaluation = _parse_evaluation(result.content) or _FALLBACK_EVALUATION.copy()
    except Exception as exc:
        print(f"[critic_node] LLM call failed: {exc}")
        evaluation = _FALLBACK_EVALUATION.copy()

    # ── Termination overrides ────────────────────

    # Perfect score — no improvement needed
    if evaluation.get("grounded_score", 0) >= 9 and not evaluation.get("hallucination_detected"):
        evaluation["improve"] = False

    # Hard iteration cap
    if iteration >= 3:
        evaluation["improve"] = False
        evaluation["feedback"] = (
            evaluation.get("feedback", "").rstrip() + " (Max iterations reached.)"
        )

    return {
        "evaluation": evaluation,
        "iteration":  iteration + 1,
    }
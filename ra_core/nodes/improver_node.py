"""
improver_node.py
────────────────
Corrects the research response based on critic audit feedback.
Produces a grounded, refined answer and an internal change log.
"""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from nodes.research_node import ResearchAssistant
from prompts.improver_prompts import IMPROVER_SYSTEM_PROMPT, IMPROVER_TASK_PROMPT


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _parse_improved_response(text: str, fallback: str) -> dict:
    """
    Robustly extract the JSON object from LLM output using three strategies:
      1. Strip markdown fences → JSON parse
      2. Regex extraction of the 'answer' key
      3. Treat the raw text as a plain answer (no JSON wrapping)
    """
    # Strategy 1: clean and parse
    try:
        cleaned = re.sub(r"```json|```", "", text).strip()
        start, end = cleaned.find("{"), cleaned.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(cleaned[start:end])
            if "answer" in parsed and isinstance(parsed["answer"], str):
                return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: regex capture of the answer value
    try:
        match = re.search(r'"answer"\s*:\s*"(.*?)"(?:\s*[,}])', text, re.DOTALL)
        if match:
            answer = match.group(1).replace("\\n", "\n").replace('\\"', '"')
            if len(answer.strip()) > 20:
                return {"analysis": "", "answer": answer}
    except Exception:
        pass

    # Strategy 3: plain-text fallback (LLM returned prose, not JSON)
    if not text.strip().startswith("{"):
        return {"analysis": "", "answer": text.strip()}

    return {"analysis": "", "answer": fallback}


# ──────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────

def improver_node(state: ResearchAssistant, llm) -> dict:
    """
    Applies targeted corrections to the draft response based on critic feedback.
    Skips improvement when the current answer is already highly grounded (score ≥ 9).
    """
    question  = state.get("question") or (
        state["messages"][-1].content if state["messages"] else ""
    )
    context    = state.get("context", "")
    response   = state.get("response", {})
    evaluation = state.get("evaluation", {})
    iteration  = state.get("iteration", 0)

    current_answer = (
        response.get("answer", "") if isinstance(response, dict) else str(response)
    )

    # ── Guard: already excellent ─────────────────
    if (
        evaluation.get("grounded_score", 0) >= 9
        and not evaluation.get("hallucination_detected")
    ):
        return {"iteration": iteration + 1}

    # ── Run improvement ──────────────────────────
    try:
        result = llm.invoke([
            SystemMessage(content=IMPROVER_SYSTEM_PROMPT),
            HumanMessage(content=IMPROVER_TASK_PROMPT.format(
                context=context,
                feedback=evaluation.get("feedback", "Improve technical grounding."),
                question=question,
                current_answer=current_answer,
            )),
        ])
        parsed   = _parse_improved_response(result.content, current_answer)
        analysis = parsed.get("analysis", "")
        answer   = parsed.get("answer", "") or current_answer

        # Sanity: never return an obviously truncated result
        if len(answer.strip()) < 20:
            answer = current_answer

    except Exception as exc:
        print(f"[improver_node] LLM call failed: {exc}")
        answer   = current_answer
        analysis = "Improvement failed; retaining original response."

    return {
        "messages": [AIMessage(content=answer)],
        "response": {
            "answer":         answer,
            "internal_audit": analysis,
        },
        "iteration": iteration + 1,
    }
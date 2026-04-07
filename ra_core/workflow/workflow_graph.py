"""
workflow_graph.py
─────────────────
Assembles the LangGraph StateGraph for the research assistant pipeline.

Flow:
  START
    └─► intent_classifier ──(rejected)──► END
                           └─(verified)─► research
                                            └─(no context)─► END
                                            └─(has context)─► critic
                                                              └─(improve=True, iter<3)─► improver ──► critic
                                                              └─(improve=False / iter≥3)─────────────► END
"""

from langgraph.graph import StateGraph, START, END


# ──────────────────────────────────────────────
# Edge Predicates
# ──────────────────────────────────────────────

def _route_after_intent(state: dict) -> str:
    """Terminate early for off-topic queries; proceed to research otherwise."""
    return "end" if state.get("intent") == "rejected" else "research"


def _route_after_research(state: dict) -> str:
    """
    Only send to the critic when there is substantive context to audit.
    A context that consists only of 'NONE' or is very short is not worth auditing.
    """
    context = (state.get("context") or "").strip()
    has_content = context and "NONE" not in context and len(context) > 100
    return "critic" if has_content else "end"


def _route_after_critic(state: dict) -> str:
    """
    Decide whether to improve or exit.
    The critic node already applies the iteration cap before setting improve=False,
    so this predicate only needs to read that flag.
    """
    return "improve" if state.get("evaluation", {}).get("improve", False) else "end"


# ──────────────────────────────────────────────
# Graph Factory
# ──────────────────────────────────────────────

def create_workflow(
    ResearchAssistant,
    intent_node_func,
    research_node_func,
    critic_node_func,
    improver_node_func,
):
    """
    Build and return the RAW (uncompiled) LangGraph StateGraph.

    Compilation is intentionally deferred to the caller (main.py) so that
    a fresh checkpointer can be injected per-request inside
    stream_formatted_answer().

    Parameters
    ----------
    ResearchAssistant   : TypedDict — the shared state schema
    intent_node_func    : callable — gatekeeper node
    research_node_func  : callable — RAG synthesis node
    critic_node_func    : callable — audit / evaluation node
    improver_node_func  : callable — targeted correction node
    """
    graph = StateGraph(ResearchAssistant)

    # ── Register nodes ────────────────────────────
    graph.add_node("intent_classifier", intent_node_func)
    graph.add_node("research",          research_node_func)
    graph.add_node("critic",            critic_node_func)
    graph.add_node("improver",          improver_node_func)

    # ── Wire edges ────────────────────────────────

    # Entry point
    graph.add_edge(START, "intent_classifier")

    # Gatekeeper gate
    graph.add_conditional_edges(
        "intent_classifier",
        _route_after_intent,
        {"research": "research", "end": END},
    )

    # Research → Critic (or early exit)
    graph.add_conditional_edges(
        "research",
        _route_after_research,
        {"critic": "critic", "end": END},
    )

    # Critic → Improver (or final exit)
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"improve": "improver", "end": END},
    )

    # Improver loops back to Critic for re-audit
    graph.add_edge("improver", "critic")

    # ── Return the UNCOMPILED graph ───────────────
    # Do NOT call graph.compile() here.
    # main.py compiles it fresh with a checkpointer on every request.
    return graph
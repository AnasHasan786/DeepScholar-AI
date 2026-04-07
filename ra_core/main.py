import os
import time
from functools import partial
from typing import Dict, Any, Generator

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.postgres import PostgresSaver

from config.settings import DB_URI, LLM_MODEL_NAME, LLM_TEMPERATURE
from nodes.research_node import research_node, classify_intent_node, ResearchAssistant
from nodes.critic_node import critic_node
from nodes.improver_node import improver_node
from workflow.workflow_graph import create_workflow

# ─── Model Configuration ──────────────────────────────────────────────────────
llm_model = ChatGroq(
    model=LLM_MODEL_NAME,
    temperature=LLM_TEMPERATURE,
    max_tokens=2048,
    streaming=True,
)

# ─── Workflow Initialization ──────────────────────────────────────────────────
# create_workflow returns an UNCOMPILED StateGraph.
# We compile it fresh with a checkpointer inside stream_formatted_answer
# so that each call gets a properly scoped checkpointer context.
_raw_graph = create_workflow(
    ResearchAssistant=ResearchAssistant,
    intent_node_func=partial(classify_intent_node, llm=llm_model),
    research_node_func=partial(research_node, llm=llm_model),
    critic_node_func=partial(critic_node, llm=llm_model),
    improver_node_func=partial(improver_node, llm=llm_model),
)

# app_graph is the raw (uncompiled) graph — exported for load_conversation
# to query checkpoints directly via PostgresSaver without re-compiling.
app_graph = _raw_graph


class APIUnavailableError(Exception):
    """Raised when the upstream LLM API is unavailable or returns an auth/rate error."""
    def __init__(self, message: str, error_code: str = ""):
        super().__init__(message)
        self.error_code = error_code


def _classify_api_error(exc: Exception) -> "APIUnavailableError | None":
    """
    Inspect an exception and return an APIUnavailableError if it looks like
    an upstream API problem (auth failure, rate limit, connection error, etc.).
    Returns None if the error is unrelated.
    """
    err_str = str(exc).lower()

    api_signals = [
        "401", "403", "429", "503", "502", "500",
        "api key", "apikey", "authentication", "invalid_api_key",
        "rate limit", "ratelimit", "quota",
        "connection error", "timeout", "service unavailable",
        "groq", "openai", "anthropic",
    ]
    if any(sig in err_str for sig in api_signals):
        code = ""
        for status in ["401", "403", "429", "500", "502", "503"]:
            if status in err_str:
                code = status
                break
        return APIUnavailableError(str(exc), error_code=code)

    return None


def stream_formatted_answer(
    retriever, question: str, config: Dict[str, Any]
) -> Generator:
    """
    Streams the agentic workflow results to the Streamlit UI.
    Yields dicts for node status updates and strings for answer tokens.
    Raises APIUnavailableError if an upstream API problem is detected.
    """
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()

        # Compile the raw graph with a live checkpointer for this request.
        workflow = app_graph.compile(checkpointer=checkpointer)

        # 1. Initialize State
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "context": "",
            "intent": "init",
            "response": {},
            "sources": [],
            "evaluation": {},
            "iteration": 0,
            "retrieved_docs": retriever.invoke(question) if retriever else [],
            "web_searched": False,
        }

        # 2. Stream Node Execution
        try:
            for update in workflow.stream(
                initial_state, config=config, stream_mode="updates"
            ):
                for node_name, node_output in update.items():
                    yield {"active_node": node_name}
                    if "sources" in node_output:
                        yield {"sources": node_output["sources"]}
        except Exception as exc:
            api_err = _classify_api_error(exc)
            if api_err:
                raise api_err from exc
            raise

        # 3. Retrieve Final State Snapshot
        state_snapshot = workflow.get_state(config)
        final_state = state_snapshot.values if state_snapshot else {}

        # 4. Extract Answer from Structured Response
        response_data = final_state.get("response", {})
        answer_text = ""

        if isinstance(response_data, dict):
            answer_text = response_data.get("answer", "")

        # Fallback: Last AI Message
        if not answer_text and "messages" in final_state and final_state["messages"]:
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    answer_text = msg.content
                    break

        # 5. Token-like Streaming
        if answer_text:
            if final_state.get("intent") == "rejected":
                yield "__GATEKEEPER__"

            words = answer_text.split(" ")
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                time.sleep(0.005)
        else:
            yield (
                "⚠️ No grounded response was generated. "
                "Please ensure your query relates to the uploaded document."
            )

        # 6. Final Metadata Payload
        yield {"final_sources": final_state.get("sources", [])}
        if final_state.get("evaluation"):
            yield {"metrics": final_state["evaluation"]}
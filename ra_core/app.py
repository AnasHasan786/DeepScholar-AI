import streamlit as st
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.postgres import PostgresSaver

from main import (
    stream_formatted_answer,
    llm_model,
    DB_URI,
    app_graph,
    APIUnavailableError,
)
from rag.rag_loader import RAGLoader
from nodes.research_node import ResearchAssistant

current_dir = Path(__file__).parent 

# --- CSS + THEME ---
css_path = current_dir / "static/styles.css"
try:
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    # This helps you debug if it still fails
    st.error(f"CSS not found at {css_path}")


# --- PAGE CONFIG ---
_FAVICON = current_dir / "static/favicon.png"
st.set_page_config(
    page_title="DeepScholarAI",
    page_icon=str(_FAVICON) if _FAVICON.exists() else "🔬",
    layout="centered",
)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
_DEFAULTS = {
    "theme": "light",
    "thread_id": None,
    "message_history": [],
    "retrievers": {},
    "uploaded_pdf_name": None,
    "chat_threads": [],
    "thread_files": {},
    # Tracks the question currently being processed so we never double-append
    "_pending_question": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

theme = st.session_state["theme"]

# ─── THEME ─────────────────────────────────────────────────────────────
st.markdown(
    f"<script>(function(){{document.documentElement.setAttribute('data-theme','{theme}');}})();</script>",
    unsafe_allow_html=True,
)

# ─── NODE METADATA ────────────────────────────────────────────────────────────
_NODES = [
    ("intent_classifier", "🛡️", "Gatekeeper", "Validating query scope"),
    ("research", "🔍", "Researcher", "Extracting &amp; synthesising evidence"),
    ("critic", "⚖️", "Critic", "Auditing for hallucinations"),
    ("improver", "🛠️", "Editor", "Refining &amp; grounding the answer"),
]


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def generate_thread_id() -> str:
    return str(uuid.uuid4())


def reset_chat():
    st.session_state["thread_id"] = None
    st.session_state["message_history"] = []
    st.session_state["uploaded_pdf_name"] = None
    st.session_state["_pending_question"] = None


def retrieve_all_threads():
    threads_with_ts, thread_to_file = [], {}
    try:
        with PostgresSaver.from_conn_string(DB_URI) as cp:
            for ct in cp.list(None):
                tid = ct.config.get("configurable", {}).get("thread_id")
                if not tid:
                    continue
                file_name = (ct.metadata or {}).get("file_name")
                ts = None
                if isinstance(ct.checkpoint, dict):
                    ts = ct.checkpoint.get("ts") or ct.checkpoint.get("v", {}).get("ts")
                threads_with_ts.append((tid, ts or datetime.min.isoformat()))
                if file_name and file_name != "None":
                    thread_to_file[tid] = file_name

        threads_with_ts.sort(key=lambda x: x[1], reverse=True)
        seen, ordered = set(), []
        for tid, _ in threads_with_ts:
            if tid not in seen:
                ordered.append(tid)
                seen.add(tid)
        return ordered, thread_to_file
    except Exception as e:
        print(f"[retrieve_all_threads] {e}")
        return [], {}


def load_conversation(thread_id: str) -> list:
    """
    Load conversation history by reading the checkpoint directly from
    PostgresSaver — no graph compilation required.
    """
    try:
        with PostgresSaver.from_conn_string(DB_URI) as cp:
            cp.setup()
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                }
            }
            checkpoint_tuple = cp.get_tuple(config)
            if not checkpoint_tuple:
                return []

            messages = (
                checkpoint_tuple.checkpoint
                .get("channel_values", {})
                .get("messages", [])
            )
            if not messages:
                return []

            history, i = [], 0
            while i < len(messages):
                msg = messages[i]
                if isinstance(msg, HumanMessage) and msg.content.strip():
                    last_ai, j = None, i + 1
                    while j < len(messages) and not isinstance(messages[j], HumanMessage):
                        if (
                            isinstance(messages[j], AIMessage)
                            and messages[j].content.strip()
                        ):
                            last_ai = messages[j].content.strip()
                        j += 1
                    history.append({"role": "user", "content": msg.content.strip()})
                    if last_ai:
                        history.append({"role": "assistant", "content": last_ai})
                    i = j
                else:
                    i += 1
            return history
    except Exception as e:
        print(f"[load_conversation] {e}")
        return []


def _pipeline_html(
    completed: list, active: "str | None", pending_start: bool = False
) -> str:
    rows = []
    for node_id, icon, label, desc in _NODES:
        if node_id == active:
            cls = "pipeline-node active"
        elif node_id in completed:
            cls = "pipeline-node done"
        else:
            cls = "pipeline-node pending"

        check = "✓" if node_id in completed else ""
        spinner = (
            '<span class="pipeline-spinner"></span>' if node_id == active else ""
        )

        rows.append(
            f'<div class="{cls}">'
            f'<div class="pipeline-dot"></div>'
            f'<span class="pipeline-node-text">'
            f'{icon} <strong>{label}</strong>'
            f'<span class="pipeline-desc"> — {desc}</span>'
            f'</span>'
            f'{spinner}'
            f'<span class="pipeline-check">{check}</span>'
            f'</div>'
        )

    return f'<div class="pipeline-wrap">{"".join(rows)}</div>'


def _error_html(exc: APIUnavailableError) -> str:
    code = exc.error_code
    if code in ("401", "403"):
        title = "API Authentication Failed"
        desc = "The LLM service rejected the request due to an invalid or expired API key. Check your configuration."
    elif code == "429":
        title = "Rate Limit Reached"
        desc = "Too many requests sent to the LLM service. Please wait a moment before trying again."
    elif code in ("500", "502", "503"):
        title = "LLM Service Unavailable"
        desc = "The upstream AI service is experiencing issues — usually temporary. Please retry in a few seconds."
    else:
        title = "AI Service Unreachable"
        desc = "Could not connect to the LLM service. Verify your API key and network connection."
    badge = f'<span class="err-code">HTTP {code}</span>' if code else ""
    return (
        f'<div class="err-banner">'
        f'<div class="err-icon">⚠️</div>'
        f'<div>'
        f'<div class="err-title">{title}</div>'
        f'<div class="err-desc">{desc}{"<br/>" + badge if badge else ""}</div>'
        f'</div>'
        f'</div>'
    )


# ─── SYNC DB THREADS ─────────────────────────────────────────────────────────
db_threads, db_titles = retrieve_all_threads()
if not st.session_state["chat_threads"]:
    st.session_state["chat_threads"] = db_threads
if not st.session_state["thread_files"]:
    st.session_state["thread_files"] = db_titles

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown(
        '<div class="sb-brand">'
        '<div class="sb-brand-icon">⟁</div>'
        '<div>'
        '<span class="sb-brand-name">DeepScholar</span>'
        '<span class="sb-brand-tag">Research Intelligence</span>'
        '</div>'
        '</div>'
        '<div class="sb-divider"></div>',
        unsafe_allow_html=True,
    )

    current_tid = st.session_state["thread_id"]
    has_retriever = current_tid in st.session_state["retrievers"]

    if has_retriever:
        title = st.session_state["thread_files"].get(current_tid) or "Active Session"
        st.markdown(
            f'<div class="kb-card">'
            f'<span class="kb-card-label">◈ Knowledge Base Active</span>'
            f'<span class="kb-card-title" title="{title}">⬡ {title}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("⊘  Reset Knowledge Base", use_container_width=True, key="reset_kb"):
            st.session_state["retrievers"].pop(current_tid, None)
            reset_chat()
            st.rerun()
    else:
        st.markdown(
            '<span class="sb-label">⬆ Upload Research Paper</span>',
            unsafe_allow_html=True,
        )
        uploaded_pdfs = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True,
            key=f"uploader_{st.session_state['thread_id']}",
            label_visibility="collapsed",
        )
        if uploaded_pdfs:
            with st.spinner(f"Indexing {len(uploaded_pdfs)} paper(s)…"):
                if not st.session_state["thread_id"]:
                    st.session_state["thread_id"] = generate_thread_id()
                tid = st.session_state["thread_id"]
                if tid not in st.session_state["chat_threads"]:
                    st.session_state["chat_threads"].insert(0, tid)

                loader = RAGLoader(llm_model)
                retriever = loader.load_multiple_pdfs(
                    [(p.getvalue(), p.name) for p in uploaded_pdfs]
                )
                st.session_state["retrievers"][tid] = retriever

                if len(uploaded_pdfs) > 1:
                    names = [p.name.rsplit(".pdf", 1)[0] for p in uploaded_pdfs]
                    clean_title = f"Comparison: {names[0]} +{len(names) - 1} more"
                else:
                    clean_title = uploaded_pdfs[0].name.rsplit(".pdf", 1)[0]

                st.session_state["uploaded_pdf_name"] = clean_title
                st.session_state["thread_files"][tid] = clean_title

                try:
                    with PostgresSaver.from_conn_string(DB_URI) as cp:
                        cp.setup()
                        cp.put(
                            {"configurable": {"thread_id": tid}},
                            checkpoint={
                                "v": 1,
                                "ts": datetime.now().isoformat(),
                                "channel_values": {},
                                "channel_versions": {},
                                "versions_seen": {},
                                "pending_sends": [],
                            },
                            metadata={"file_name": clean_title, "source": "uploader"},
                            new_versions={},
                        )
                except Exception as e:
                    print(f"[DB metadata write] {e}")

                st.rerun()

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    if st.button("✦  New Analysis", key="new_analysis", use_container_width=True):
        reset_chat()
        st.rerun()

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sb-divider"></div>'
        '<span class="sb-label">◷ Recent Analyses</span>',
        unsafe_allow_html=True,
    )

    search_query = st.text_input(
        "Search",
        placeholder="⌕  Filter by paper title…",
        label_visibility="collapsed",
        key="history_search",
    ).lower()

    all_threads = st.session_state["chat_threads"]
    filtered = (
        [
            tid
            for tid in all_threads
            if search_query
            in (
                st.session_state["thread_files"].get(tid) or db_titles.get(tid) or ""
            ).lower()
        ]
        if search_query
        else all_threads
    )

    st.markdown('<div style="height:2px;"></div>', unsafe_allow_html=True)

    if filtered:
        for tid in filtered[:15]:
            full_title = (
                st.session_state["thread_files"].get(tid)
                or db_titles.get(tid)
                or f"Analysis {tid[:5]}"
            )
            label = (full_title[:27] + "…") if len(full_title) > 27 else full_title
            is_active = st.session_state["thread_id"] == tid
            icon = "▸" if is_active else "○"
            btn_key = f"thread_{tid}"
            if st.button(f"{icon}  {label}", key=btn_key, use_container_width=True):
                st.session_state["thread_id"] = tid
                st.session_state["message_history"] = load_conversation(tid)
                st.session_state["uploaded_pdf_name"] = full_title
                st.session_state["_pending_question"] = None
                st.rerun()
    else:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">⬡</div>'
            '<div class="empty-state-title">No analyses yet</div>'
            '<div class="empty-state-text">Upload a paper above<br/>to begin your first analysis.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# ─── CHAT HISTORY ────────────────────────────────────────────────────────────
user_question = st.chat_input("Ask anything about your paper(s)…")
show_home = (
    not st.session_state["message_history"]
    and not user_question
    and not st.session_state["_pending_question"]
)

# Render only confirmed history — the pending question is shown via placeholder below,
# never stored here until the pipeline succeeds.
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── EXECUTE PIPELINE ────────────────────────────────────────────────────────
if user_question:
    tid = st.session_state["thread_id"]
    retriever = st.session_state["retrievers"].get(tid) if tid else None

    if not retriever:
        st.warning("⚠️ Please upload a research paper first to begin your analysis.")
    else:
        question = user_question.strip()

        # Guard: Streamlit re-executes the whole script on every streaming
        # update. If this question is already in-flight, stop here so we
        # don't re-enter the pipeline or append to history a second time.
        if st.session_state["_pending_question"] == question:
            st.stop()

        # Mark as in-flight BEFORE any yield so reruns hit the guard above
        st.session_state["_pending_question"] = question

        user_msg_ph = st.empty()   # wiped on error so no orphaned user bubble
        asst_chat_ph = st.empty()  # pipeline / streaming answer placeholder

        completed: list[str] = []
        active: "str | None" = None
        full_text = ""
        is_streaming = False

        # Render user bubble and the initial pending pipeline tracker
        with user_msg_ph.container():
            with st.chat_message("user"):
                st.markdown(question)

        with asst_chat_ph.container():
            with st.chat_message("assistant"):
                st.markdown(
                    _pipeline_html(completed, active, pending_start=True),
                    unsafe_allow_html=True,
                )

        try:
            for chunk in stream_formatted_answer(
                retriever,
                question,
                {
                    "configurable": {"thread_id": tid},
                    "metadata": {
                        "file_name": st.session_state.get("uploaded_pdf_name")
                    },
                },
            ):
                if isinstance(chunk, dict):
                    if "active_node" in chunk:
                        if active and active not in completed:
                            completed.append(active)
                        active = chunk["active_node"]
                        is_streaming = False
                        with asst_chat_ph.container():
                            with st.chat_message("assistant"):
                                st.markdown(
                                    _pipeline_html(completed, active),
                                    unsafe_allow_html=True,
                                )

                elif isinstance(chunk, str):
                    if not is_streaming:
                        if active and active not in completed:
                            completed.append(active)
                        is_streaming = True

                    if chunk == "__GATEKEEPER__":
                        continue

                    full_text += chunk
                    with asst_chat_ph.container():
                        with st.chat_message("assistant"):
                            st.markdown(
                                _pipeline_html(completed, None),
                                unsafe_allow_html=True,
                            )
                            st.markdown(full_text + "▌")

            # ── Stream finished successfully ──────────────────────────────
            with asst_chat_ph.container():
                with st.chat_message("assistant"):
                    st.markdown(
                        _pipeline_html(completed, None),
                        unsafe_allow_html=True,
                    )
                    st.markdown(full_text)

            # Append to history ONLY after confirmed success
            st.session_state["message_history"].append({"role": "user", "content": question})
            st.session_state["message_history"].append({"role": "assistant", "content": full_text})
            st.session_state["_pending_question"] = None  # clear in-flight flag

            st.rerun()

        except APIUnavailableError as api_err:
            # Clear both placeholders — user bubble disappears, nothing in history
            user_msg_ph.empty()
            asst_chat_ph.empty()
            st.session_state["_pending_question"] = None
            with st.chat_message("assistant"):
                st.markdown(_error_html(api_err), unsafe_allow_html=True)

        except Exception as e:
            # Clear both placeholders — user bubble disappears, nothing in history
            user_msg_ph.empty()
            asst_chat_ph.empty()
            st.session_state["_pending_question"] = None
            with st.chat_message("assistant"):
                st.markdown(
                    f'<div class="err-banner">'
                    f'<div class="err-icon">⚠️</div>'
                    f'<div>'
                    f'<div class="err-title">Unexpected Error</div>'
                    f'<div class="err-desc">An error occurred during analysis. Please try again.</div>'
                    f'<span class="err-code">{type(e).__name__}: {str(e)}</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ─── HOME SCREEN ─────────────────────────────────────────────────────────────
elif show_home:
    st.markdown(
        '<div class="home-wrap">'
        '<div class="home-hero">'
        '<div class="home-logo-ring">⟁</div>'
        '<div class="home-badge">✦ Multi-Agent Research Pipeline</div>'
        '<h1 class="home-title">Deep<span>Scholar</span>AI</h1>'
        '<p class="home-sub">Upload any research paper and get grounded, hallucination-free answers — powered by four specialised AI agents working in concert.</p>'
        '</div>'

        '<div class="home-pipeline-card">'
        '<div class="home-pipeline-label">How It Works</div>'
        '<div class="home-pipeline">'

        '<div class="home-step">'
        '<div class="home-step-num">01</div>'
        '<div class="home-step-icon">🛡️</div>'
        '<div class="home-step-name">Gatekeeper</div>'
        '<div class="home-step-desc">Validates your query is within research scope</div>'
        '</div>'

        '<div class="home-arrow">→</div>'

        '<div class="home-step">'
        '<div class="home-step-num">02</div>'
        '<div class="home-step-icon">🔍</div>'
        '<div class="home-step-name">Researcher</div>'
        '<div class="home-step-desc">Extracts and synthesises evidence from your paper</div>'
        '</div>'

        '<div class="home-arrow">→</div>'

        '<div class="home-step">'
        '<div class="home-step-num">03</div>'
        '<div class="home-step-icon">⚖️</div>'
        '<div class="home-step-name">Critic</div>'
        '<div class="home-step-desc">Audits every claim for hallucinations</div>'
        '</div>'

        '<div class="home-arrow">→</div>'

        '<div class="home-step">'
        '<div class="home-step-num">04</div>'
        '<div class="home-step-icon">🛠️</div>'
        '<div class="home-step-name">Editor</div>'
        '<div class="home-step-desc">Refines &amp; grounds the final answer</div>'
        '</div>'

        '<div class="home-arrow">→</div>'

        '<div class="home-step home-step-final">'
        '<div class="home-step-num">✓</div>'
        '<div class="home-step-icon">✅</div>'
        '<div class="home-step-name">Answer</div>'
        '<div class="home-step-desc">Verified, grounded response</div>'
        '</div>'

        '</div>'  # .home-pipeline
        '</div>'  # .home-pipeline-card

        '<div class="home-features">'

        '<div class="home-feature">'
        '<div class="home-feature-icon">📄</div>'
        '<div class="home-feature-body">'
        '<div class="home-feature-title">Multi-PDF Support</div>'
        '<div class="home-feature-desc">Upload one or more papers and ask comparative questions across all of them simultaneously.</div>'
        '</div>'
        '</div>'

        '<div class="home-feature">'
        '<div class="home-feature-icon">🔒</div>'
        '<div class="home-feature-body">'
        '<div class="home-feature-title">Hallucination-Free</div>'
        '<div class="home-feature-desc">Every claim is grounded in your documents and independently audited by the Critic agent.</div>'
        '</div>'
        '</div>'

        '<div class="home-feature">'
        '<div class="home-feature-icon">💬</div>'
        '<div class="home-feature-body">'
        '<div class="home-feature-title">Persistent History</div>'
        '<div class="home-feature-desc">All analyses are saved to the database — resume any past conversation from the sidebar.</div>'
        '</div>'
        '</div>'

        '</div>'  # .home-features

        '<div class="home-cta">'
        '<div class="home-cta-arrow">←</div>'
        '<div class="home-cta-text">Upload a PDF in the sidebar to begin your first analysis</div>'
        '</div>'

        '</div>',  # .home-wrap
        unsafe_allow_html=True,
    )